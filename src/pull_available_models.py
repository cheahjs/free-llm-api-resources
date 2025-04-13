#!/usr/bin/env python3

from collections import defaultdict
import logging
import json
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
from google.cloud import cloudquotas_v1
from mistralai import Mistral
from concurrent.futures import ThreadPoolExecutor
import time

from data import (
    MODEL_TO_NAME_MAPPING,
    HYPERBOLIC_IGNORED_MODELS,
    LAMBDA_IGNORED_MODELS,
    OPENROUTER_IGNORED_MODELS,
)


load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Global clients
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
last_mistral_request_time = 0


def create_logger(provider_name):
    logger = logging.getLogger(provider_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"{provider_name}: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


MISSING_MODELS = set()


def get_model_name(id):
    id = id.lower()
    if id in MODEL_TO_NAME_MAPPING:
        return MODEL_TO_NAME_MAPPING[id]
    MISSING_MODELS.add(id)
    return id


def get_groq_limits_for_stt_model(model_id, logger):
    logger.info(f"Getting limits for STT model {model_id}...")
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={
            "Authorization": f'Bearer {os.environ["GROQ_API_KEY"]}',
        },
        data={
            "model": model_id,
        },
        files={
            "file": open(os.path.join(script_dir, "1-second-of-silence.mp3"), "rb"),
        },
    )
    r.raise_for_status()
    audio_seconds_per_minute = int(r.headers["x-ratelimit-limit-audio-seconds"])
    rpd = int(r.headers["x-ratelimit-limit-requests"])
    return {
        "audio-seconds/minute": audio_seconds_per_minute,
        "requests/day": rpd,
    }


def get_groq_limits_for_model(model_id, script_dir, logger):
    if "whisper" in model_id:
        return get_groq_limits_for_stt_model(model_id, logger)
    if "tts" in model_id:
        return None
    logger.info(f"Getting limits for chat model {model_id}...")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f'Bearer {os.environ["GROQ_API_KEY"]}',
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi!"}],
            "max_tokens": 1,
            "stream": True,
        },
        stream=True,
    )
    try:
        r.raise_for_status()
        rpd = int(r.headers["x-ratelimit-limit-requests"])
        tpm = int(r.headers["x-ratelimit-limit-tokens"])
        return {"requests/day": rpd, "tokens/minute": tpm}
    except Exception as e:
        logger.error(f"Failed to get limits for model {model_id}: {e}")
        logger.error(r.text)
        return {"requests/day": "Unknown", "tokens/minute": "Unknown"}


def fetch_groq_models(logger):
    logger.info("Fetching Groq models...")
    r = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={
            "Authorization": f'Bearer {os.environ["GROQ_API_KEY"]}',
            "Content-Type": "application/json",
        },
    )
    r.raise_for_status()
    models = r.json()["data"]
    logger.debug(json.dumps(models, indent=4))
    ret_models = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for model in models:
            future = executor.submit(
                get_groq_limits_for_model, model["id"], script_dir, logger
            )
            futures.append((model, future))

        for model, future in futures:
            limits = future.result()
            if limits is None:
                continue
            ret_models.append(
                {
                    "id": model["id"],
                    "name": get_model_name(model["id"]),
                    "limits": limits,
                }
            )
    ret_models = sorted(ret_models, key=lambda x: x["name"])
    return ret_models


def fetch_openrouter_models(logger):
    logger.info("Fetching OpenRouter models...")
    r = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={
            "Content-Type": "application/json",
        },
    )
    r.raise_for_status()
    models = r.json()["data"]
    logger.info(f"Fetched {len(models)} models from OpenRouter")
    ret_models = []
    for model in models:
        pricing = float(model.get("pricing", {}).get("completion", "1")) + float(
            model.get("pricing", {}).get("prompt", "1")
        )
        if pricing != 0:
            continue
        if ":free" not in model["id"]:
            continue
        if model["id"].lower() in OPENROUTER_IGNORED_MODELS:
            logger.debug(f"Ignoring model {model['id']}")
            continue
        ret_models.append(
            {
                "id": model["id"],
                "name": get_model_name(model["id"]),
                "limits": {
                    "requests/minute": 20,
                    "requests/day": 50,
                },
            }
        )
    ret_models = sorted(ret_models, key=lambda x: x["name"])
    return ret_models


def fetch_cloudflare_models(logger):
    logger.info("Fetching Cloudflare models...")
    r = requests.get(
        f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai/models/search?search=Text+Generation",
        headers={
            "Authorization": f'Bearer {os.environ["CLOUDFLARE_API_KEY"]}',
            "Content-Type": "application/json",
        },
    )
    r.raise_for_status()
    models = r.json()["result"]
    logger.info(f"Fetched {len(models)} models from Cloudflare")
    ret_models = []
    for model in models:
        ret_models.append(
            {
                "id": model["name"],
                "name": get_model_name(model["name"]),
            }
        )
    ret_models = sorted(ret_models, key=lambda x: x["name"])
    return ret_models


def fetch_ovh_models(logger):
    logger.info("Fetching OVH models...")
    r = requests.get(
        "https://endpoints-backend.ai.cloud.ovh.net/rest/v1/models_v2",
        params={"select": "*", "order": "id.desc", "offset": "0", "limit": "100"},
        headers={
            "accept": "*/*",
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
            "accept-profile": "public",
            "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogImFub24iLAogICJpc3MiOiAic3VwYWJhc2UiLAogICJpYXQiOiAxNzEwNzE2NDAwLAogICJleHAiOiAxODY4NDgyODAwCn0.Jty_eO4oWqLm4Lx_LfbpRW5WESXYXtT2humbBq2Pal8",
            "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogImFub24iLAogICJpc3MiOiAic3VwYWJhc2UiLAogICJpYXQiOiAxNzEwNzE2NDAwLAogICJleHAiOiAxODY4NDgyODAwCn0.Jty_eO4oWqLm4Lx_LfbpRW5WESXYXtT2humbBq2Pal8",
            "priority": "u=1, i",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "x-client-info": "supabase-js-web/2.39.7",
        },
    )
    r.raise_for_status()
    models = list(filter(lambda x: x["available"] and "LLM" in x["category"], r.json()))
    logger.info(f"Fetched {len(models)} models from OVH")
    ret_models = []
    for model in models:
        ret_models.append(
            {
                "id": model["name"],
                "name": get_model_name(model["name"]),
                "limits": {
                    "requests/minute": 12,
                },
            }
        )
    ret_models = sorted(ret_models, key=lambda x: x["name"])
    return ret_models


def fetch_hyperbolic_models(logger):
    logger.info("Fetching Hyperbolic models from API...")
    r = requests.get(
        "https://api.hyperbolic.xyz/v1/models",
        headers={
            "accept": "application/json",
            "authorization": f"Bearer {os.environ['HYPERBOLIC_API_KEY']}",
        },
    )
    r.raise_for_status()
    models = r.json()["data"]
    logger.info(f"Fetched {len(models)} models from Hyperbolic's API")
    ret_models = []
    for model in models:
        if model["id"] in HYPERBOLIC_IGNORED_MODELS:
            logger.debug(f"Ignoring model {model['id']}")
            continue
        ret_models.append(
            {
                "id": model["id"],
                "name": get_model_name(model["id"]),
                "limits": {
                    "requests/minute": 60,
                },
            }
        )
    logger.debug(json.dumps(ret_models, indent=4))
    return sorted(ret_models, key=lambda x: x["name"])


def fetch_github_models(logger):
    logger.info("Fetching GitHub models...")
    r = requests.get(
        "https://github.com/marketplace/models",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-requested-with": "XMLHttpRequest",
        },
    )
    r.raise_for_status()
    models = r.json()
    logger.info(f"Fetched {len(models)} models from GitHub")
    ret_models = []
    for model in models:
        ret_models.append(
            {
                "id": model["name"],
                "name": model["friendly_name"],
            }
        )
    ret_models = sorted(ret_models, key=lambda x: x["name"])
    return ret_models


def fetch_gemini_limits(logger):
    logger.info("Fetching Gemini limits...")
    client = cloudquotas_v1.CloudQuotasClient()
    request = cloudquotas_v1.ListQuotaInfosRequest(
        parent=f"projects/{os.environ["GCP_PROJECT_ID"]}/locations/global/services/generativelanguage.googleapis.com"
    )
    pager = client.list_quota_infos(request=request)
    models = defaultdict(dict)
    for quota in pager:
        if (
            quota.metric
            == "generativelanguage.googleapis.com/generate_content_free_tier_input_token_count"
        ):
            for dimension in quota.dimensions_infos:
                models[dimension.dimensions.get("model")][
                    f"tokens/{quota.refresh_interval}"
                ] = dimension.details.value
        elif (
            quota.metric
            == "generativelanguage.googleapis.com/generate_content_free_tier_requests"
        ):
            for dimension in quota.dimensions_infos:
                models[dimension.dimensions.get("model")][
                    f"requests/{quota.refresh_interval}"
                ] = dimension.details.value
        elif quota.metric == "generativelanguage.googleapis.com/embed_text_requests":
            for dimension in quota.dimensions_infos:
                models["project-embedding"][f"requests/{quota.refresh_interval}"] = (
                    dimension.details.value
                )
        elif (
            quota.metric
            == "generativelanguage.googleapis.com/batch_embed_text_requests"
        ):
            for dimension in quota.dimensions_infos:
                models["project-embedding"][
                    f"batch requests/{quota.refresh_interval}"
                ] = dimension.details.value
    logger.debug(json.dumps(models, indent=4))
    return models


def fetch_lambda_models(logger):
    logger.info("Fetching Lambda Labs models...")
    r = requests.get(
        "https://api.lambdalabs.com/v1/models",
        headers={
            "Authorization": f"Bearer {os.environ['LAMBDA_API_KEY']}",
        },
    )
    r.raise_for_status()
    models = r.json()["data"]
    logger.info(f"Fetched {len(models)} models from Lambda Labs")
    ret_models = []
    for model in models:
        if model["id"] in LAMBDA_IGNORED_MODELS:
            logger.debug(f"Ignoring model {model['id']}")
            continue
        ret_models.append(
            {
                "id": model["id"],
                "name": get_model_name(model["id"]),
            }
        )
    ret_models = sorted(ret_models, key=lambda x: x["name"])
    return ret_models


def rate_limited_mistral_chat(client, **kwargs):
    global last_mistral_request_time

    # Ensure at least 1 second between requests
    current_time = time.time()
    time_since_last = current_time - last_mistral_request_time
    if time_since_last < 1:
        time.sleep(1 - time_since_last)

    response = client.chat.complete(**kwargs)
    last_mistral_request_time = time.time()
    return response


def fetch_samba_models(logger):
    logger.info("Fetching SambaNova models...")
    r = requests.get("https://cloud.sambanova.ai/api/pricing")
    r.raise_for_status()
    models = r.json()["prices"]
    logger.info(f"Fetched {len(models)} models from SambaNova")
    ret_models = []
    for model in models:
        ret_models.append(
            {
                "id": model["model_id"],
                "name": model["model_name"],
            }
        )
    ret_models = sorted(ret_models, key=lambda x: x["name"])
    return ret_models


def fetch_scaleway_models(logger):
    logger.info("Fetching Scaleway models...")
    r = requests.get(
        "https://api.scaleway.ai/v1/models",
        headers={"Authorization": f"Bearer {os.environ['SCALEWAY_API_KEY']}"},
    )
    r.raise_for_status()
    models = r.json()["data"]
    logger.info(f"Fetched {len(models)} models from Scaleway")
    ret_models = []
    for model in models:
        ret_models.append(
            {
                "id": model["id"],
                "name": get_model_name(model["id"]),
            }
        )
    ret_models = sorted(ret_models, key=lambda x: x["name"])
    logger.info("Fetching Scaleway rate limits")
    r = requests.get(
        "https://www.scaleway.com/en/docs/generative-apis/reference-content/rate-limits/"
    )
    r.raise_for_status()
    # Extract <main> content
    soup = BeautifulSoup(r.text, "html.parser")
    body = str(soup.find("main"))
    prompt = f"""
Here is the web page to extract data from:
```html
{body}
```
    """
    logger.info("Extracting model rate limits from the provided web page...")
    chat_response = rate_limited_mistral_chat(
        mistral_client,
        model="mistral-large-latest",
        messages=[
            {
                "role": "system",
                "content": """Extract the model rate limits as only integers from the provided web page into JSON of the following format:
```json
{
  "model name here": {
    "requests/minute": 10,
    "tokens/minute": 100
  }
}
```
ONLY OUTPUT JSON!""",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={
            "type": "json_object",
        },
        temperature=0,
    )
    logger.info(chat_response.choices[0].message.content)
    extracted_data = json.loads(chat_response.choices[0].message.content)
    for model in extracted_data:
        for m in ret_models:
            if m["id"] == model:
                m["limits"] = extracted_data[model]
                break

    return ret_models


def fetch_chutes_models(logger):
    logger.info("Fetching Chutes models...")
    r = requests.get(
        "https://api.chutes.ai/chutes/?include_public=true&limit=1000",
        headers={
            "Content-Type": "application/json",
        },
    )
    r.raise_for_status()
    models = r.json()["items"]
    logger.info(f"Fetched {len(models)} models from Chutes")

    # Filter for free models based on per_million_token price
    free_models = []
    for model in models:
        price_info = model.get("current_estimated_price", {})
        # Check if per_million_tokens field exists and is set to 0 for USD
        if price_info.get("per_million_tokens", {}).get("usd", 1) == 0:
            model_name = model.get("name", "Unknown model")
            free_models.append(
                {
                    "id": model_name,
                    "name": get_model_name(model_name),
                    "description": model.get("tagline", ""),
                }
            )

    logger.info(f"Found {len(free_models)} free models from Chutes")
    return sorted(free_models, key=lambda x: x["name"])


def get_human_limits(model):
    if "limits" not in model:
        return ""
    limits = model["limits"]
    return "<br>".join([f"{value:,} {key}" for key, value in limits.items()])


def main():
    logger = create_logger("Main")
    groq_logger = create_logger("Groq")
    openrouter_logger = create_logger("OpenRouter")
    google_ai_studio_logger = create_logger("Google AI Studio")
    ovh_logger = create_logger("OVH")
    cloudflare_logger = create_logger("Cloudflare")
    github_logger = create_logger("GitHub")
    hyperbolic_logger = create_logger("Hyperbolic")
    # lambda_logger = create_logger("Lambda Labs")
    samba_logger = create_logger("SambaNova")
    scaleway_logger = create_logger("Scaleway")
    chutes_logger = create_logger("Chutes")

    fetch_concurrently = os.getenv("FETCH_CONCURRENTLY", "false").lower() == "true"

    if fetch_concurrently:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(fetch_gemini_limits, google_ai_studio_logger),
                executor.submit(fetch_openrouter_models, openrouter_logger),
                executor.submit(fetch_hyperbolic_models, hyperbolic_logger),
                executor.submit(fetch_ovh_models, ovh_logger),
                executor.submit(fetch_cloudflare_models, cloudflare_logger),
                executor.submit(fetch_github_models, github_logger),
                executor.submit(fetch_samba_models, samba_logger),
                executor.submit(fetch_scaleway_models, scaleway_logger),
                executor.submit(fetch_chutes_models, chutes_logger),
            ]
            (
                gemini_models,
                openrouter_models,
                hyperbolic_models,
                ovh_models,
                cloudflare_models,
                github_models,
                samba_models,
                scaleway_models,
                chutes_models,
            ) = [f.result() for f in futures]

            # Fetch groq models after others complete
            groq_models = fetch_groq_models(groq_logger)
    else:
        gemini_models = fetch_gemini_limits(google_ai_studio_logger)
        openrouter_models = fetch_openrouter_models(openrouter_logger)
        hyperbolic_models = fetch_hyperbolic_models(hyperbolic_logger)
        ovh_models = fetch_ovh_models(ovh_logger)
        cloudflare_models = fetch_cloudflare_models(cloudflare_logger)
        github_models = fetch_github_models(github_logger)
        # lambda_models = fetch_lambda_models(lambda_logger)
        samba_models = fetch_samba_models(samba_logger)
        scaleway_models = fetch_scaleway_models(scaleway_logger)
        chutes_models = fetch_chutes_models(chutes_logger)
        groq_models = fetch_groq_models(groq_logger)

    table = """<table>
    <thead>
        <tr>
            <th>Provider</th>
            <th>Provider Limits/Notes</th>
            <th>Model Name</th>
            <th>Model Limits</th>
        </tr>
    </thead>
    <tbody>
"""

    for idx, model in enumerate(openrouter_models):
        table += "<tr>"

        if idx == 0:
            table += f'<td rowspan="{len(openrouter_models)}">'
            table += '<a href="https://openrouter.ai" target="_blank">OpenRouter</a>'
            table += "</td>"
            table += f'<td rowspan="{len(openrouter_models)}"><a href="https://openrouter.ai/docs/api-reference/limits" target="_blank">{get_human_limits(model)}<br>1000 requests/day with $10 credit balance</a></td>'

        table += f"<td><a href='https://openrouter.ai/{model['id']}' target='_blank'>{model['name']}</a></td>"
        if idx == 0:
            table += f'<td rowspan="{len(openrouter_models)}">Shared Quota</td>'
        table += "</tr>\n"

    gemini_text_models = [
        {
            "id": "gemini-2.5-pro-exp-03-25",
            "name": "Gemini 2.5 Pro (Experimental)",
            "limits": gemini_models["gemini-2.0-pro-exp"],
        },
        {
            "id": "gemini-2.0-flash",
            "name": "Gemini 2.0 Flash",
            "limits": gemini_models["gemini-2.0-flash"],
        },
        {
            "id": "gemini-2.0-flash-lite",
            "name": "Gemini 2.0 Flash-Lite",
            "limits": gemini_models["gemini-2.0-flash-lite"],
        },
        {
            "id": "gemini-2.0-flash-exp",
            "name": "Gemini 2.0 Flash (Experimental)",
            "limits": gemini_models["gemini-2.0-flash-exp"],
        },
        {
            "id": "gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "limits": gemini_models["gemini-1.5-flash"],
        },
        {
            "id": "gemini-1.5-flash-8b",
            "name": "Gemini 1.5 Flash-8B",
            "limits": gemini_models["gemini-1.5-flash-8b"],
        },
        {
            "id": "gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "limits": gemini_models["gemini-1.5-pro"],
        },
        {
            "id": "learnlm-1.5-pro-experimental",
            "name": "LearnLM 1.5 Pro (Experimental)",
            "limits": gemini_models["learnlm-1.5-pro-experimental"],
        },
        {
            "id": "gemma-3-27b-it",
            "name": "Gemma 3 27B Instruct",
            "limits": gemini_models["gemma-3-27b"],
        },
        {
            "id": "gemma-3-12b-it",
            "name": "Gemma 3 12B Instruct",
            "limits": gemini_models["gemma-3-12b"],
        },
        {
            "id": "gemma-3-4b-it",
            "name": "Gemma 3 4B Instruct",
            "limits": gemini_models["gemma-3-4b"],
        },
        {
            "id": "gemma-3-1b-it",
            "name": "Gemma 3 1B Instruct",
            "limits": gemini_models["gemma-3-1b"],
        },
    ]
    gemini_embedding_models = [
        {
            "id": "text-embedding-004",
            "name": "text-embedding-004",
            "limits": gemini_models["project-embedding"],
        },
        {
            "id": "embedding-001",
            "name": "embedding-001",
            "limits": gemini_models["project-embedding"],
        },
    ]

    for idx, model in enumerate(gemini_text_models):
        table += "<tr>"
        if idx == 0:
            table += f'<td rowspan="{len(gemini_text_models) + len(gemini_embedding_models)}">'
            table += '<a href="https://aistudio.google.com" target="_blank">Google AI Studio</a>'
            table += "</td>"
            table += f'<td rowspan="{len(gemini_text_models) + len(gemini_embedding_models)}">Data is used for training (when used outside of the UK/CH/EEA/EU).</td>'
        table += f"<td>{model['name']}</td>"
        table += f"<td>{get_human_limits(model)}</td>"
        table += "</tr>\n"

    for idx, model in enumerate(gemini_embedding_models):
        table += "<tr>"
        table += f"<td>{model['name']}</td>"
        if idx == 0:
            table += f'<td rowspan="{len(gemini_embedding_models)}">{get_human_limits(model)}<br>100 content/batch<br>Shared Quota</td>'
        table += "</tr>\n"

    table += """<tr>
        <td><a href="https://build.nvidia.com/explore/discover">NVIDIA NIM</a></td>
        <td>Phone number verification required.<br>Models tend to be context window limited.</td>
        <td><a href="https://build.nvidia.com/models" target="_blank">Various open models</a></td>
        <td>40 requests/minute</td>
    </tr>"""

    table += """<tr>
        <td><a href="https://console.mistral.ai/" target="_blank">Mistral (La Plateforme)</a></td>
        <td>Free tier (Experiment plan) requires opting into data training, requires phone number verification.</td>
        <td><a href="https://docs.mistral.ai/getting-started/models/models_overview/" target="_blank">Open and Proprietary Mistral models</a></td>
        <td>1 request/second<br>500,000 tokens/minute<br>1,000,000,000 tokens/month</td>
    </tr>"""

    table += """<tr>
        <td><a href="https://codestral.mistral.ai/" target="_blank">Mistral (Codestral)</a></td>
        <td>Currently free to use, monthly subscription based, requires phone number verification.</td>
        <td>Codestral</td>
        <td>30 requests/minute<br>2,000 requests/day</td>
    </tr>"""

    table += """<tr>
            <td><a href="https://huggingface.co/docs/api-inference/en/index" target="_blank">HuggingFace Serverless Inference</a></td>
            <td>Limited to models smaller than 10GB.<br>Some popular models are supported even if they exceed 10GB.</td>
            <td>Various open models</td>
            <td><a href="https://huggingface.co/docs/api-inference/pricing" target="_blank">Variable credits per month, currently $0.10</a></td>
        </tr>"""

    table += """<tr>
        <td rowspan="3"><a href="https://cloud.cerebras.ai/" target="_blank">Cerebras</a></td>
        <td rowspan="3">Free tier restricted to 8K context</td>
        <td>Llama 4 Scout</td>
        <td>30 requests/minute<br>60,000 tokens/minute<br>900 requests/hour<br>1,000,000 tokens/hour<br>14,400 requests/day<br>1,000,000 tokens/day</td>
    </tr>
    <tr>
        <td>Llama 3.1 8B</td>
        <td>30 requests/minute<br>60,000 tokens/minute<br>900 requests/hour<br>1,000,000 tokens/hour<br>14,400 requests/day<br>1,000,000 tokens/day</td>
    </tr>
    <tr>
        <td>Llama 3.3 70B</td>
        <td>30 requests/minute<br>60,000 tokens/minute<br>900 requests/hour<br>1,000,000 tokens/hour<br>14,400 requests/day<br>1,000,000 tokens/day</td>
    </tr>"""

    for idx, model in enumerate(groq_models):
        table += "<tr>"

        if idx == 0:
            table += f'<td rowspan="{len(groq_models)}">'
            table += '<a href="https://console.groq.com" target="_blank">Groq</a>'
            table += "</td>"
            table += f'<td rowspan="{len(groq_models)}"></td>'

        table += f"<td>{model['name']}</td>"
        table += f"<td>{get_human_limits(model)}</td>"
        table += "</tr>\n"

    for idx, model in enumerate(ovh_models):
        table += "<tr>"
        if idx == 0:
            table += '<td rowspan="' + str(len(ovh_models)) + '">'
            table += '<a href="https://endpoints.ai.cloud.ovh.net/" target="_blank">OVH AI Endpoints (Free Beta)</a>'
            table += "</td>"
            table += '<td rowspan="' + str(len(ovh_models)) + '"></td>'
        table += f"<td>{model['name']}</td>"
        table += f"<td>{get_human_limits(model)}</td>"
        table += "</tr>\n"

    together_models = [
        {
            "id": "meta-llama/Llama-Vision-Free",
            "name": "Llama 3.2 11B Vision Instruct",
            "urlId": "llama-3-2-11b-free",
        },
        {
            "id": "llmeta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "name": "Llama 3.3 70B Instruct",
            "urlId": "llama-3-3-70b-free",
        },
        {
            "id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            "name": "DeepSeek R1 Distil Llama 70B",
            "urlId": "deepseek-r1-distilled-llama-70b-free",
        },
    ]

    for idx, model in enumerate(together_models):
        table += "<tr>"
        if idx == 0:
            table += f'<td rowspan="{len(together_models)}">'
            table += '<a href="https://together.ai" target="_blank">Together</a>'
            table += "</td>"
            table += (
                f'<td rowspan="{len(together_models)}">Up to 60 requests/minute</td>'
            )
        table += f"<td><a href='https://together.ai/{model['urlId']}' target='_blank'>{model['name']}</a></td>"
        table += f"<td>{get_human_limits(model)}</td>"
        table += "</tr>\n"

    cohere_models = [
        {"id": "command-a-03-2025", "name": "Command-A"},
        {"id": "command-r7b-12-2024", "name": "Command-R7B"},
        {"id": "command-r-plus", "name": "Command-R+"},
        {"id": "command-r", "name": "Command-R"},
        {"id": "c4ai-aya-expanse-8b", "name": "Aya Expanse 8B"},
        {"id": "c4ai-aya-expanse-32b", "name": "Aya Expanse 32B"},
        {"id": "c4ai-aya-vision-8b", "name": "Aya Vision 8B"},
        {"id": "c4ai-aya-vision-32b", "name": "Aya Vision 32B"},
    ]

    for idx, model in enumerate(cohere_models):
        table += "<tr>"
        if idx == 0:
            table += f'<td rowspan="{len(cohere_models)}">'
            table += '<a href="https://cohere.com" target="_blank">Cohere</a>'
            table += "</td>"
            table += f'<td rowspan="{len(cohere_models)}"><a href="https://docs.cohere.com/docs/rate-limits">20 requests/minute<br>1,000 requests/month</a></td>'
        table += f"<td>{model['name']}</td>"
        if idx == 0:
            table += '<td rowspan="{len(cohere_models)}">Shared Limit</td>'
        table += "</tr>\n"

    for idx, model in enumerate(github_models):
        table += "<tr>"
        table += (
            f'<td rowspan="{len(github_models)}"><a href="https://github.com/marketplace/models" target="_blank">GitHub Models</a></td>'
            if idx == 0
            else ""
        )
        table += (
            f'<td rowspan="{len(github_models)}">Extremely restrictive input/output token limits.<br><a href="https://docs.github.com/en/github-models/prototyping-with-ai-models#rate-limits" target="_blank">Rate limits dependent on Copilot subscription tier (Free/Pro/Business/Enterprise)</a></td>'
            if idx == 0
            else ""
        )
        table += f"<td>{model['name']}</td>"
        table += "<td></td>"
        table += "</tr>\n"

    for idx, model in enumerate(chutes_models):
        table += "<tr>"
        if idx == 0:
            table += '<td rowspan="' + str(len(chutes_models)) + '">'
            table += '<a href="https://chutes.ai/" target="_blank">Chutes</a>'
            table += "</td>"
            table += (
                '<td rowspan="'
                + str(len(chutes_models))
                + '">Distributed, decentralized crypto-based compute. Data is sent to individual hosts.</td>'
            )
        table += f"<td>{model['name']}</td>"
        table += "<td></td>"
        table += "</tr>\n"

    for idx, model in enumerate(cloudflare_models):
        table += "<tr>"
        if idx == 0:
            table += '<td rowspan="' + str(len(cloudflare_models)) + '">'
            table += '<a href="https://developers.cloudflare.com/workers-ai" target="_blank">Cloudflare Workers AI</a>'
            table += "</td>"
            table += '<td rowspan="' + str(len(cloudflare_models)) + '">'
            table += '<a href="https://developers.cloudflare.com/workers-ai/platform/pricing/#free-allocation">10,000 neurons/day</a>'
            table += "</td>"
        table += f"<td>{model['name']}</td>"
        table += "<td></td>"
        table += "</tr>\n"

    vertex_llama_models = [
        {
            "id": "llama-4-maverick-17b-128e-instruct-maas",
            "name": "Llama 4 Maverick Instruct",
            "urlId": "llama-4-maverick-17b-128e-instruct-maas",
            "limits": {"requests/minute": 60},
        },
        {
            "id": "llama-4-scout-17b-16e-instruct-maas",
            "name": "Llama 4 Scout Instruct",
            "urlId": "llama-4-maverick-17b-128e-instruct-maas",
            "limits": {"requests/minute": 60},
        },
        {
            "id": "llama-3.3-70b-instruct-maas",
            "name": "Llama 3.3 70B Instruct",
            "urlId": "llama-3-3-70b-instruct-maas",
            "limits": {"requests/minute": 30},
        },
        {
            "id": "llama-3.2-90b-vision-instruct-maas",
            "name": "Llama 3.2 90B Vision Instruct",
            "urlId": "llama-3-2-90b-vision-instruct-maas",
            "limits": {"requests/minute": 30},
        },
        {
            "id": "llama-3.1-70b-instruct-maas",
            "name": "Llama 3.1 70B Instruct",
            "urlId": "llama-3-1-405b-instruct-maas",
            "limits": {"requests/minute": 60},
        },
        {
            "id": "llama-3.1-8b-instruct-maas",
            "name": "Llama 3.1 8B Instruct",
            "urlId": "llama-3-1-405b-instruct-maas",
            "limits": {"requests/minute": 60},
        },
    ]
    vertex_gemini_models = [
        {
            "id": "gemini-2.5-pro-exp-03-25",
            "name": "Gemini 2.5 Pro (Experimental)",
            "limits": {"requests/minute": 10},
        },
        {
            "id": "gemini-2.0-flash-exp",
            "name": "Gemini 2.0 Flash (Experimental)",
            "limits": {"requests/minute": 10},
        },
        {
            "id": "gemini-2.0-flash-thinking-exp-01-21",
            "name": "Gemini 2.0 Flash Thinking (Experimental)",
            "limits": {"requests/minute": 10},
        },
        {
            "id": "gemini-exp-1206",
            "name": "Gemini 2.0 Pro (Experimental)",
            "limits": {"requests/minute": 10},
        },
    ]

    for idx, model in enumerate(vertex_gemini_models):
        table += "<tr>"
        if idx == 0:
            table += (
                f'<td rowspan="{len(vertex_llama_models) + len(vertex_gemini_models)}">'
            )
            table += '<a href="https://console.cloud.google.com/vertex-ai/model-garden" target="_blank">Google Cloud Vertex AI</a>'
            table += "</td>"
            table += f'<td rowspan="{len(vertex_llama_models) + len(vertex_gemini_models)}">Very stringent payment verification for Google Cloud.</td>'
        table += f'<td><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental" target="_blank">{model['name']}</a></td>'
        if idx == 0:
            table += f"<td rowspan='{len(vertex_gemini_models)}'>{get_human_limits(model)}<br>Shared Quota</td>"
        table += "</tr>\n"

    for idx, model in enumerate(vertex_llama_models):
        table += "<tr>"
        table += f"<td><a href='https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/{model['urlId']}' target='_blank'>{model['name']}</a></td>"
        table += f"<td>{get_human_limits(model)}<br>Free during preview</td>"
        table += "</tr>\n"

    table += "</tbody></table>"

    trial_table = ""
    for idx, model in enumerate(hyperbolic_models):
        trial_table += "<tr>"
        if idx == 0:
            trial_table += f'<td rowspan="{len(hyperbolic_models)}">'
            trial_table += (
                '<a href="https://app.hyperbolic.xyz/" target="_blank">Hyperbolic</a>'
            )
            trial_table += "</td>"
            trial_table += f'<td rowspan="{len(hyperbolic_models)}">$1</td>'
            trial_table += f'<td rowspan="{len(hyperbolic_models)}"></td>'
        trial_table += f"<td>{model['name']}</td>"
        trial_table += "</tr>\n"

    for idx, model in enumerate(samba_models):
        trial_table += "<tr>"

        if idx == 0:
            trial_table += f'<td rowspan="{len(samba_models)}">'
            trial_table += '<a href="https://cloud.sambanova.ai/" target="_blank">SambaNova Cloud</a>'
            trial_table += "</td>"
            trial_table += f'<td rowspan="{len(samba_models)}">$5 for 3 months</td>'

        trial_table += f"<td></td>"
        trial_table += f"<td>{model['name']}</td>"
        trial_table += "</tr>\n"

    for idx, model in enumerate(scaleway_models):
        trial_table += "<tr>"
        if idx == 0:
            trial_table += '<td rowspan="' + str(len(scaleway_models)) + '">'
            trial_table += '<a href="https://console.scaleway.com/generative-api/models" target="_blank">Scaleway Generative APIs</a>'
            trial_table += "</td>"
            trial_table += (
                '<td rowspan="'
                + str(len(scaleway_models))
                + '">1,000,000 free tokens</td>'
            )
        trial_table += f"<td></td>"
        trial_table += f"<td>{model['name']}</td>"
        trial_table += "</tr>\n"

    if MISSING_MODELS:
        logger.warning("Missing models:")
        logger.warning(
            "\n" + "\n".join([f'"{model}": "{model}",' for model in MISSING_MODELS])
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "README_template.md"), "r") as f:
        readme = f.read()
    warning = """<!---
WARNING: DO NOT EDIT THIS FILE DIRECTLY. IT IS GENERATED BY src/pull_available_models.py
--->
"""
    with open(os.path.join(script_dir, "..", "README.md"), "w") as f:
        f.write(
            (warning + readme)
            .replace("{{MODEL_LIST}}", table)
            .replace("{{TRIAL_MODEL_LIST}}", trial_table)
        )
    logger.info("Wrote models to README.md")


if __name__ == "__main__":
    main()
