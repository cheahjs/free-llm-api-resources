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
import re

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


def get_human_limits(model, seperator="<br>"):
    if "limits" not in model:
        return ""
    limits = model["limits"]
    return seperator.join([f"{value:,} {key}" for key, value in limits.items()])


def generate_toc(markdown):
    toc_lines = []
    # Find all ## and ### headings, but skip the main title (# ...)
    headings = re.findall(r"^(#{2,3}) +(.+)", markdown, re.MULTILINE)
    for hashes, title in headings:
        # Remove markdown links for anchor text, keep display text
        display = re.sub(r"\[(.*?)\]\([^)]*\)", r"\1", title)
        # Build anchor (GitHub style)
        anchor = display.lower()
        anchor = re.sub(r"[^a-z0-9 \-_]", "", anchor)
        anchor = anchor.replace(" ", "-")
        anchor = anchor.replace("--", "-")
        anchor = anchor.strip("-")
        indent = "  " if len(hashes) == 3 else ""
        toc_lines.append(f"{indent}- [{display}](#{anchor})")
    return "\n".join(toc_lines)


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

    # Initialize markdown string for free providers
    model_list_markdown = ""

    # --- OpenRouter ---
    model_list_markdown += "### [OpenRouter](https://openrouter.ai)\n\n"
    if openrouter_models:
        provider_limits = get_human_limits(
            openrouter_models[0]
        ) 
        model_list_markdown += "**Limits:**\n\n"
        model_list_markdown += f"[{provider_limits}<br>1000 requests/day with $10 credit balance](https://openrouter.ai/docs/api-reference/limits)\n\n"
        model_list_markdown += "Models share a common quota.\n\n"
        for model in openrouter_models:
            model_list_markdown += (
                f"- [{model['name']}](https://openrouter.ai/{model['id']})\n"
            )
    model_list_markdown += "\n"

    # --- Google AI Studio ---
    model_list_markdown += "### [Google AI Studio](https://aistudio.google.com)\n\n"
    model_list_markdown += (
        "Data is used for training when used outside of the UK/CH/EEA/EU.\n\n"
    )
    model_list_markdown += "<table><thead><tr><th>Model Name</th><th>Model Limits</th></tr></thead><tbody>\n"

    gemini_text_models = [
        {
            "id": "gemini-2.5-pro-exp-03-25",
            "name": "Gemini 2.5 Pro (Experimental)",
            "limits": gemini_models.get("gemini-2.0-pro-exp", {}),
        },
        {
            "id": "gemini-2.5-flash-preview-04-17",
            "name": "Gemini 2.5 Flash (Preview)",
            "limits": gemini_models.get("gemini-2.5-flash", {}),
        },
        {
            "id": "gemini-2.0-flash",
            "name": "Gemini 2.0 Flash",
            "limits": gemini_models.get("gemini-2.0-flash", {}),
        },
        {
            "id": "gemini-2.0-flash-lite",
            "name": "Gemini 2.0 Flash-Lite",
            "limits": gemini_models.get("gemini-2.0-flash-lite", {}),
        },
        {
            "id": "gemini-2.0-flash-exp",
            "name": "Gemini 2.0 Flash (Experimental)",
            "limits": gemini_models.get("gemini-2.0-flash-exp", {}),
        },
        {
            "id": "gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "limits": gemini_models.get("gemini-1.5-flash", {}),
        },
        {
            "id": "gemini-1.5-flash-8b",
            "name": "Gemini 1.5 Flash-8B",
            "limits": gemini_models.get("gemini-1.5-flash-8b", {}),
        },
        {
            "id": "gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "limits": gemini_models.get("gemini-1.5-pro", {}),
        },
        {
            "id": "learnlm-1.5-pro-experimental",
            "name": "LearnLM 1.5 Pro (Experimental)",
            "limits": gemini_models.get("learnlm-1.5-pro-experimental", {}),
        },
        {
            "id": "gemma-3-27b-it",
            "name": "Gemma 3 27B Instruct",
            "limits": gemini_models.get("gemma-3-27b", {}),
        },
        {
            "id": "gemma-3-12b-it",
            "name": "Gemma 3 12B Instruct",
            "limits": gemini_models.get("gemma-3-12b", {}),
        },
        {
            "id": "gemma-3-4b-it",
            "name": "Gemma 3 4B Instruct",
            "limits": gemini_models.get("gemma-3-4b", {}),
        },
        {
            "id": "gemma-3-1b-it",
            "name": "Gemma 3 1B Instruct",
            "limits": gemini_models.get("gemma-3-1b", {}),
        },
    ]
    gemini_embedding_models = [
        {
            "id": "text-embedding-004",
            "name": "text-embedding-004",
            "limits": gemini_models.get("project-embedding", {}),
        },
        {
            "id": "embedding-001",
            "name": "embedding-001",
            "limits": gemini_models.get("project-embedding", {}),
        },
    ]

    # Write text models to table
    for model in gemini_text_models:
        limits_str = get_human_limits(model)
        model_list_markdown += (
            f"<tr><td>{model['name']}</td><td>{limits_str}</td></tr>\n"
        )

    # Write embedding models to table
    first_embedding = True
    for model in gemini_embedding_models:
        limits_str = get_human_limits(model)
        model_list_markdown += f"<tr><td>{model['name']}</td>"
        if first_embedding:
            model_list_markdown += f'<td rowspan="{len(gemini_embedding_models)}">{limits_str}<br>100 content/batch<br>Shared Quota</td>'
            first_embedding = False
        model_list_markdown += "</tr>\n"

    model_list_markdown += "</tbody></table>\n\n"

    # --- NVIDIA NIM ---
    model_list_markdown += (
        "### [NVIDIA NIM](https://build.nvidia.com/explore/discover)\n\n"
    )
    model_list_markdown += "Phone number verification required.\n"
    model_list_markdown += "Models tend to be context window limited.\n\n"
    model_list_markdown += "**Limits:** 40 requests/minute\n\n"
    model_list_markdown += "- [Various open models](https://build.nvidia.com/models)\n"
    model_list_markdown += "\n"

    # --- Mistral (La Plateforme) ---
    model_list_markdown += (
        "### [Mistral (La Plateforme)](https://console.mistral.ai/)\n\n"
    )
    model_list_markdown += "* Free tier (Experiment plan) requires opting into data training\n"
    model_list_markdown += "* Requires phone number verification.\n\n"
    model_list_markdown += "**Limits (per-model):** 1 request/second, 500,000 tokens/minute, 1,000,000,000 tokens/month\n\n"
    model_list_markdown += "- [Open and Proprietary Mistral models](https://docs.mistral.ai/getting-started/models/models_overview/)\n"
    model_list_markdown += "\n"

    # --- Mistral (Codestral) ---
    model_list_markdown += (
        "### [Mistral (Codestral)](https://codestral.mistral.ai/)\n\n"
    )
    model_list_markdown += "* Currently free to use\n"
    model_list_markdown += "* Monthly subscription based\n"
    model_list_markdown += "* Requires phone number verification\n\n"
    model_list_markdown += "**Limits:** 30 requests/minute, 2,000 requests/day\n\n"
    model_list_markdown += "- Codestral\n"
    model_list_markdown += "\n"

    # --- HuggingFace Serverless Inference ---
    model_list_markdown += "### [HuggingFace Inference Providers](https://huggingface.co/docs/inference-providers/en/index)\n\n"
    model_list_markdown += "HuggingFace Serverless Inference limited to models smaller than 10GB. Some popular models are supported even if they exceed 10GB.\n\n"
    model_list_markdown += "**Limits:** [$0.10/month in credits](https://huggingface.co/docs/inference-providers/en/pricing)\n\n"
    model_list_markdown += "- Various open models across supported providers\n"
    model_list_markdown += "\n"

    # --- Cerebras ---
    model_list_markdown += "### [Cerebras](https://cloud.cerebras.ai/)\n\n"
    model_list_markdown += "Free tier restricted to 8K context.\n\n"
    model_list_markdown += "<table><thead><tr><th>Model Name</th><th>Model Limits</th></tr></thead><tbody>\n"
    cerebras_limit_text = "30 requests/minute<br>60,000 tokens/minute<br>900 requests/hour<br>1,000,000 tokens/hour<br>14,400 requests/day<br>1,000,000 tokens/day"
    cerebras_models = [
        {"name": "Llama 4 Scout", "limits_text": cerebras_limit_text},
        {"name": "Llama 3.1 8B", "limits_text": cerebras_limit_text},
        {"name": "Llama 3.3 70B", "limits_text": cerebras_limit_text},
    ]
    for model in cerebras_models:
        model_list_markdown += (
            f"<tr><td>{model['name']}</td><td>{model['limits_text']}</td></tr>\n"
        )
    model_list_markdown += "</tbody></table>\n\n"

    # --- Groq ---
    model_list_markdown += "### [Groq](https://console.groq.com)\n\n"
    if groq_models:
        model_list_markdown += "<table><thead><tr><th>Model Name</th><th>Model Limits</th></tr></thead><tbody>\n"
        for model in groq_models:
            limits_str = get_human_limits(model)
            model_list_markdown += (
                f"<tr><td>{model['name']}</td><td>{limits_str}</td></tr>\n"
            )
        model_list_markdown += "</tbody></table>\n"
    model_list_markdown += "\n"

    # --- OVH AI Endpoints ---
    model_list_markdown += (
        "### [OVH AI Endpoints (Free Beta)](https://endpoints.ai.cloud.ovh.net/)\n\n"
    )
    if ovh_models:
        model_list_markdown += "<table><thead><tr><th>Model Name</th><th>Model Limits</th></tr></thead><tbody>\n"
        for model in ovh_models:
            limits_str = get_human_limits(model)
            model_list_markdown += (
                f"<tr><td>{model['name']}</td><td>{limits_str}</td></tr>\n"
            )
        model_list_markdown += "</tbody></table>\n"
    model_list_markdown += "\n"

    # --- Together ---
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
    model_list_markdown += "### [Together (Free)](https://together.ai)\n\n"
    model_list_markdown += "**Limits:** Up to 60 requests/minute\n\n"
    if together_models:
        for model in together_models:
            model_list_markdown += (
                f"- [{model['name']}](https://together.ai/{model['urlId']})\n"
            )
    model_list_markdown += "\n"

    # --- Cohere ---
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
    model_list_markdown += "### [Cohere](https://cohere.com)\n\n"
    model_list_markdown += "**Limits:**\n\n"
    model_list_markdown += "[20 requests/minute<br>1,000 requests/month](https://docs.cohere.com/docs/rate-limits)\n\n"
    model_list_markdown += "Models share a common quota.\n\n"
    if cohere_models:
        for model in cohere_models:
            model_list_markdown += f"- {model['name']}\n"
    model_list_markdown += "\n"

    # --- GitHub Models ---
    model_list_markdown += (
        "### [GitHub Models](https://github.com/marketplace/models)\n\n"
    )
    model_list_markdown += "Extremely restrictive input/output token limits.\n\n"
    model_list_markdown += "**Limits:** [Dependent on Copilot subscription tier (Free/Pro/Pro+/Business/Enterprise)](https://docs.github.com/en/github-models/prototyping-with-ai-models#rate-limits)\n\n"
    if github_models:
        for model in github_models:
            model_list_markdown += f"- {model['name']}\n"
    model_list_markdown += "\n"

    # --- Chutes ---
    model_list_markdown += "### [Chutes](https://chutes.ai/)\n\n"
    model_list_markdown += "Distributed, decentralized crypto-based compute.\n"
    model_list_markdown += "Data is sent to individual hosts.\n\n"
    if chutes_models:
        for model in chutes_models:
            model_list_markdown += f"- {model['name']}\n"
    model_list_markdown += "\n"

    # --- Cloudflare Workers AI ---
    model_list_markdown += (
        "### [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai)\n\n"
    )
    model_list_markdown += "**Limits:** [10,000 neurons/day](https://developers.cloudflare.com/workers-ai/platform/pricing/#free-allocation)\n\n"
    if cloudflare_models:
        for model in cloudflare_models:
            model_list_markdown += f"- {model['name']}\n"
    model_list_markdown += "\n"

    # --- Google Cloud Vertex AI ---
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
    model_list_markdown += "### [Google Cloud Vertex AI](https://console.cloud.google.com/vertex-ai/model-garden)\n\n"
    model_list_markdown += "Very stringent payment verification for Google Cloud.\n\n"
    model_list_markdown += "<table><thead><tr><th>Model Name</th><th>Model Limits</th></tr></thead><tbody>\n"

    # Write Gemini models to table
    first_gemini = True
    if vertex_gemini_models:
        for model in vertex_gemini_models:
            limits_str = get_human_limits(model)
            model_list_markdown += f'<tr><td><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental" target="_blank">{model['name']}</a></td>'
            if first_gemini:
                model_list_markdown += f'<td rowspan="{len(vertex_gemini_models)}">{limits_str}<br>Shared Quota</td>'
                first_gemini = False
            model_list_markdown += "</tr>\n"

    # Write Llama models to table
    if vertex_llama_models:
        for model in vertex_llama_models:
            limits_str = get_human_limits(model)
            model_list_markdown += f'<tr><td><a href="https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/{model['urlId']}" target="_blank">{model['name']}</a></td><td>{limits_str}<br>Free during preview</td></tr>\n'

    model_list_markdown += "</tbody></table>\n\n"

    # --- Trial Providers Section Generation ---
    trial_list_markdown = ""

    # --- Static Trial Providers (Markdown List/Simple Entry) ---
    trial_providers_static = [
        {
            "name": "Together",
            "url": "https://together.ai",
            "credits": "$1 when you add a payment method",
            "requirements": "",
            "models_desc": "[Various open models](https://together.ai/models)",
        },
        {
            "name": "Fireworks",
            "url": "https://fireworks.ai/",
            "credits": "$1",
            "requirements": "",
            "models_desc": "[Various open models](https://fireworks.ai/models)",
        },
        {
            "name": "Unify",
            "url": "https://unify.ai/",
            "credits": "$5 when you add a payment method",
            "requirements": "",
            "models_desc": "Routes to other providers, various open models and proprietary models (OpenAI, Gemini, Anthropic, Mistral, Perplexity, etc)",
        },
        {
            "name": "Baseten",
            "url": "https://app.baseten.co/",
            "credits": "$30",
            "requirements": "",
            "models_desc": "[Any supported model - pay by compute time](https://www.baseten.co/library/)",
        },
        {
            "name": "Nebius",
            "url": "https://studio.nebius.com/",
            "credits": "$1",
            "requirements": "",
            "models_desc": "[Various open models](https://studio.nebius.ai/models)",
        },
        {
            "name": "Novita",
            "url": "https://novita.ai/referral?invited_code=E5R0CA&ref=ytblmjc&utm_source=affiliate",
            "credits": "$0.5 for 1 year, $20 for 3 months for DeepSeek models with [referral code](https://novita.ai/referral?invited_code=E5R0CA&ref=ytblmjc&utm_source=affiliate) + GitHub account connection",
            "requirements": "",
            "models_desc": "[Various open models](https://novita.ai/models)",
        },
        {
            "name": "AI21",
            "url": "https://studio.ai21.com/",
            "credits": "$10 for 3 months",
            "requirements": "",
            "models_desc": "Jamba family of models",
        },
        {
            "name": "Upstage",
            "url": "https://console.upstage.ai/",
            "credits": "$10 for 3 months",
            "requirements": "",
            "models_desc": "Solar Pro/Mini",
        },
        {
            "name": "NLP Cloud",
            "url": "https://nlpcloud.com/home",
            "credits": "$15",
            "requirements": "Phone number verification",
            "models_desc": "Various open models",
        },
        {
            "name": "Alibaba Cloud (International) Model Studio",
            "url": "https://bailian.console.alibabacloud.com/",
            "credits": "Token/time-limited trials on a per-model basis",
            "requirements": "",
            "models_desc": "[Various open and proprietary Qwen models](https://www.alibabacloud.com/en/product/modelstudio)",
        },
        {
            "name": "Modal",
            "url": "https://modal.com",
            "credits": "$30/month",
            "requirements": "",
            "models_desc": "Any supported model - pay by compute time",
        },
    ]

    for provider in trial_providers_static:
        trial_list_markdown += f"### [{provider['name']}]({provider['url']})\n\n"
        trial_list_markdown += f"**Credits:** {provider['credits']}\n\n"
        if provider["requirements"]:
            trial_list_markdown += f"**Requirements:** {provider['requirements']}\n\n"
        trial_list_markdown += f"**Models:** {provider['models_desc']}\n\n"

    # --- Hyperbolic (Trial - Table) ---
    if hyperbolic_models:
        trial_list_markdown += "### [Hyperbolic](https://app.hyperbolic.xyz/)\n\n"
        trial_list_markdown += "**Credits:** $1\n\n"
        trial_list_markdown += "**Models:**\n"
        for model in hyperbolic_models:
            trial_list_markdown += f"- {model['name']}\n"
        trial_list_markdown += "\n"

    # --- SambaNova Cloud (Trial - Table) ---
    if samba_models:
        trial_list_markdown += "### [SambaNova Cloud](https://cloud.sambanova.ai/)\n\n"
        trial_list_markdown += "**Credits:** $5 for 3 months\n\n"
        trial_list_markdown += "**Models:**\n"
        for model in samba_models:
            trial_list_markdown += f"- {model['name']}\n"
        trial_list_markdown += "\n"

    # --- Scaleway Generative APIs (Trial - Table) ---
    if scaleway_models:
        trial_list_markdown += "### [Scaleway Generative APIs](https://console.scaleway.com/generative-api/models)\n\n"
        trial_list_markdown += "**Credits:** 1,000,000 free tokens\n\n"
        trial_list_markdown += "**Models:**\n"
        for model in scaleway_models:
            trial_list_markdown += f"- {model['name']}\n"
        trial_list_markdown += "\n"

    if MISSING_MODELS:
        logger.warning("Missing models:")
        logger.warning(
            "\n" + "\n".join([f'"{model}": "{model}",' for model in MISSING_MODELS])
        )

    with open(os.path.join(script_dir, "README_template.md"), "r") as f:
        readme = f.read()
    warning = """<!---
WARNING: DO NOT EDIT THIS FILE DIRECTLY. IT IS GENERATED BY src/pull_available_models.py
--->
"""
    initial_templated = (
        (warning + readme)
        .replace("{{MODEL_LIST}}", model_list_markdown)
        .replace("{{TRIAL_LIST_MARKDOWN}}", trial_list_markdown)
    )
    toc_markdown = generate_toc(initial_templated)
    with open(os.path.join(script_dir, "..", "README.md"), "w") as f:
        f.write(initial_templated.replace("{{TOC}}", toc_markdown))
    logger.info("Wrote models to README.md")


if __name__ == "__main__":
    main()
