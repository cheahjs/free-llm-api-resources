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


load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Global clients
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
last_mistral_request_time = 0

MODEL_TO_NAME_MAPPING = {
    "@cf/deepseek-ai/deepseek-math-7b-instruct": "Deepseek Math 7B Instruct",
    "@cf/defog/sqlcoder-7b-2": "SQLCoder 7B 2",
    "@cf/fblgit/una-cybertron-7b-v2-bf16": "Una Cybertron 7B v2 (BF16)",
    "@cf/google/gemma-2b-it-lora": "Gemma 2B Instruct (LoRA)",
    "@cf/google/gemma-7b-it-lora": "Gemma 7B Instruct (LoRA)",
    "@cf/meta-llama/llama-2-7b-chat-hf-lora": "Llama 2 7B Chat (LoRA)",
    "@cf/meta/llama-2-7b-chat-fp16": "Llama 2 7B Chat (FP16)",
    "@cf/meta/llama-2-7b-chat-int8": "Llama 2 7B Chat (INT8)",
    "@cf/meta/llama-3-8b-instruct-awq": "Llama 3 8B Instruct (AWQ)",
    "@cf/meta/llama-3-8b-instruct": "Llama 3 8B Instruct",
    "@cf/meta/llama-3.1-8b-instruct-awq": "Llama 3.1 8B Instruct (AWQ)",
    "@cf/meta/llama-3.1-8b-instruct-fp8": "Llama 3.1 8B Instruct (FP8)",
    "@cf/meta/llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
    "@cf/microsoft/phi-2": "Phi-2",
    "@cf/mistral/mistral-7b-instruct-v0.1-vllm": "Mistral 7B Instruct v0.1",
    "@cf/mistral/mistral-7b-instruct-v0.1": "Mistral 7B Instruct v0.1",
    "@cf/mistral/mistral-7b-instruct-v0.2-lora": "Mistral 7B Instruct v0.2 (LoRA)",
    "@cf/openchat/openchat-3.5-0106": "OpenChat 3.5 0106",
    "@cf/qwen/qwen1.5-0.5b-chat": "Qwen 1.5 0.5B Chat",
    "@cf/qwen/qwen1.5-1.8b-chat": "Qwen 1.5 1.8B Chat",
    "@cf/qwen/qwen1.5-14b-chat-awq": "Qwen 1.5 14B Chat (AWQ)",
    "@cf/qwen/qwen1.5-7b-chat-awq": "Qwen 1.5 7B Chat (AWQ)",
    "@cf/thebloke/discolm-german-7b-v1-awq": "Discolm German 7B v1 (AWQ)",
    "@cf/tiiuae/falcon-7b-instruct": "Falcom 7B Instruct",
    "@cf/tinyllama/tinyllama-1.1b-chat-v1.0": "TinyLlama 1.1B Chat v1.0",
    "@hf/google/gemma-7b-it": "Gemma 7B Instruct",
    "@hf/meta-llama/meta-llama-3-8b-instruct": "Llama 3 8B Instruct",
    "@hf/mistral/mistral-7b-instruct-v0.2": "Mistral 7B Instruct v0.2",
    "@hf/nexusflow/starling-lm-7b-beta": "Starling LM 7B Beta",
    "@hf/nousresearch/hermes-2-pro-mistral-7b": "Hermes 2 Pro Mistral 7B",
    "@hf/thebloke/deepseek-coder-6.7b-base-awq": "Deepseek Coder 6.7B Base (AWQ)",
    "@hf/thebloke/deepseek-coder-6.7b-instruct-awq": "Deepseek Coder 6.7B Instruct (AWQ)",
    "@hf/thebloke/llama-2-13b-chat-awq": "Llama 2 13B Chat (AWQ)",
    "@hf/thebloke/llamaguard-7b-awq": "LlamaGuard 7B (AWQ)",
    "@hf/thebloke/mistral-7b-instruct-v0.1-awq": "Mistral 7B Instruct v0.1 (AWQ)",
    "@hf/thebloke/neural-chat-7b-v3-1-awq": "Neural Chat 7B v3.1 (AWQ)",
    "@hf/thebloke/openhermes-2.5-mistral-7b-awq": "OpenHermes 2.5 Mistral 7B (AWQ)",
    "@hf/thebloke/zephyr-7b-beta-awq": "Zephyr 7B Beta (AWQ)",
    "codellama-13b-instruct-hf": "CodeLlama 13B Instruct",
    "distil-whisper-large-v3-en": "Distil Whisper Large v3",
    "gemma-7b-it": "Gemma 7B Instruct (Deprecated)",
    "gemma2-9b-it": "Gemma 2 9B Instruct",
    "google/gemma-2-9b-it:free": "Gemma 2 9B Instruct",
    "google/gemma-7b-it:free": "Gemma 7B Instruct",
    "gryphe/mythomist-7b:free": "Mythomist 7B",
    "huggingfaceh4/zephyr-7b-beta:free": "Zephyr 7B Beta",
    "llama-2-13b-chat-hf": "Llama 2 13B Chat",
    "llama-3-70b-instruct": "Llama 3 70B Instruct",
    "llama-3-8b-instruct": "Llama 3 8B Instruct",
    "llama-3.1-405b-reasoning": "Llama 3.1 405B",
    "llama-3.1-70b-versatile": "Llama 3.1 70B",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "llama-guard-3-8b": "Llama Guard 3 8B",
    "llama3-70b-8192": "Llama 3 70B",
    "llama3-8b-8192": "Llama 3 8B",
    "llama3-groq-70b-8192-tool-use-preview": "Llama 3 70B - Groq Tool Use Preview",
    "llama3-groq-8b-8192-tool-use-preview": "Llama 3 8B - Groq Tool Use Preview",
    "meta-llama/llama-3-8b-instruct:free": "Llama 3 8B Instruct",
    "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B Instruct",
    "meta-llama/meta-llama-3-70b-instruct": "Llama 3 70B Instruct",
    "meta-llama/meta-llama-3.1-405b": "Llama 3.1 405B Base",
    "meta-llama/meta-llama-3.1-405b-fp8": "Llama 3.1 405B Base (FP8)",
    "meta-llama/meta-llama-3.1-405b-instruct": "Llama 3.1 405B Instruct",
    "meta-llama/meta-llama-3.1-70b-instruct": "Llama 3.1 70B Instruct",
    "meta-llama/meta-llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
    "microsoft/phi-3-medium-128k-instruct:free": "Phi-3 Medium 128k Instruct",
    "microsoft/phi-3-mini-128k-instruct:free": "Phi-3 Mini 128k Instruct",
    "mistral-7b-instruct": "Mistral 7B Instruct",
    "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct",
    "mixtral-8x22b-instruct": "Mixtral 8x22B Instruct",
    "mixtral-8x7b-32768": "Mixtral 8x7B",
    "mixtral-8x7b-instruct": "Mixtral 8x7B Instruct",
    "nousresearch/hermes-3-llama-3.1-70b": "Hermes 3 Llama 3.1 70B",
    "nousresearch/nous-capybara-7b:free": "Nous Capybara 7B",
    "openchat/openchat-7b:free": "OpenChat 7B",
    "qwen/qwen-2-7b-instruct:free": "Qwen 2 7B Instruct",
    "qwen/qwen2-72b-instruct": "Qwen 2 72B Instruct",
    "undi95/toppy-m-7b:free": "Toppy M 7B",
    "whisper-large-v3": "Whisper Large v3",
    "whisper-large-v3-turbo": "Whisper Large v3 Turbo",
    "01-ai/yi-34b-chat": "Yi 34B Chat",
    "01-ai/yi-1.5-34b-chat": "Yi 1.5 34B Chat",
    "nousresearch/hermes-3-llama-3.1-70b-fp8": "Hermes 3 Llama 3.1 70B (FP8)",
    "nousresearch/hermes-3-llama-3.1-405b:free": "Hermes 3 Llama 3.1 405B",
    "llava-v1.5-7b-4096-preview": "LLaVA 1.5 7B",
    "mattshumer/reflection-llama-3.1-70b": "Reflection Llama 3.1 70B",
    "mattshumer/reflection-70b:free": "Reflection Llama 3.1 70B",
    "mattshumer/reflection-llama-3.1-70b-completions": "Reflection Llama 3.1 70B Completions",
    "deepseek-ai/deepseek-v2.5": "DeepSeek V2.5",
    "mistralai/pixtral-12b-2409": "Pixtral 12B (2409)",
    "qwen/qwen2-vl-7b-instruct": "Qwen2-VL 7B Instruct",
    "mistralai/pixtral-12b:free": "Pixtral 12B",
    "qwen/qwen-2-vl-7b-instruct:free": "Qwen2-VL 7B Instruct",
    "qwen/qwen2-vl-72b-instruct": "Qwen2-VL 72B Instruct",
    "qwen/qwen2.5-72b-instruct": "Qwen2.5 72B Instruct",
    "llama-3.2-90b-text-preview": "Llama 3.2 90B (Text Only)",
    "llama-3.2-3b-preview": "Llama 3.2 3B",
    "llama-3.2-11b-text-preview": "Llama 3.2 11B (Text Only)",
    "llama-3.2-1b-preview": "Llama 3.2 1B",
    "@cf/meta/llama-3.2-1b-instruct": "Llama 3.2 1B Instruct",
    "meta-llama/llama-3.2-11b-vision-instruct:free": "Llama 3.2 11B Vision Instruct",
    "@cf/meta/llama-3.2-11b-vision-instruct": "Llama 3.2 11B Vision Instruct",
    "@cf/meta/llama-3.2-3b-instruct": "Llama 3.2 3B Instruct",
    "meta-llama/llama-3.2-90b-vision-instruct": "Llama 3.2 90B Vision Instruct",
    "meta-llama/llama-3.2-3b-instruct": "Llama 3.2 3B Instruct",
    "llama-3.2-11b-vision-preview": "Llama 3.2 11B Vision",
    "llama-3.2-90b-vision-preview": "Llama 3.2 90B Vision",
    "meta-llama/llama-3.2-90b-vision": "Llama 3.2 90B Vision",
    "meta-llama/llama-3.1-70b-instruct:free": "Llama 3.1 70B Instruct",
    "meta-llama/llama-3.2-1b-instruct:free": "Llama 3.2 1B Instruct",
    "liquid/lfm-40b:free": "Liquid LFM 40B",
    "meta-llama/llama-3.2-3b-instruct:free": "Llama 3.2 3B Instruct",
    "meta-llama/llama-3.1-405b-instruct:free": "Llama 3.1 405B Instruct",
    "mathstral-7b-v0.1": "Mathstral 7B v0.1",
    "llama-3.1-70b-instruct": "Llama 3.1 70B Instruct",
    "gryphe/mythomax-l2-13b:free": "Mythomax L2 13B",
    "meta-llama/llama-3.2-90b-vision-instruct:free": "Llama 3.2 90B Vision Instruct",
    "mamba-codestral-7b-v0-1": "Codestral Mamba 7B v0.1",
    "hermes3-70b": "Hermes 3 70B",
    "llama3.1-nemotron-70b-instruct": "Llama 3.1 Nemotron 70B Instruct",
    "llama3.2-3b-instruct": "Llama 3.2 3B Instruct",
    "llama3.1-8b-instruct": "Llama 3.1 8B Instruct",
    "llama3.1-70b-instruct-fp8": "Llama 3.1 70B Instruct (FP8)",
    "llama3.1-405b-instruct-fp8": "Llama 3.1 405B Instruct (FP8)",
    "hermes3-405b": "Hermes 3 405B",
    "deepseek-coder-v2-lite-instruct": "DeepSeek Coder v2 Lite Instruct",
    "hermes3-8b": "Hermes 3 8B",
    "dracarys2-72b-instruct": "Dracarys 2 72B Instruct",
    "lfm-40b": "Liquid LFM 40B",
    "qwen/qwen2.5-coder-32b-instruct": "Qwen2.5 Coder 32B Instruct",
    "thedrummer/unslopnemo-12b:free": "UnslopNemo 12B",
    "mistral-nemo-instruct-2407": "Mistral Nemo 2407",
    "google/gemini-exp-1121:free": "Gemini Experimental 1121",
    "meta-llama/llama-3.1-70b-instruct-fp8": "Llama 3.1 70B Instruct (FP8)",
    "google/learnlm-1.5-pro-experimental:free": "LearnLM 1.5 Pro Experimental",
    "google/gemini-exp-1114:free": "Gemini Experimental 1114",
    "qwen25-coder-32b-instruct": "Qwen2.5 Coder 32B Instruct",
    "qwen/qwq-32b-preview": "Qwen QwQ 32B Preview",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B Instruct",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "google/gemini-exp-1206:free": "Gemini Experimental 1206",
    "llama3.1-nemotron-70b-instruct-fp8": "Llama 3.1 Nemotron 70B Instruct (FP8)",
    "llama-3.3-70b-specdec": "Llama 3.3 70B (Speculative Decoding)",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": "Llama 3.3 70B Instruct (FP8)",
    "google/gemini-2.0-flash-exp:free": "Gemini 2.0 Flash Experimental",
    "qwen2.5-coder-32b-instruct": "Qwen2.5 Coder 32B Instruct",
    "bge-multilingual-gemma2": "BGE-Multilingual-Gemma2",
    "pixtral-12b-2409": "Pixtral 12B (2409)",
    "google/gemini-2.0-flash-thinking-exp:free": "Gemini 2.0 Flash Thinking Experimental",
    "sentence-t5-xxl": "sentence-t5-xxl",
    "meta-llama/meta-llama-3.1-405b-instruct-virtuals": "Llama 3.1 405B Instruct Virtuals",
    "llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
    "deepseek-ai/deepseek-v3": "DeepSeek V3",
    "llava-next-mistral-7b": "Llava Next Mistral 7B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B Instruct",
    "google/gemini-2.0-flash-thinking-exp-1219:free": "Gemini 2.0 Flash Thinking Experimental 1219",
    "sophosympatheia/rogue-rose-103b-v0.2:free": "Rogue Rose 103B v0.2",
    "deepseek-ai/deepseek-r1": "DeepSeek R1",
    "deepseek-ai/deepseek-r1-zero": "DeepSeek R1-Zero",
    "deepseek/deepseek-r1:free": "DeepSeek R1",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
    "deepseek-ai/janus-pro-7b": "DeepSeek Janus Pro 7B",
    "deepseek-r1-distill-llama-8b": "DeepSeek R1 Distill Llama 8B",
    "nvidia/llama-3.1-nemotron-70b-instruct:free": "Llama 3.1 Nemotron 70B Instruct",
    "deepseek/deepseek-r1-distill-llama-70b:free": "DeepSeek R1 Distill Llama 70B",
    "qwen/qwen2.5-vl-72b-instruct:free": "Qwen2.5 VL 72B Instruct",
    "google/gemini-2.0-flash-lite-preview-02-05:free": "Gemini 2.0 Flash Lite Preview 02-05",
    "qwen/qwen-vl-plus:free": "Qwen VL Plus",
    "google/gemini-2.0-pro-exp-02-05:free": "Gemini 2.0 Pro Experimental 02-05",
    "deepseek-r1": "DeepSeek R1",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B Instruct",
    "deepseek/deepseek-chat:free": "DeepSeek V3",
    "deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
    "mistralai/mistral-nemo:free": "Mistral Nemo",
    "allam-2-7b": "Allam 2 7B",
    "mistralai/mistral-small-24b-instruct-2501:free": "Mistral Small 24B Instruct 2501",
    "qwen-2.5-32b": "Qwen 2.5 32B",
}


def create_logger(provider_name):
    logger = logging.getLogger(provider_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"{provider_name}: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


MISSING_MODELS = set()

HYPERBOLIC_IGNORED_MODELS = {
    "Wifhat",
    "FLUX.1-dev",
    "StableDiffusion",
    "Monad",
    "TTS",
    "deepseek-ai/Janus-Pro-7B",
    "test",
    # Ignore DeepSeek R1 and R1-Zero because they are not available in the free tier.
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Zero",
}

LAMBDA_IGNORED_MODELS = {"lfm-40b-vllm", "hermes3-405b-fp8-128k"}

OPENROUTER_IGNORED_MODELS = {
    "google/gemini-exp-1121:free",
    "google/learnlm-1.5-pro-experimental:free",
    "google/gemini-exp-1114:free",
    "google/gemini-exp-1206:free",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-2.0-flash-thinking-exp:free",
    "google/gemini-2.0-flash-thinking-exp-1219:free",
    "google/gemini-flash-1.5-exp:free",
    "google/gemini-2.0-pro-exp-02-05:free",
}  # Ignore gemini experimental free models because rate limits mean they are unusable.


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
            future = executor.submit(get_groq_limits_for_model, model["id"], script_dir, logger)
            futures.append((model, future))
        
        for model, future in futures:
            limits = future.result()
            ret_models.append({
                "id": model["id"],
                "name": get_model_name(model["id"]),
                "limits": limits,
            })
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
                    "requests/day": 200,
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
    models = list(
        filter(lambda x: x["available"] and "LLM" in x["category"], r.json())
    )
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
    logger.info("Fetching Hyperbolic models...")
    firestore_models = fetch_hyperbolic_models_firestore(logger)
    api_models = fetch_hyperbolic_models_api(logger)
    for model in api_models:
        if model["id"] not in [m["id"] for m in firestore_models]:
            logger.debug(f"Adding model {model['id']} from API")
            firestore_models.append(model)
    for model in firestore_models:
        if model["id"] not in [m["id"] for m in api_models]:
            logger.warning(f"Model {model['id']} from Firestore not in API")
    return sorted(firestore_models, key=lambda x: x["name"])


def fetch_hyperbolic_models_firestore(logger):
    logger.info("Fetching Hyperbolic models from Firestore...")

    r = requests.post(
        "https://firestore.googleapis.com/v1/projects/ai-dashboard-cfd6a/databases/(default)/documents:runQuery",
        headers={
            "accept": "*/*",
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
            "content-type": "text/plain",
            "dnt": "1",
            "google-cloud-resource-prefix": "projects/ai-dashboard-cfd6a/databases/(default)",
            "origin": "https://app.hyperbolic.xyz",
            "priority": "u=1, i",
            "referer": "https://app.hyperbolic.xyz/",
            "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "x-goog-api-client": "gl-js/ fire/10.10.0_lite",
            "x-goog-request-params": "project_id=ai-dashboard-cfd6a",
        },
        json={
            "structuredQuery": {
                "from": [{"collectionId": "models"}],
                "where": {
                    "compositeFilter": {
                        "op": "AND",
                        "filters": [
                            {
                                "fieldFilter": {
                                    "field": {"fieldPath": "type"},
                                    "op": "IN",
                                    "value": {
                                        "arrayValue": {
                                            "values": [
                                                {"stringValue": "llm"},
                                                # {"stringValue": "vlm"},
                                            ]
                                        }
                                    },
                                },
                            },
                            # {
                            #     "fieldFilter": {
                            #         "field": {"fieldPath": "hidden"},
                            #         "op": "NOT_EQUAL",
                            #         "value": {"booleanValue": True},
                            #     },
                            # },
                        ],
                    },
                },
            }
        },
    )
    r.raise_for_status()
    models = r.json()
    logger.info(f"Fetched {len(models)} models from Hyperbolic's Firestore")
    ret_models = []
    for model in models:
        model_data = model["document"]["fields"]
        if "hidden" in model_data and model_data["hidden"]["booleanValue"]:
            continue
        ret_models.append(
            {
                "id": model_data["model"]["stringValue"],
                "name": get_model_name(model_data["model"]["stringValue"]),
                "limits": {
                    "requests/minute": 60,
                },
            }
        )
    logger.debug(json.dumps(ret_models, indent=4))
    return ret_models


def fetch_hyperbolic_models_api(logger):
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
    return ret_models


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
                models["project-embedding"][
                    f"requests/{quota.refresh_interval}"
                ] = dimension.details.value
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
            ]
            gemini_models, openrouter_models, hyperbolic_models, ovh_models, \
            cloudflare_models, github_models, samba_models, scaleway_models = \
            [f.result() for f in futures]
            
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
            table += (
                f'<td rowspan="{len(openrouter_models)}">{get_human_limits(model)}</td>'
            )

        table += f"<td>{model['name']}</td>"
        table += "<td></td>"
        table += "</tr>\n"

    table += f"""<tr>
            <td rowspan="11"><a href="https://aistudio.google.com" target="_blank">Google AI Studio</a></td>
            <td rowspan="11">Data is used for training (when used outside of the UK/CH/EEA/EU).</td>
            <td>Gemini 2.0 Flash</td>
            <td>{get_human_limits({"limits": gemini_models["gemini-2.0-flash"]})}</td>
        </tr>
        <tr>
            <td>Gemini 2.0 Flash-Lite</td>
            <td>{get_human_limits({"limits": gemini_models["gemini-2.0-flash-lite"]})}</td>
        </tr>
        <tr>
            <td>Gemini 2.0 Flash (Experimental)</td>
            <td>{get_human_limits({"limits": gemini_models["gemini-2.0-flash-exp"]})}</td>
        </tr>
        <tr>
            <td>Gemini 2.0 Pro (Experimental)</td>
            <td>{get_human_limits({"limits": gemini_models["gemini-2.0-pro-exp"]})}</td>
        </tr>
        <tr>
            <td>Gemini 1.5 Flash</td>
            <td>{get_human_limits({"limits": gemini_models["gemini-1.5-flash"]})}</td>
        </tr>
        <tr>
            <td>Gemini 1.5 Flash-8B</td>
            <td>{get_human_limits({"limits": gemini_models["gemini-1.5-flash-8b"]})}</td>
        </tr>
        <tr>
            <td>Gemini 1.5 Pro</td>
            <td>{get_human_limits({"limits": gemini_models["gemini-1.5-pro"]})}</td>
        </tr>
        <tr>
            <td>LearnLM 1.5 Pro (Experimental)</td>
            <td>{get_human_limits({"limits": gemini_models["learnlm-1.5-pro-experimental"]})}</td>
        </tr>
        <tr>
            <td>Gemini 1.0 Pro</td>
            <td>{get_human_limits({"limits": gemini_models["gemini-1.0-pro"]})}</td>
        </tr>
        <tr>
            <td>text-embedding-004</td>
            <td rowspan="2">{get_human_limits({"limits": gemini_models["project-embedding"]})}<br>100 content/batch</td>
        </tr>
        <tr>
            <td>embedding-001</td>
        </tr>"""

    table += """<tr>
        <td><a href="https://console.mistral.ai/" target="_blank">Mistral (La Plateforme)</a></td>
        <td>Free tier (Experiment plan) requires opting into data training, requires phone number verification.</td>
        <td>Open and Proprietary Mistral models</td>
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
        <td rowspan="2"><a href="https://cloud.cerebras.ai/" target="_blank">Cerebras</a></td>
        <td rowspan="2">Free tier restricted to 8K context</td>
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

    for idx, model in enumerate(scaleway_models):
        table += "<tr>"
        if idx == 0:
            table += '<td rowspan="' + str(len(scaleway_models)) + '">'
            table += '<a href="https://console.scaleway.com/generative-api/models" target="_blank">Scaleway Generative APIs (Free Beta until 12 March 2025)</a>'
            table += "</td>"
            table += '<td rowspan="' + str(len(scaleway_models)) + '"></td>'
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

    table += """<tr>
        <td rowspan="3"><a href="https://together.ai">Together</a></td>
        <td rowspan="3"></td>
        <td>Llama 3.2 11B Vision Instruct</td>
        <td></td>
    </tr>
    <tr>
        <td>Llama 3.3 70B Instruct</td>
        <td></td>
    </tr>
    <tr>
        <td>DeepSeek R1 Distil Llama 70B</td>
        <td></td>
    </tr>"""

    table += """<tr>
            <td rowspan="2"><a href="https://cohere.com" target="_blank">Cohere</a></td>
            <td rowspan="2"><a href="https://docs.cohere.com/docs/rate-limits">20 requests/min<br>1,000 requests/month</a></td>
            <td>Command-R</td>
            <td rowspan="2">Shared Limit</td>
        </tr>
        <tr>
            <td>Command-R+</td>
        </tr>"""

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

    for idx, model in enumerate(cloudflare_models):
        table += "<tr>"
        if idx == 0:
            table += '<td rowspan="' + str(len(cloudflare_models)) + '">'
            table += '<a href="https://developers.cloudflare.com/workers-ai" target="_blank">Cloudflare Workers AI</a>'
            table += "</td>"
            table += '<td rowspan="' + str(len(cloudflare_models)) + '">'
            table += '<a href="https://developers.cloudflare.com/workers-ai/platform/pricing/#free-allocation">10,000 tokens/day</a>'
            table += "</td>"
        table += f"<td>{model['name']}</td>"
        table += "<td></td>"
        table += "</tr>\n"

    table += """<tr>
        <td rowspan="6"><a href="https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden" target="_blank">Google Cloud Vertex AI</a></td>
        <td rowspan="6">Very stringent payment verification for Google Cloud.</td>
        <td><a href="https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.1-405b-instruct-maas" target="_blank">Llama 3.1 70B Instruct</a></td>
        <td>Llama 3.1 API Service free during preview.<br>60 requests/minute</td>
    </tr>
    <tr>
        <td><a href="https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.1-405b-instruct-maas" target="_blank">Llama 3.1 8B Instruct</a></td>
        <td>Llama 3.1 API Service free during preview.<br>60 requests/minute</td>
    </tr>
    <tr>
        <td><a href="https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.2-90b-vision-instruct-maas" target="_blank">Llama 3.2 90B Vision Instruct</a></td>
        <td>Llama 3.2 API Service free during preview.<br>30 requests/minute</td>
    </tr>
    <tr>
        <td><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental" target="_blank">Gemini 2.0 Flash Experimental</a></td>
        <td rowspan="3">Experimental Gemini model.<br>10 requests/minute</td>
    </tr>
    <tr>
        <td><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental" target="_blank">Gemini 2.0 Flash Thinking Experimental</a></td>
    </tr>
    <tr>
        <td><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental" target="_blank">Gemini 2.0 Experimental</a></td>
    </tr>"""

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

        trial_table += f"<td>{get_human_limits(model)}</td>"
        trial_table += f"<td>{model['name']}</td>"
        trial_table += "</tr>\n"

    if MISSING_MODELS:
        logger.warning("Missing models:")
        logger.warning("\n" + "\n".join([f'"{model}": "{model}",' for model in MISSING_MODELS]))

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
