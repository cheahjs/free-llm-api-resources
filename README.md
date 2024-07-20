# Free LLM API resources

This lists various services that provide free access or credits towards API-based LLM usage.

**This repo explicitly excludes any services that are not legitimate (eg reverse engineers an existing chatbot)**

## Free Providers

<table>
    <thead>
        <tr>
            <th>Provider</th>
            <th>Provider Limits/Notes</th>
            <th>Model Name</th>
            <th>Model Limits</th>
        </tr>
    </thead>
    <tbody>
<tr><td rowspan="8"><a href="https://console.groq.com">Groq</a></td><td rowspan="8"></td><td>Gemma 2 9B Instruct</td><td>14400 requests/day<br>15000 tokens/minute</td></tr>
<tr><td>Gemma 7B Instruct</td><td>14400 requests/day<br>15000 tokens/minute</td></tr>
<tr><td>Llama 3 70B</td><td>14400 requests/day<br>6000 tokens/minute</td></tr>
<tr><td>Llama 3 70B - Groq Tool Use Preview</td><td>14400 requests/day<br>15000 tokens/minute</td></tr>
<tr><td>Llama 3 8B</td><td>14400 requests/day<br>30000 tokens/minute</td></tr>
<tr><td>Llama 3 8B - Groq Tool Use Preview</td><td>14400 requests/day<br>15000 tokens/minute</td></tr>
<tr><td>Mixtral 8x7B</td><td>14400 requests/day<br>5000 tokens/minute</td></tr>
<tr><td>Whisper Large v3</td><td>7200 audio-seconds/minute<br>2000 requests/day</td></tr>
<tr><td rowspan="12"><a href="https://openrouter.ai">OpenRouter</a></td><td rowspan="12"></td><td>Gemma 2 9B Instruct</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Gemma 7B Instruct</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Llama 3 8B Instruct</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Mistral 7B Instruct</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Mythomist 7B</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Nous Capybara 7B</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>OpenChat 7B</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Phi-3 Medium 128k Instruct</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Phi-3 Mini 128k Instruct</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Qwen 2 7B Instruct</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Toppy M 7B</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr><td>Zephyr 7B Beta</td><td>20 requests/minute<br>200 requests/day</td></tr>
<tr>
            <td rowspan="5"><a href="https://aistudio.google.com">Google AI Studio</a></td>
            <td rowspan="3">Free tier Gemini API access not available within UK/CH/EEA/EU/<br>Data is used for training.</td>
            <td>Gemini 1.5 Flash</td>
            <td>15 requests/min<br>1500 requests/day<br>1 million tokens/min</td>
        </tr>
        <tr>
            <td>Gemini 1.5 Pro</td>
            <td>2 request/min<br>50 requests/day<br>32000 tokens/min</td>
        </tr>
        <tr>
            <td>Gemini 1.0 Pro</td>
            <td>15 requests/min<br>1500 requests/day<br>32000 tokens/min</td>
        </tr>
        <tr>
            <td rowspan="2">Embeddings are available in UK/CH/EEA/EU</td>
            <td>text-embedding-004</td>
            <td>1500 requests/min<br>100 content/batch</td>
        </tr>
        <tr>
            <td>embedding-001</td>
            <td>1500 requests/min<br>100 content/batch</td>
        </tr><tr>
            <td rowspan="2"><a href="https://cohere.com">Cohere</a></td>
            <td rowspan="2">10 requests/min<br>1000 requests/month</td>
            <td>Command-R</td>
            <td>Shared Limit</td>
        </tr>
        <tr>
            <td>Command-R+</td>
            <td>Shared Limit</td>
        </tr><tr>
            <td><a href="https://huggingface.co/docs/api-inference/en/index">HuggingFace Serverless Inference</a></td>
            <td>Dynamic Rate Limits.<br>Limited to models smaller than 10GB.<br>Some popular models are supported even if they exceed 10GB.</td>
            <td>Various open models</td>
            <td></td>
        </tr><tr><td rowspan="7"><a href="https://endpoints.ai.cloud.ovh.net/">OVH AI Endpoints (Free Alpha)</a></td><td rowspan="7">Token expires every 2 weeks.</td><td>CodeLlama 13B Instruct</td><td>12 requests/minute</td></tr>
<tr><td>Llama 2 13B Chat</td><td>12 requests/minute</td></tr>
<tr><td>Llama 3 70B Instruct</td><td>12 requests/minute</td></tr>
<tr><td>Llama 3 8B Instruct</td><td>12 requests/minute</td></tr>
<tr><td>Mistral 7B Instruct</td><td>12 requests/minute</td></tr>
<tr><td>Mixtral 8x22B Instruct</td><td>12 requests/minute</td></tr>
<tr><td>Mixtral 8x7B Instruct</td><td>12 requests/minute</td></tr>
<tr><td rowspan="34"><a href="https://developers.cloudflare.com/workers-ai">Cloudflare Workers AI</a></td><td rowspan="34">10000 neurons/day<br>Beta models have unlimited usage.<br>Typically 300 requests/min for text models.</td><td>Deepseek Coder 6.7B Base (AWQ)</td><td></td></tr>
<tr><td>Deepseek Coder 6.7B Instruct (AWQ)</td><td></td></tr>
<tr><td>Deepseek Math 7B Instruct</td><td></td></tr>
<tr><td>Discolm German 7B v1 (AWQ)</td><td></td></tr>
<tr><td>Falcom 7B Instruct</td><td></td></tr>
<tr><td>Gemma 2B Instruct (LoRA)</td><td></td></tr>
<tr><td>Gemma 7B Instruct</td><td></td></tr>
<tr><td>Gemma 7B Instruct (LoRA)</td><td></td></tr>
<tr><td>Hermes 2 Pro Mistral 7B</td><td></td></tr>
<tr><td>Llama 2 13B Chat (AWQ)</td><td></td></tr>
<tr><td>Llama 2 7B Chat (FP16)</td><td></td></tr>
<tr><td>Llama 2 7B Chat (INT8)</td><td></td></tr>
<tr><td>Llama 2 7B Chat (LoRA)</td><td></td></tr>
<tr><td>Llama 3 8B Instruct</td><td></td></tr>
<tr><td>Llama 3 8B Instruct</td><td></td></tr>
<tr><td>Llama 3 8B Instruct (AWQ)</td><td></td></tr>
<tr><td>LlamaGuard 7B (AWQ)</td><td></td></tr>
<tr><td>Mistral 7B Instruct v0.1</td><td></td></tr>
<tr><td>Mistral 7B Instruct v0.1 (AWQ)</td><td></td></tr>
<tr><td>Mistral 7B Instruct v0.2</td><td></td></tr>
<tr><td>Mistral 7B Instruct v0.2 (LoRA)</td><td></td></tr>
<tr><td>Neural Chat 7B v3.1 (AWQ)</td><td></td></tr>
<tr><td>OpenChat 3.5 0106</td><td></td></tr>
<tr><td>OpenHermes 2.5 Mistral 7B (AWQ)</td><td></td></tr>
<tr><td>Phi-2</td><td></td></tr>
<tr><td>Qwen 1.5 0.5B Chat</td><td></td></tr>
<tr><td>Qwen 1.5 1.8B Chat</td><td></td></tr>
<tr><td>Qwen 1.5 14B Chat (AWQ)</td><td></td></tr>
<tr><td>Qwen 1.5 7B Chat (AWQ)</td><td></td></tr>
<tr><td>SQLCoder 7B 2</td><td></td></tr>
<tr><td>Starling LM 7B Beta</td><td></td></tr>
<tr><td>TinyLlama 1.1B Chat v1.0</td><td></td></tr>
<tr><td>Una Cybertron 7B v2 (BF16)</td><td></td></tr>
<tr><td>Zephyr 7B Beta (AWQ)</td><td></td></tr>
</tbody></table>

## Providers with trial credits

<table>
    <thead>
        <tr>
            <th>Provider</th>
            <th>Credits</th>
            <th>Requirements</th>
            <th>Models</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://console.anthropic.com/">Anthropic</a></td>
            <td>$5 for 14 days</td>
            <td>Phone number verification</td>
            <td>Claude 3</td>
        </tr>
        <tr>
            <td><a href="https://console.mistral.ai/">Mistral</a></td>
            <td>$5 for 14 days</td>
            <td></td>
            <td>Mistral Open/Proprietary Models</td>
        </tr>
        <tr>
            <td><a href="https://together.ai">Together</a></td>
            <td>$5</td>
            <td></td>
            <td>Various open models</td>
        </tr>
        <tr>
            <td><a href="https://fireworks.ai/">Fireworks</a></td>
            <td>$1</td>
            <td></td>
            <td>Various open models</td>
        </tr>
        <tr>
            <td><a href="https://octo.ai/">OctoAI</a></td>
            <td>$10</td>
            <td></td>
            <td>Various open models</td>
        </tr>
        <tr>
            <td><a href="https://unify.ai/">Unify</a></td>
            <td>$10</td>
            <td></td>
            <td>Routes to other providers, various open models and proprietary models (OpenAI, Anthropic, Mistral, Perplexity)</td>
        </tr>
        <tr>
            <td><a href="https://deepinfra.com/">DeepInfra</a></td>
            <td>$1.80</td>
            <td></td>
            <td>Various open models</td>
        </tr>
        <tr>
            <td><a href="https://build.nvidia.com/explore/discover">NVIDIA NIM</a></td>
            <td>1000 API calls</td>
            <td></td>
            <td>Various open models</td>
        </tr>
    </tbody>
</table>