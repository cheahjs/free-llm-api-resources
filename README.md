# Free LLM resources

This lists various services that provide free access or credits towards API-based LLM usage.

**This repo explicitly excludes any services that are not legitimate (eg reverse engineers an existing chatbot)**

## Free Providers

<table>
    <thead>
        <tr>
            <th>Provider</th>
            <th>Provider Limits/Notes</th>
            <th>Model</th>
            <th>Model Limits</th>
        </tr>
    </thead>
    <tbody>
        <!-- OpenRouter Start -->
        <tr>
            <td rowspan="11"><a href="https://openrouter.ai">OpenRouter</a></td>
            <td></td>
            <td>Llama 3 8B Instruct</td>
            <td>20 req/min, 200 req/day</td>
        </tr>
        <tr>
            <td></td>
            <td>Gemma 2 9B</td>
            <td>20 req/min, 200 req/day</td>
        </tr>
        <tr>
            <td></td>
            <td>Gemma 7B</td>
            <td>20 req/min, 200 req/day</td>
        </tr>
        <tr>
            <td></td>
            <td>Mistral 7B Instruct</td>
            <td>20 req/min, 200 req/day</td>
        </tr>
        <tr>
            <td></td>
            <td>Phi-3 Mini 128K Instruct</td>
            <td>10 req/day (shared among Phi-3 models)</td>
        </tr>
        <tr>
            <td></td>
            <td>Phi-3 Medium 128K Instruct</td>
            <td>10 req/day (shared among Phi-3 models)</td>
        </tr>
        <tr>
            <td></td>
            <td>Zephyr 7B</td>
            <td>20 req/min, 200 req/day</td>
        </tr>
        <tr>
            <td></td>
            <td>Capybara 7B</td>
            <td>20 req/min, 200 req/day</td>
        </tr>
        <tr>
            <td></td>
            <td>OpenChat 3.5 7B</td>
            <td>20 req/min, 200 req/day</td>
        </tr>
        <tr>
            <td></td>
            <td>MythoMist 7B</td>
            <td>20 req/min, 200 req/day</td>
        </tr>
        <tr>
            <td></td>
            <td>Toppy M 7B</td>
            <td>20 req/min, 200 req/day</td>
        </tr>
        <!-- OpenRouter End -->
        <!-- GroqCloud Start -->
        <tr>
            <td rowspan="7"><a href="https://console.groq.com">GroqCloud</a></td>
        </tr>
        <tr>
            <td></td>
            <td>Gemma 2 9B Instruct</td>
            <td>30 req/min, 14,400 req/day, 15,000 tokens/min</td>
        </tr>
        <tr>
            <td></td>
            <td>Gemma 7B Instruct</td>
            <td>30 req/min, 14,400 req/day, 15,000 tokens/min</td>
        </tr>
        <tr>
            <td></td>
            <td>Llama 3 70B</td>
            <td>30 req/min, 14,400 req/day, 6,000 tokens/min</td>
        </tr>
        <tr>
            <td></td>
            <td>Llama 3 8B</td>
            <td>30 req/min, 14,400 req/day, 30,000 tokens/min</td>
        </tr>
        <tr>
            <td></td>
            <td>Mixtral 8x7B</td>
            <td>30 req/min, 14,400 req/day, 5,000 tokens/min</td>
        </tr>
        <tr>
            <td></td>
            <td>Whisper Large V3</td>
            <td>20 req/min, 2,000 req/day, 7,200 audio-seconds/hour, 28,800 audio-seconds/day</td>
        </tr>
        <!-- GroqCloud End -->
        <!-- Gemini Start -->
        <tr>
            <td rowspan="5"><a href="https://aistudio.google.com">Google AI Studio</a></td>
            <td rowspan="3">Free tier Gemini API access not available within UK/CH/EEA/EU<br>Data is trained upon in the free tier.</td>
            <td>Gemini 1.5 Flash</td>
            <td>15 req/min, 1,500 req/day, 1 million tokens/min</td>
        </gr>
        <tr>
            <td>Gemini 1.5 Pro</td>
            <td>2 req/min, 50 req/day, 32,000 tokens/min</td>
        </tr>
        <tr>
            <td>Gemini 1.0 Pro</td>
            <td>15 req/min, 1,500 req/day, 32,000 tokens/min</td>
        </tr>
        <tr>
            <td rowspan="2">Embeddings are available in UK/CH/EEA/EU</td>
            <td>text-embedding-004</td>
            <td>1,500 req/min, 100 content/batch</td>
        </tr>
        <tr>
            <td>embedding-001</td>
            <td>1,500 req/min, 100 content/batch</td>
        </tr>
        <!-- Gemini End -->
        <!-- Cohere Start -->
        <tr>
            <td rowspan="2"><a href="https://cohere.com">Cohere</a></td>
            <td rowspan="2">10 req/min, 1,000 req/month</td>
            <td>Command-R</td>
            <td>Shared Limit</td>
        </tr>
        <tr>
            <td>Command-R+</td>
            <td>Shared Limit</td>
        </tr>
        <!-- Cohere End -->
        <!-- HuggingFace Start -->
        <tr>
            <td><a href="https://huggingface.co/docs/api-inference/en/index">HuggingFace Serverless Inference</a></td>
            <td>Dynamic Rate Limits.<br>Limited to models smaller than 10GB.<br>Some popular models are supported even if they exceed 10GB.</td>
            <td>Various open models</td>
            <td></td>
        </tr>
        <!-- HuggingFace End -->
        <!-- OVH AI Endpoints Start -->
        <tr>
            <td rowspan="7"><a href="https://endpoints.ai.cloud.ovh.net/">OVH AI Endpoints (Free Alpha)</a></td>
            <td rowspan="7">Token expires every 2 weeks</td>
            <td>Llama 3 70B Instruct</td>
            <td>12 req/min</td>
        </tr>
        <tr>
            <td>Llama 3 8B Instruct</td>
            <td>12 req/min</td>
        </tr>
        <tr>
            <td>Mixtral 8x22B Instruct</td>
            <td>12 req/min</td>
        </tr>
        <tr>
            <td>Mixtral 8x7B Instruct</td>
            <td>12 req/min</td>
        </tr>
        <tr>
            <td>Mistral 7B Instruct</td>
            <td>12 req/min</td>
        </tr>
        <tr>
            <td>Llama 2 13B Chat</td>
            <td>12 req/min</td>
        </tr>
        <tr>
            <td>CodeLlama 13B Instruct</td>
            <td>12 req/min</td>
        </tr>
        <!-- OVH AI Endpoints End -->
        <!-- Cloudflare Workers AI Start -->
        <tr>
            <td rowspan="33"><a href="https://developers.cloudflare.com/workers-ai/">Cloudflare Workers AI</a></td>
            <td rowspan="33">10,000 neurons/day.<br>Beta models have unlimited usage.<br>Typical 300 req/min for text models.</td>
            <td>Llama 2 7B Chat</td>
            <td></td>
        </tr>
        <tr>
            <td>Mistral 7B Instruct</td>
            <td></td>
        </tr>
        <tr>
            <td>Deepseek Coder 6.7B Base AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>Deepseek Coder 6.7B Instruct AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>Deepseek Math 7B Base</td>
            <td></td>
        </tr>
        <tr>
            <td>Deepseek Math 7B Instruct</td>
            <td></td>
        </tr>
        <tr>
            <td>DiscoLM German 7B AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>Falcon 7B Instruct</td>
            <td></td>
        </tr>
        <tr>
            <td>Gemma 2B Instruct LoRA</td>
            <td></td>
        </tr>
        <tr>
            <td>Gemma 7B Instruct</td>
            <td></td>
        </tr>
        <tr>
            <td>Gemma 7B Instruct LoRA</td>
            <td></td>
        </tr>
        <tr>
            <td>Hermes 2 Pro Mistral 7B</td>
            <td></td>
        </tr>
        <tr>
            <td>Llama 2 13B Chat AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>Llama 2 7B Chat LoRA</td>
            <td></td>
        </tr>
        <tr>
            <td>Llama 3 8B Instruct</td>
            <td></td>
        </tr>
        <tr>
            <td>Llama 3 8B Instruct AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>Llama Guard 7B AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>Mistral 7B Instruct v0.1 AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>Mistral 7B Instruct v0.2</td>
            <td></td>
        </tr>
        <tr>
            <td>Mistral 7B Instruct v0.2 LoRA</td>
            <td></td>
        </tr>
        <tr>
            <td>Neural Chat 7B v3.1 AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>OpenChat 3.5</td>
            <td></td>
        </tr>
        <tr>
            <td>OpenHermes 2.5 Mistral 7B AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>Phi 2</td>
            <td></td>
        </tr>
        <tr>
            <td>Qwen 1.5 0.5B Chat</td>
            <td></td>
        </tr>
        <tr>
            <td>Qwen 1.5 1.8B Chat</td>
            <td></td>
        </tr>
        <tr>
            <td>Qwen 1.5 14B Chat AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>Qwen 1.5 7B Chat AWQ</td>
            <td></td>
        </tr>
        <tr>
            <td>SQLCoder 7B</td>
            <td></td>
        </tr>
        <tr>
            <td>Starling LM 7B</td>
            <td></td>
        </tr>
        <tr>
            <td>TinyLlama 1.1B Chat</td>
            <td></td>
        </tr>
        <tr>
            <td>Una Cybertron 7B v2</td>
            <td></td>
        </tr>
        <tr>
            <td>Zephyr 7B AWQ</td>
            <td></td>
        </tr>
        <!-- Cloudflare Workers AI End -->
    </tbody>
</table>

## Providers with trial credits

<table>
    <thead>
        <tr>
            <th>Provider</th>
            <th>Credits</th>
            <th>Models</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://console.anthropic.com/">Anthropic</a></td>
            <td>$5 for 14 days</td>
            <td>Claude 3</td>
        </tr>
        <tr>
            <td><a href="https://console.mistral.ai/">Mistral</a></td>
            <td>$5 for 14 days</td>
            <td>Mistral Open/Proprietary Models</td>
        </tr>
        <tr>
            <td><a href="https://together.ai">Together</a></td>
            <td>$5</td>
            <td>Various open models</td>
        </tr>
        <tr>
            <td><a href="https://fireworks.ai/">Fireworks</a></td>
            <td>$1</td>
            <td>Various open models</td>
        </tr>
        <tr>
            <td><a href="https://octo.ai/">OctoAI</a></td>
            <td>$10</td>
            <td>Various open models</td>
        </tr>
        <tr>
            <td><a href="https://unify.ai/">Unify</a></td>
            <td>$10</td>
            <td>Routes to other providers, various open models and proprietary models (OpenAI, Anthropic, Mistral, Perplexity)</td>
        </tr>
        <tr>
            <td><a href="https://deepinfra.com/">DeepInfra</a></td>
            <td>$1.80</td>
            <td>Various open models</td>
        </tr>
        <tr>
            <td><a href="https://build.nvidia.com/explore/discover">NVIDIA NIM</a></td>
            <td>1000 API calls</td>
            <td>Various open models</td>
        </tr>
    </tbody>
</table>