# LM Studio Tuning

Short version: **LM Studio doesn’t expose a “HTTP workers / threads” knob** for its server like Uvicorn/Gunicorn. Its OpenAI-compatible server handles requests and **queues per-model inference**; you scale *throughput* mainly by (a) speeding up each request (model “threads”, batch size, GPU offload), and (b) **running multiple model instances or multiple LM Studio servers** and load-balancing in front of them (e.g., with your LiteLLM gateway).

Here’s what’s available today and what I recommend.

---

## What LM Studio lets you tune

- **Server settings:** Port, CORS, “serve on local network”, etc. — no documented server-level concurrency/worker setting. ([LM Studio](https://lmstudio.ai/docs/cli/server-start?utm_source=chatgpt.com "lms server start | LM Studio Docs"))

- **Per-model performance knobs:** CPU thread count, context length, batch size, GPU offload (Metal), etc. These increase *single-request* throughput, not server worker concurrency. ([hongkiat.com](https://www.hongkiat.com/blog/?p=73375&utm_source=chatgpt.com "Running Large Language Models (LLMs) Locally with LM ..."))

- **Concurrency behavior:** Historically, parallel requests to the *same* model were queued; recent releases improved handling of parallel tool calls/streaming, but there’s still no public “max parallel requests” slider like Ollama’s. ([GitHub](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/204?utm_source=chatgpt.com "Queued request always return empty response · Issue #204"))

---

## Practical ways to scale LM Studio behind LiteLLM

### 1) Keep a **single LM Studio server**, load **multiple models**

LM Studio supports multiple models loaded at once; each can serve requests in parallel to some extent (with queuing per model). This is the simplest setup and works well when traffic is spread across different models (two chat models + embeddings). ([LM Studio](https://lmstudio.ai/docs/typescript/manage-models/loading?utm_source=chatgpt.com "Manage Models in Memory | LM Studio Docs"))

### 2) Run **multiple LM Studio instances** on different ports (horizontal scale on one Mac)

If one busy model becomes your bottleneck, spin up **additional LM Studio servers** (or additional instances of the *same* model) on new ports and **round-robin** across them in LiteLLM:

- Start extra servers (examples):

   ```
   lms server start --port 1234
   lms server start --port 1235
   ```

   (Running multiple instances is a supported use-case discussed by users; each instance hosts models on its own port.) ([GitHub](https://github.com/lmstudio-ai/lms/issues/159?utm_source=chatgpt.com "Run multiple LM Studio instances on Ubuntu? · Issue #159"))

- In **LiteLLM `config.yaml`**, treat each LM Studio port as an **OpenAI-compatible backend** by setting `api_base` per target; then define a LiteLLM **router** or **multiple routes** and round-robin between them:

```yaml
model_list:
  - model_name: local-llama
    litellm_params:
      # point at LM Studio server #1
      model: openai/llama-3.1-8b-instruct
      api_base: http://127.0.0.1:1234/v1
      api_key: "lm-studio"   # if LM Studio is open, any string works

  - model_name: local-llama-b
    litellm_params:
      # point at LM Studio server #2
      model: openai/llama-3.1-8b-instruct
      api_base: http://127.0.0.1:1235/v1
      api_key: "lm-studio"
```

Then use LiteLLM’s **router/fallback** features to distribute load across these two identical backends (simple round-robin or failover). LiteLLM explicitly supports OpenAI-compatible backends with per-route `api_base`. ([LiteLLM](https://docs.litellm.ai/docs/providers/openai_compatible?utm_source=chatgpt.com "OpenAI-Compatible Endpoints"))

> Why use `openai/…` here instead of `lm_studio/...`? Because LiteLLM’s OpenAI-compatible provider lets you set a **custom `api_base`** per route (perfect for pointing at different LM Studio ports). ([LiteLLM](https://docs.litellm.ai/docs/providers/openai_compatible?utm_source=chatgpt.com "OpenAI-Compatible Endpoints"))

### 3) (Alternative) Multiple **instances of the same model** within LM Studio

Programmatically, LM Studio’s SDK lets you **load multiple instances** of the same model under different identifiers; each instance can process separately. You can then expose them via a single server or multiple servers, and route in LiteLLM. ([LM Studio](https://lmstudio.ai/docs/typescript/manage-models/loading?utm_source=chatgpt.com "Manage Models in Memory | LM Studio Docs"))

---

## Tuning tips

- **Per-model “threads”**: Set the model’s CPU thread count appropriately (often ≈ physical cores) and adjust batch/kv-cache to use your M4 Pro efficiently. This boosts tokens/sec for each request, which indirectly helps concurrency by finishing faster. ([GitHub](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/130?utm_source=chatgpt.com "LM-Studio 0.3.2 \"clamps\" the numbers of CPU cores from ..."))

- **Streaming (SSE)**: LM Studio streams via SSE; nothing to configure for “concurrent streams.” Your LiteLLM front end should run multiple workers; if you add a reverse proxy, disable buffering so streams flush. ([LM Studio](https://lmstudio.ai/docs/developer/api-changelog?utm_source=chatgpt.com "API Changelog | LM Studio Docs"))

- **Hot model mix**: If two LLMs + embeddings are loaded together, contention can happen on memory. Keep an eye on RAM/VRAM and consider dedicating a separate LM Studio instance/port to embeddings so chat traffic doesn’t queue behind long generation jobs.

---

## Bottom line

- **There’s no LM-Studio setting for “HTTP workers” or “max parallel requests.”** You scale concurrency by **running more instances** (ports or model instances) and **load-balancing in LiteLLM**, and by **tuning per-model performance** (threads/offload/batch). ([LM Studio](https://lmstudio.ai/docs/cli/server-start?utm_source=chatgpt.com "lms server start | LM Studio Docs"))

- Your earlier plan (LiteLLM as the single public gateway) is perfect: run **multiple LiteLLM workers** for client concurrency, and scale **LM Studio horizontally** behind it when one port becomes saturated. Use LiteLLM routes with per-backend `api_base` to spread load across `:1234`, `:1235`, … ([LiteLLM](https://docs.litellm.ai/docs/providers/openai_compatible?utm_source=chatgpt.com "OpenAI-Compatible Endpoints"))

If you want, I can drop a ready-to-run `docker-compose.yml` that:

- runs LiteLLM with `--num_workers` tuned to your CPU,

- defines two LM Studio backends on `:1234`/`:1235`,

- round-robins requests across them,

- and includes a tiny healthcheck script to auto-disable a backend if its queue grows too long.