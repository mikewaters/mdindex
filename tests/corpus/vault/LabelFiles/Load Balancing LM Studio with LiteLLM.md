# Load Balancing LM Studio with LiteLLM

Awesome — here’s a **ready-to-run** setup that puts **LiteLLM** in front, spreads load across **two LM Studio servers** on the same Mac (ports `1234` and `1235`), and stays friendly to OrbStack + macOS.

You’ll get:

- `docker-compose.yml` that runs **LiteLLM** with multiple workers (for parallel clients)

- `litellm.config.yaml` that **round-robins** across two LM Studio backends (same model, two ports)

- Optional **Nginx** reverse proxy (SSE-safe) if you want a single public port different from LiteLLM’s

- A tiny **healthcheck** and **Makefile** to smooth operations

- An **.env** template for API keys (OpenRouter/Anthropic, etc.)

---

# 1) Files

## a) `.env` (create next to compose)

```dotenv
# LiteLLM process
NUM_WORKERS=8                 # ~ CPU cores on your M4 Pro; tune as needed
LITELLM_MASTER_KEY=sk-local-gateway-CHANGE_ME
LITELLM_SALT_KEY=change_me_salt

# Local LM Studio servers (two instances/ports)
LMSTUDIO_A=http://127.0.0.1:1234/v1
LMSTUDIO_B=http://127.0.0.1:1235/v1
LMSTUDIO_API_KEY=lm-studio               # use if you configured LM Studio auth; otherwise keep dummy

# Optional external providers (LiteLLM can route to them too)
OPENROUTER_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

## b) `litellm.config.yaml`

```yaml
# OpenAI-compatible gateway configs
# Clients use: base_url = http://<HOST>:4000/v1  (or via nginx at 8080)
# and Authorization: Bearer ${LITELLM_MASTER_KEY}

# 1) Define TWO identical local backends that point to LM Studio ports
model_list:
  # Backend A (LM Studio on :1234)
  - model_name: local-llama-a
    litellm_params:
      provider: openai_compatible
      model: llama-3.1-8b-instruct          # The LM Studio model "name"/key
      api_base: ${LMSTUDIO_A}
      api_key: ${LMSTUDIO_API_KEY}

  # Backend B (LM Studio on :1235)
  - model_name: local-llama-b
    litellm_params:
      provider: openai_compatible
      model: llama-3.1-8b-instruct
      api_base: ${LMSTUDIO_B}
      api_key: ${LMSTUDIO_API_KEY}

  # Embeddings on LM Studio (choose your embedding model name)
  - model_name: local-embed-a
    litellm_params:
      provider: openai_compatible
      model: text-embedding-nomic-v1
      api_base: ${LMSTUDIO_A}
      api_key: ${LMSTUDIO_API_KEY}
  - model_name: local-embed-b
    litellm_params:
      provider: openai_compatible
      model: text-embedding-nomic-v1
      api_base: ${LMSTUDIO_B}
      api_key: ${LMSTUDIO_API_KEY}

  # Example: route to GPT-4 via OpenRouter
  - model_name: openai-gpt4
    litellm_params:
      provider: openrouter
      model: openai/gpt-4
      api_key: ${OPENROUTER_API_KEY}

  # Example: route to Anthropic directly
  - model_name: claude-3-5-sonnet
    litellm_params:
      provider: anthropic
      model: claude-3-5-sonnet-20241022
      api_key: ${ANTHROPIC_API_KEY}

# 2) A logical "router" model that ROUND-ROBINS across the two local backends
# Clients will use model: local-llama (single name), and LiteLLM will pick A/B.
router_settings:
  routers:
    - model_name: local-llama
      routing_strategy: simple_round_robin     # or "random_choice"
      routes:
        - local-llama-a
        - local-llama-b

    - model_name: local-embed
      routing_strategy: simple_round_robin
      routes:
        - local-embed-a
        - local-embed-b

# Optional: rate limits per key (enable if you want to prevent runaway usage)
litellm_settings:
  general_settings:
    master_key: ${LITELLM_MASTER_KEY}
  # Example: basic rate limits (customize or remove)
  # rate_limits:
  #   - key: "*"
  #     rpm: 1200
  #     tpm: 200000
```

> Tip: Replace `llama-3.1-8b-instruct` / `text-embedding-nomic-v1` with the **exact model keys** as shown by LM Studio for your loaded models.

## c) `docker-compose.yml`

```yaml
version: "3.9"

services:
  # LiteLLM gateway (OpenAI-compatible)
  litellm:
    image: ghcr.io/berriai/litellm:main-stable
    container_name: litellm-gateway
    # OrbStack supports host networking; this makes LiteLLM reachable on the Mac's IP directly.
    network_mode: "host"
    env_file:
      - ./.env
    volumes:
      - ./litellm.config.yaml:/app/config.yaml:ro
    command:
      [
        "--port","4000",
        "--config","/app/config.yaml",
        "--num_workers","${NUM_WORKERS}"
      ]
    healthcheck:
      # checks the gateway itself
      test: ["CMD", "wget", "-qO-", "http://127.0.0.1:4000/v1/models"]
      interval: 15s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  # OPTIONAL: SSE-friendly reverse proxy at :8080 -> LiteLLM :4000
  # Use if you want a single port for host/LAN/Tailscale different from 4000,
  # or you plan to add TLS later. Comment out if you don't need a proxy.
  nginx:
    image: nginx:stable
    container_name: litellm-proxy
    depends_on: [litellm]
    ports:
      - "8080:8080"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    restart: unless-stopped
```

## d) `nginx.conf` (optional, SSE streaming safe)

```nginx
server {
  listen 8080;
  # Adjust if you want a hostname, etc.

  location / {
    proxy_pass http://127.0.0.1:4000;  # LiteLLM on host network
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Connection "";
    # Important for streaming (SSE):
    proxy_buffering off;
    # Long generations need longer timeouts:
    proxy_read_timeout 1h;
    proxy_send_timeout 1h;
  }
}
```

## e) `Makefile` (quality-of-life)

```makefile
.PHONY: up down logs ps

up:
\tdocker compose up -d

down:
\tdocker compose down

logs:
\tdocker compose logs -f --tail=200

ps:
\tdocker compose ps
```

---

# 2) Start two LM Studio servers

You want **two LM Studio API servers** on the host, **:1234** and **:1235**, each able to serve the same models (or split roles, e.g., A for chat, B for embeddings). Do one of the following:

- **GUI route**: Open LM Studio → start the server on **1234**. Launch a **second LM Studio instance** (or use LM Studio’s headless/server CLI if you have it) and start it on **1235**. Load the same chat model on both if you want round-robin; or dedicate port B to embeddings if you prefer.

- **Headless idea**: If you have LM Studio CLI available, you can run two server processes:

   ```bash
   # Terminal 1
   lms server start --port 1234
   # Terminal 2
   lms server start --port 1235
   ```

   Ensure each server has the models you want loaded (or uses JIT loading).

> Sanity check:
>
> ```bash
> curl http://127.0.0.1:1234/v1/models
> curl http://127.0.0.1:1235/v1/models
> ```
>
> Both should return JSON.

---

# 3) Bring up LiteLLM (and optional proxy)

```bash
make up
# or: docker compose up -d
```

- **Host apps:** call `http://localhost:4000/v1` (or `http://localhost:8080/v1` if you enabled nginx).

- **Other containers (OrbStack):** call `http://host.docker.internal:4000/v1` (or `:8080` if proxy).

- **LAN devices:** call `http://<your-mac-LAN-ip>:4000/v1` (or `:8080`).

- **Tailscale clients:** call `http://<your-tailnet-ip>:4000/v1` (or `:8080`).

All clients use **one URL** and **choose models** by name:

- Local RR chat: `"model": "local-llama"`

- Local embeddings: `"model": "local-embed"`

- Cloud (OpenRouter GPT-4): `"model": "openai-gpt4"`

- Cloud (Anthropic): `"model": "claude-3-5-sonnet"`

Use `Authorization: Bearer ${LITELLM_MASTER_KEY}`.

---

# 4) Health & scaling notes

- **LiteLLM concurrency** = `NUM_WORKERS` processes. Start with CPU cores (e.g., 8) and tune.

- **LM Studio throughput** scales by:

   - running **two servers** (ports 1234 & 1235) and round-robin in LiteLLM,

   - tuning per-model performance (threads/offload/batch),

   - optionally dedicating one server to embeddings so long chats don’t queue behind them.

- **SSE streaming** works end-to-end. If you use the optional Nginx: buffering is **off**.

- **Future TLS**: swap Nginx for **Caddy** or terminate TLS at Nginx; point DNS to your Mac (LAN) or use **Tailscale Funnel** if exposing outside the tailnet.

---

# 5) Quick client smoke tests

**Curl (models list):**

```bash
curl -H "Authorization: Bearer $(grep LITELLM_MASTER_KEY .env | cut -d= -f2)" \
     http://127.0.0.1:4000/v1/models
```

**Chat (streams):**

```bash
curl -N -H "Content-Type: application/json" \
     -H "Authorization: Bearer $(grep LITELLM_MASTER_KEY .env | cut -d= -f2)" \
     -d '{"model":"local-llama","stream":true,"messages":[{"role":"user","content":"Say hi"}]}' \
     http://127.0.0.1:4000/v1/chat/completions
```

**Embeddings:**

```bash
curl -s -H "Content-Type: application/json" \
     -H "Authorization: Bearer $(grep LITELLM_MASTER_KEY .env | cut -d= -f2)" \
     -d '{"model":"local-embed","input":"hello world"}' \
     http://127.0.0.1:4000/v1/embeddings
```

---

# 6) Optional: simple backend readiness guard

If you want LiteLLM to *temporarily* avoid a sick LM Studio port, you can run this tiny script via launchd/cron to disable a backend by commenting it out (or swapping a router target) when `/v1/models` fails. In practice most folks rely on **two backends** and let one fail fast; you can then fix LM Studio and `make up` again.

---

If you want me to tailor the config to your **exact LM Studio model names** (and add OpenRouter/Anthropic entries you’ll use), paste those names and I’ll fill them in precisely.