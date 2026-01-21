# LiteLLM Tuning

Short answer: you don’t need a separate HTTP server just to “allow multiple threads.” **LiteLLM already runs on Uvicorn** and you scale concurrency by running **multiple workers** (processes). For heavier traffic you add a **reverse proxy (optional)** for TLS/hardening and **horizontal scale** LiteLLM behind it, sharing quotas/rate-limits via **Redis**.

Here’s a practical way to run it “production-style” on your Mac/OrbStack box today, plus how folks run it in larger deployments.

---

## What to do on your Mac Mini (OrbStack)

### 1) Run more LiteLLM workers (process concurrency)

Give LiteLLM more Uvicorn workers so multiple requests stream in parallel:

```yaml
# docker-compose.yml
services:
  litellm:
    image: ghcr.io/berriai/litellm:main-stable
    network_mode: "host"   # so it binds on the Mac's ports directly
    command: ["--port","4000","--config","/app/config.yaml","--num_workers","8"]
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    env_file:
      - ./.env
```

`--num_workers` is the supported flag (also via `NUM_WORKERS` env var). A common rule is **1 worker per CPU core**; you can tune up/down based on latency and memory. ([LiteLLM](https://docs.litellm.ai/docs/proxy/cli?utm_source=chatgpt.com "CLI Arguments"))

> Rationale: Uvicorn workers are separate processes that handle requests concurrently. You don’t need threads or a separate server just to fan out connections.

### 2) (Optional) Put a reverse proxy in front

You **don’t need** Nginx/Caddy for concurrency, but it’s useful if you want **TLS**, **HTTP/2**, request size limits, and **nice timeouts**. If you proxy streaming (SSE), remember to **disable proxy buffering** so tokens flush continuously:

```nginx
# nginx.conf (snippet)
location / {
  proxy_pass http://127.0.0.1:4000;
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

  # Streaming (SSE) friendly:
  proxy_buffering off;
  proxy_read_timeout 1h;
  # If your app sets it, honor this:
  # add_header X-Accel-Buffering no;
}
```

Uvicorn’s own deployment notes recommend running behind a proxy for self-hosted setups. When proxying SSE, turning buffering off (or returning `X-Accel-Buffering: no`) avoids chunking delays. ([uvicorn.dev](https://uvicorn.dev/deployment/?utm_source=chatgpt.com "Deployment"))

### 3) Rate limits, budgets, and shared quotas

If multiple local clients will hammer the gateway, use LiteLLM’s built-ins:

- **RPM/TPM limits**, **max parallel requests**, per-key/per-team budgets (enable via admin or config). ([LiteLLM](https://docs.litellm.ai/docs/proxy/users?utm_source=chatgpt.com "Budgets, Rate Limits"))

- If you ever run **more than one LiteLLM instance**, point them to **Redis** so limits/usage are shared across instances. ([LiteLLM](https://docs.litellm.ai/docs/proxy/load_balancing?utm_source=chatgpt.com "Proxy - Load Balancing"))

### 4) Caching (optional)

LiteLLM supports response caching via config if your usage benefits from it (e.g., same prompts). ([LiteLLM](https://docs.litellm.ai/docs/proxy/caching?utm_source=chatgpt.com "Caching"))

---

## How people run LiteLLM “in production”

**Single node / edge box (your case):**

- **Multiple Uvicorn workers** (match CPU cores), optional **reverse proxy** for TLS/H2 and clean timeouts. ([LiteLLM](https://docs.litellm.ai/docs/proxy/prod?utm_source=chatgpt.com "Best Practices for Production"))

**Kubernetes / autoscale:**

- Run **several LiteLLM replicas** (workers per pod ≈ vCPU per pod), fronted by an Ingress/Load-Balancer. Share **RPM/TPM** and cooldowns via **Redis** so limits are global. LiteLLM’s docs show the “match workers to CPU” pattern and k8s examples. ([LiteLLM](https://docs.litellm.ai/docs/proxy/prod?utm_source=chatgpt.com "Best Practices for Production"))

**Reverse proxy in front (common, but optional):**

- Teams often put **Nginx/Traefik/Caddy** in front for TLS, logging, IP allow-lists, request size caps, and to publish a clean hostname. If you do this with streaming, ensure buffering is off so SSE tokens flush. ([uvicorn.dev](https://uvicorn.dev/deployment/?utm_source=chatgpt.com "Deployment"))

**Governance:**

- Use LiteLLM’s **virtual keys**, **teams**, **budgets**, and **rate-limits**; recent releases add priority-based quotas for critical workloads. ([LiteLLM](https://docs.litellm.ai/docs/proxy/users?utm_source=chatgpt.com "Budgets, Rate Limits"))

---

## Quick sizing guidance

- Start with **workers = CPU cores** (e.g., 8). Load test; if latency rises, raise workers until CPU/memory become the bottleneck. LiteLLM also supports streaming well; the main limiter becomes backend throughput (LM Studio or remote providers). ([LiteLLM](https://docs.litellm.ai/docs/proxy/prod?utm_source=chatgpt.com "Best Practices for Production"))

- For **SSE** heavy traffic, a proxy is helpful mostly for TLS & connection lifecycle, not throughput per se—just keep buffering off. ([uvicorn.dev](https://uvicorn.dev/deployment/?utm_source=chatgpt.com "Deployment"))

- If you later need **more headroom**, run **2–3 LiteLLM containers** on the same host (different ports) and round-robin them with your proxy; plug all of them into **Redis** so RPM/TPM limits remain accurate across instances. ([LiteLLM](https://docs.litellm.ai/docs/proxy/load_balancing?utm_source=chatgpt.com "Proxy - Load Balancing"))

---

## TL;DR

- **No** extra HTTP server is required just to handle many local clients.

- Use `--num_workers <N>` to scale concurrency inside LiteLLM. ([LiteLLM](https://docs.litellm.ai/docs/proxy/cli?utm_source=chatgpt.com "CLI Arguments"))

- Add a **reverse proxy** only for TLS/edge concerns; if you do, disable buffering for streaming. ([uvicorn.dev](https://uvicorn.dev/deployment/?utm_source=chatgpt.com "Deployment"))

- For bigger loads, run **multiple LiteLLM instances** and share **limits via Redis**. ([LiteLLM](https://docs.litellm.ai/docs/proxy/load_balancing?utm_source=chatgpt.com "Proxy - Load Balancing"))

If you want, I can drop a ready-to-run `docker-compose.yml` that sets workers from CPU count, adds optional Nginx with SSE-safe settings, and a Redis sidecar for shared quotas.