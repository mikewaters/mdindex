---
tags:
  - document 📑
---
# FastAPI Service

## Alternative: Jina.ai

Really fast to shove together some code for multimodal without thinking about the BS

<https://github.com/jina-ai/serve>

> a framework for building and deploying multimodal AI services that communicate via gRPC, HTTP and WebSockets.

From [Jina.ai](Jina.ai), a model vendor and ML SaaS and API provider. [Reader API,](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/5aa5e89b-7c38-4e54-b0bc-4e0bc204f12a#eae96155-f810-452c-9441-4fb57649276e) [Embeddings API](https://jina.ai/embeddings/), [Reranker API](https://jina.ai/reranker/), [Classifier API](https://jina.ai/classifier/), [Segmenter API](https://jina.ai/segmenter/) many of these are open source/hostable

## Websockets

- <https://github.com/QuentinFuxa/whisper_streaming_web/blob/main/whisper_fastapi_online_server.py>

## LLMs

### LitServe

- Proxies/abstracts ML provider APIs

- Implements patterns like batch an d streaming (via SSE)

- Manages asgi lifecycle

> serving engine for AI models built on FastAPI. It augments FastAPI with features like batching, streaming, and GPU autoscaling

### Whisper streaming over ws

<https://github.com/QuentinFuxa/whisper_streaming_web/tree/main>

> Whisper Streaming with Websocket and Fastapi server