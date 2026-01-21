---
tags:
  - landscape
Document Type:
  - Technical
Topic:
  - software
Subject:
  - Solution
---
# ML Provider Proxies

Application egress-to-LLM-API proxies.

Related: [Prompt Routing Solutions.md](./Prompt%20Routing%20Solutions.md) (LLM ingress proxies)

### AISuite

> Simple, unified interface to multiple Generative AI providers

> `aisuite` makes it easy for developers to use multiple LLM through a standardized interface. Using an interface similar to OpenAI's, `aisuite` makes it easy to interact with the most popular LLMs and compare the results. It is a thin wrapper around python client libraries, and allows creators to seamlessly swap out and test responses from different LLM providers without changing their code. Today, the library is primarily focussed on chat completions. We will expand it cover more use cases in near future.
>
> <https://github.com/andrewyng/aisuite>

### LiteLLM

Referenced in [QuicKB](https://github.com/ALucek/QuicKB) in ALucek’s [video](https://youtu.be/kPTKlg-RDVs?si=V4pfkTGMRZ0ukm7C) 

### LitServe (based on FastAPI)

- works for both LLM and image genai

- runs locally

- can use gpu

- openai spec compat

<https://github.com/Lightning-AI/LitServe>

> easy-to-use, flexible serving engine for AI models built on FastAPI. It augments FastAPI with features like batching, streaming, and GPU autoscaling eliminate the need to rebuild a FastAPI server per model.

> [examples](https://github.com/Lightning-AI/LitServe?tab=readme-ov-file#examples)

From low-code AI vendor [Lightning.ai](https://lightning.ai):

> Focus on data and models - not infrastructure

<https://github.com/BerriAI/litellm>

<https://goapi.ai/>