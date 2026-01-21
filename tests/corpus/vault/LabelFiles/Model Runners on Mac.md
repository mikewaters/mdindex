# Model Runners on Mac

Inference engines etc

## Model Packaging

### OCI Model Artifact

Docker model runner uses custom OCI Artifacts <https://www.docker.com/blog/introducing-docker-model-runner/> 

It’s a subtype of OCI Artifact, defined here: <https://github.com/docker/model-spec>

### ModelKit

ModelKit is an OCI package standardized by KitOps <https://kitops.org/> and supports Mac.

Can be deployed to a container

Can include objects like Jupyter notebooks, codebases etc

Can run a model locally for development or CI, supports [GGUF GGML.md](./GGUF%20GGML.md) models and so is probably built on top of Llama.cpp

### Cog (Replicate)

> Cog is an open-source tool that lets you package machine learning models in a standard, production-ready container.
>
> You can deploy your packaged model to your own infrastructure, or to ++[Replicate](https://replicate.com/)++.

<https://github.com/replicate/cog>

### [Llamafile .md](./Llamafile%20.md)(Mozilla)

## Model Execution Tools

### Max (Modular)

<https://docs.modular.com/max/get-started/>

### VLLM

Not supported on Mac

### Ollama

Runs on top of [LLama.cpp.md](./LLama.cpp.md)

Supports OCI registries

Uses [GGUF GGML.md](./GGUF%20GGML.md)

### Docker Model Runner

Creates OCI images out of models, and runs on top of [LLama.cpp.md](./LLama.cpp.md)

### KitOps

## Model Execution Libraries

### [LLama.cpp.md](./LLama.cpp.md)

For running [GGUF GGML.md](./GGUF%20GGML.md) formatted models