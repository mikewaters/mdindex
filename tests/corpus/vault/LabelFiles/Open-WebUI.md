---
tags:
  - app/service
Installed State:
  - Trying
Environment:
  - Desktop
---
# Open-WebUI

Installed via `uv tools`

> **Open WebUI is an ++[extensible](https://docs.openwebui.com/features/plugin/)++, feature-rich, and user-friendly self-hosted AI platform designed to operate entirely offline.** It supports various LLM runners like **Ollama** and **OpenAI-compatible APIs**, with **built-in inference engine** for RAG, making it a **powerful AI deployment solution**.

> Open WebUI is an extensible, self-hosted AI interface that adapts to your workflow, all while operating entirely offline.

> <https://github.com/open-webui/open-webui>\
> <https://openwebui.com>

### Pipelines

<https://docs.openwebui.com/pipelines/>

Separate Docker container, the pipeline functions are executed there instead on the OpenWebUI container or host.

### Examples

- Pipelines: <https://zohaib.me/extending-openwebui-using-pipelines/>

- Channels bot: <https://github.com/open-webui/bot/blob/main/examples/ai.py>

### Interesting uses

#### Hosting on fly.io (local inference with Ollama)

Host the UI and Ollama on the same Fly machine

<https://github.com/fly-apps/ollama-open-webui/tree/e168239c26fb2548ee26d1e44e1df3ab1278497d>

#### YT transcripts

<https://github.com/ay4t/open-webui-tools>

#### ChannelGPT

> Open-WebUI Tool for searching information across a YouTube channel
>
> <https://github.com/NotYuSheng/ChannelGPT-Tool>

#### n8n Integration

[n8n Pipeline](https://github.com/owndev/Open-WebUI-Functions/blob/master/pipelines/n8n/n8n.py)

> Sends messages from Open WebUI to an N8N webhook.
>
> <https://github.com/owndev/Open-WebUI-Functions>

### Github topics

- <https://github.com/topics/open-webui>

- <https://github.com/topics/ollama-webui>

## Key differences between text-generation-webui and Open-WebUI 

\#claude-3.5-sonnet

Open-WebUI is specifically designed for Ollama and local LLM deployment, while text-generation-webui is more model-agnostic and supports a broader range of model formats and frameworks.

Key advantages of Open-WebUI:

- Tighter integration with Ollama's ecosystem

- More modern, polished UI with better mobile support

- Built-in function calling support

- Native multimodal capabilities (image input/output)

- Better container support and easier deployment

- Lower system requirements due to Ollama's optimization

Key advantages of text-generation-webui:

- Supports more model formats and frameworks

- More extensive training and fine-tuning capabilities

- More detailed model parameter controls

- Built-in character creation tools

- More extensive prompt templates system

- Better support for instruction-tuning

If you're primarily working with Ollama models, Open-WebUI would likely be the better choice due to its specialized integration and modern features. However, if you need broader model support or more extensive training capabilities, text-generation-webui might be more suitable.

Would you like me to elaborate on any specific aspect of these differences?