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
# Visual Programming Tools

Related: [Frontend Whiteboard, Node(-y) Tools.md](./Frontend%20Whiteboard,%20Node\(-y\)%20Tools.md), [Workflow Orchestration Components.md](./Workflow%20Orchestration%20Components.md)

Data in, data out, with some visual draggy low-code interface

## AI Automation

<https://github.com/lmnr-ai/flow>

### Dify

[Dify.ai](Dify.ai)

[Dify builder](https://cloud.dify.ai/apps)

Basically ComfyUI but for multi-modal

### ComfyUI

`#using`

### ChainForge

`#using`

> ChainForge is an open-source visual programming environment for prompt engineering, LLM evaluation and experimentation. With ChainForge, you can evaluate the robustness of prompts and text generation models with little to no coding required. 

Basically a ComfyUI + text-generation-webui, but geared toward [Model Evals and Scoring.md](./Model%20Evals%20and%20Scoring.md)

> <https://chainforge.ai/docs/>\
> <https://github.com/ianarawjo/ChainForge>

+ Features include:

   - **Query multiple LLMs at once** to test prompt ideas and variations quickly and effectively. Or query the same LLM at different settings.

   - **Compare response quality across prompt permutations, across models, and across model settings** to choose the best prompt and model for your use case. 

   - **Setup evaluation metrics** with code or LLM-based scorers and automatically plot results across prompts, prompt parameters, models, or model settings.

   - **Hold multiple conversations at once across template parameters and chat models.**Template chat messages and inspect and evaluate outputs at each turn of a chat conversation.

+ Visual programming UI:

   ![Pasted 2025-03-14-12-41-44.png](./Visual%20Programming%20Tools-assets/Pasted%202025-03-14-12-41-44.png)

Doesn’t currently support running in Safari, due to lack of lookbehind in js regex. This is actually [supported in Safari as of recently](https://caniuse.com/js-regexp-lookbehind), and so I expect this will be remediated soon.

### Langflow

`#using` LangGraph Studio

<https://github.com/langflow-ai/langflow>

[Examples](https://github.com/langflow-ai/langflow_examples/tree/main/examples) (json files)

Backend: Python, Uv, hatch, Langchain, FastAPI, sqlalchemy, sqlite

Frontend: React, react-flow, tailwind, shadcn, vite

From [DataStax.md](./DataStax.md)

> Langflow is an intuitive visual flow builder. This drag-and-drop interface allows developers to create complex AI workflows without writing extensive code. You can easily connect different components, such as prompts, language models, and data sources, to build sophisticated AI applications.

![image 1.png](./Visual%20Programming%20Tools-assets/image%201.png)

## Generic Workflow Automation

### Node-red

> Low-code programming for event-driven applications
>
> [nodered.org](nodered.org)
>
> <https://github.com/node-red/node-red>

Open source

Self-hosted only

Active

Javascript [node.js.md](./node.js.md)

### Huginn

<https://github.com/huginn/huginn>

Open source

Self-hosted

Low activty

### n8n

<https://github.com/n8n-io/n8n>

Doesnt support Apple/iOS targets

Low-code, but developer-friendly

Open source

Paid hosting

Has a beta [LangChain.md](./LangChain.md) integration

[n8n.md](./n8n.md)

### Node-red and n8n Comparison

[Comparison! node-red and n8n.md](./Comparison!%20node-red%20and%20n8n.md)

> N8n can accept events or can be triggered via web hook, node red can have listeners listening on different ports using different protocols.

## Hosted Low-code

[Make.com.md](./Make.com.md)

[Zapier.md](./Zapier.md)

[IFTTT.md](./IFTTT.md)

### Developer-focused

[Pipedream.md](./Pipedream.md)

## Libraries

### Rete

> JavaScript framework for visual programming
>
> <https://github.com/retejs/rete>