# Examples of Open WebUI Using Zep, Mem0, or Cognee

Direct, production-grade examples are sparse, but there are workable paths and community threads that show how to wire these together via Open WebUI’s Pipelines/MCP and OpenAI-compatible endpoints. Here are the best concrete references found, plus how they map to Zep, Mem0, and Cognee.

## What’s directly documented today

- Open WebUI supports OpenAI-compatible “Pipelines,” letting a middleware intercept and enrich prompts before the LLM call. This is the supported way to plug in external memory systems to Open WebUI without forking core code.[openwebui+2](https://docs.openwebui.com/openapi-servers/open-webui/)

- There’s an official guide showing how to run the Pipelines server and connect it to Open WebUI via an OpenAI API connection, which is exactly the mechanism to insert memory logic (search/store) between UI and model.[openwebui+2](https://docs.openwebui.com/features/)

## Zep + Open WebUI

- Zep has mature docs, SDKs, and an admin UI. While there isn’t a published “Zep + Open WebUI” turnkey repo, Zep’s long‑term memory API is straightforward to call from an Open WebUI Pipeline filter: retrieve relevant memories on each user message and append them to the system/prompt, then store the new exchange back to Zep.[github+2](https://github.com/getzep/docs.getzep.com/blob/main/docs/deployment/webui.md)

- Zep resources you can adapt into an Open WebUI Pipeline:

   - Zep’s open‑source server and examples/SDKs provide memory search/store endpoints and a web admin UI for sessions/memories.[github+1](https://github.com/getzep/zep)

   - LangChain Zep integration examples show how to wire Zep into a chat loop (these patterns translate 1:1 into a Pipeline filter function).[python.langchain+1](https://python.langchain.com/docs/integrations/providers/zep/)

- Security note: Zep’s web UI is not authenticated; only enable it on non-public endpoints per docs.[github](https://github.com/getzep/docs.getzep.com/blob/main/docs/deployment/webui.md)

Evidence:

- Zep Web UI and deployment docs (incl. security warning)[github](https://github.com/getzep/docs.getzep.com/blob/main/docs/deployment/webui.md)

- Zep repo with examples and integrations[github](https://github.com/getzep/zep)

- LangChain Zep integration pages (reference-level patterns)[python.langchain+1](https://python.langchain.com/docs/integrations/memory/zep_memory/)

- Zep intro and architecture overview for memory/enrichment[getzep](https://blog.getzep.com/introducing-zep-memory-ai/)

## Mem0 + Open WebUI

- Practical path: Use Open WebUI Pipelines to call Mem0’s SDK before/after the model call. This is supported architecturally (Pipeline filter can enrich prompts and then store memories after the response).[openwebui+1](https://docs.openwebui.com/openapi-servers/open-webui/)

- Example tutorial you can adapt:

   - A step-by-step Mem0 + LangGraph tutorial (with code) demonstrates add/search/update flows and Qdrant backing; the same pre/post pattern is what a Pipeline filter needs, so it’s a strong blueprint for Open [WebUI.youtube](WebUI.youtube)

- Community discussion specific to Open WebUI:

   - A GitHub discussion on “mem0 compatible with openwebui and ollama?” indicates demand and notes limitations around Ollama compatibility at the time; it suggests an integration would need an API/middleware—exactly what Pipelines provide.[github](https://github.com/open-webui/open-webui/discussions/4074)

Evidence:

- Open WebUI Pipelines connection and filter mechanisms[openwebui+1](https://docs.openwebui.com/features/)

- Mem0 + LangGraph integration tutorial/video (code and approach to embed into middleware)youtube

- Open WebUI discussion on Mem0 compatibility considerations[github](https://github.com/open-webui/open-webui/discussions/4074)

## Cognee + Open WebUI

- No public “Open WebUI + Cognee” repo surfaced. However, Cognee is a Python library designed to add memory (vector + graph) in “five lines,” and it’s commonly integrated as a preprocessing layer. This fits Open WebUI Pipelines: a Python filter can call Cognee’s add/search in pre/post hooks before forwarding the enriched prompt to the model.[github+2](https://github.com/topoteretes/cognee)

- Given Open WebUI’s “Install from GitHub URL” filter pattern (example shown for Langfuse), a Cognee filter can be packaged and installed similarly.[langfuse](https://langfuse.com/docs/integrations/openwebui)

Evidence:

- Cognee open-source repo and “5 lines” integration positioning[cohorte+2](https://www.cohorte.co/blog/cognee-building-ai-agent-memory-in-five-lines-of-code--a-friendly-no-hype-field-guide)

- Open WebUI Pipelines filter install workflow (Install from GitHub URL)[langfuse](https://langfuse.com/docs/integrations/openwebui)

## Additional connective tissue and examples

- Open WebUI docs: feature set and OpenAI-compatible endpoint configuration, necessary to point UI at the Pipelines proxy which handles memory enrichment.[openwebui+1](https://docs.openwebui.com/openapi-servers/open-webui/)

- General Zep community posts and docs that show how to operate Zep’s memory services and UI; these are applicable once wired to Open WebUI via a Pipeline.[reddit+2](https://www.reddit.com/r/LangChain/comments/16nufn5/zeps_new_web_ui_easily_manage_your_llm_apps_users/)

- Workato lists “Open WebUI and ZEP integration” as a no-code integration connector, which signals ecosystem-level pairing even if it’s not a memory filter; it can still be useful for workflow orchestration around the same stack.[workato](https://www.workato.com/integrations/open-webui\~zep)

## How to implement quickly (recipe)

- Spin up Open WebUI Pipelines per docs, and set Open WebUI’s model URL to the Pipelines server.[openwebui+1](https://docs.openwebui.com/features/)

- Add a custom filter (similar to the Langfuse filter install flow) that:

   - On each user message:

      - Calls Zep/Mem0/Cognee to search memories for the user/session

      - Prepends a system context block with top-k retrieved memories

   - After receiving the model’s response:

      - Stores the (user, assistant) exchange back into the memory store

This approach mirrors the LangChain+Zep or LangGraph+Mem0 examples, but wrapped as an Open WebUI Pipeline filter.[python.langchain+3](https://python.langchain.com/docs/integrations/providers/zep/)youtube

## Bottom line

- Direct, public “Open WebUI + Zep/Mem0/Cognee” turnkey repos are not widely published yet. The supported and documented path is to use Open WebUI Pipelines as a memory-aware proxy and adapt Zep/Mem0/Cognee sample code from LangChain/LangGraph tutorials into a Pipeline filter. The references above provide all necessary building blocks to do this today.[openwebui+8](https://docs.openwebui.com/)youtube

1. <https://docs.openwebui.com/openapi-servers/open-webui/>

2. <https://docs.openwebui.com/features/>

3. <https://docs.openwebui.com>

4. <https://langfuse.com/docs/integrations/openwebui>

5. <https://github.com/getzep/docs.getzep.com/blob/main/docs/deployment/webui.md>

6. <https://docs.flowiseai.com/integrations/langchain/memory/zep-memory>

7. <https://blog.getzep.com/introducing-zep-memory-ai/>

8. <https://github.com/getzep/zep>

9. <https://python.langchain.com/docs/integrations/providers/zep/>

10. <https://python.langchain.com/docs/integrations/memory/zep_memory/>

11. <https://www.youtube.com/watch?v=e-wBojpJrrQ>

12. <https://github.com/open-webui/open-webui/discussions/4074>

13. <https://github.com/topoteretes/cognee>

14. <https://www.cohorte.co/blog/cognee-building-ai-agent-memory-in-five-lines-of-code--a-friendly-no-hype-field-guide>

15. <https://www.producthunt.com/products/cognee?launch=cognee>

16. <https://www.reddit.com/r/LangChain/comments/16nufn5/zeps_new_web_ui_easily_manage_your_llm_apps_users/>

17. <https://www.workato.com/integrations/open-webui~zep>

18. <https://github.com/zed-industries/zed/discussions/23986>