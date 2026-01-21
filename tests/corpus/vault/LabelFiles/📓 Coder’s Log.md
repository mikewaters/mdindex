---
tags:
  - document 📑
---
# 📓 Coder’s Log

- [ ] Try these LLM frameworks out in **notebooks**; there are plenty of examples

Dec 1, 2024 I am not going to use `ell`; there are other more mature frameworks out there,

Nov 24, 2024 I do not need to have this universal connector; I can instead stream data to one place using native system tools - at least to start with.

Example: Rather than having an agent read my Gmail account, I can forward to mail to some collector system which writes data that the agent reads.

Oct 14, 2024

The ell project looks like a great abstraction over prompts: <https://github.com/MadcowD/ell>

> `ell` is a lightweight, functional prompt engineering framework

Oct 12, 2024

The langroid project seems exactly what I need for inference, and even ships a FastAPI server: <https://github.com/langroid/fastapi-server>

Some examples of langroid agents:

- <https://github.com/ashim-mahara/rag-mama-rag/blob/main/langroid_agent_mavis.py>

   - cybersec assessor

- <https://github.com/jihyechoi77/malade/blob/main/malade/critic_agent.py>

   - medical interaction finder with reasoning critic agent

- 
