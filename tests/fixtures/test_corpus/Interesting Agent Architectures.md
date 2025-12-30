# Interesting Agent Architectures

### MassGen

Has an interesting [Agentic Loop Patterns.md](./Agentic%20Loop%20Patterns.md)

> This project started with the "threads of thought" and "iterative refinement" ideas presented in ++[The Myth of Reasoning](https://docs.ag2.ai/latest/docs/blog/2025/04/16/Reasoning/)++, and extends the classic "multi-agent conversation" idea in ++[AG2](https://github.com/ag2ai/ag2)++

[docs.massgen.ai](docs.massgen.ai)

<https://github.com/massgen/MassGen>

> It assigns a task to multiple AI agents who work in parallel, observe each other's progress, and refine their approaches to converge on the best solution to deliver a comprehensive and high-quality result. The power of this "parallel study group" approach is exemplified by advanced systems like xAI's Grok Heavy and Google DeepMind's Gemini Deep Think.

Interestingly, it can generate agent system prompts:

> - Automatic persona generation with multiple strategies (complementary, diverse, specialized, adversarial)

### Honcho (memory library)

Establishes some interesting entities:

- **Peers** - Agents and users, together

   - Multi-participant sessions with mixed human and AI agents

   - Configurable observation settings (which peers observe which others)

- **Theory-of-Mind System**: Multiple implementation methods that extract facts from interactions and build comprehensive models of peer psychology

- **Dialectic API**: Provides theory-of-mind informed responses that integrate long-term facts with current context

- **The Representation** is Honcho's core data structure. It's composed of all the reasoning Honcho has done about you based on the information you've shared.

   - This is kinda what I am doing with Substrate

<https://github.com/plastic-labs/honcho>

<https://blog.plasticlabs.ai/blog/Introducing-Honcho-Chat>

Ships a chat app:

> *Honcho Chat is the interface to your personal memory. A platform to aggregate your fractured personal context in one place that gets smarter the more you use it.*

> 


