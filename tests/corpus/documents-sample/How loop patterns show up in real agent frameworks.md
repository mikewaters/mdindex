# How loop patterns show up in real agent frameworks

2025-11-18 — ChatGPT (GPT-5.1 Thinking)Model: GPT-5.1

I’ll use some of the big open-source frameworks you’re likely to touch: **LangChain/LangGraph, AutoGen, Semantic Kernel, CrewAI, AutoGPT-style task runners, LlamaIndex agents, OpenAI Agents SDK / “Swarm”-style**, etc. [ampcome.com](ampcome.com)[\+4Turing+4Analytics Vidhya+4](https://www.turing.com/resources/ai-agent-frameworks?utm_source=chatgpt.com)

### A. ReAct-like loops

Core idea (from the original ReAct paper): maintain a trajectory of\
`Thought → Action → Observation → Thought → ...` [arXiv+3arXiv+3Prompting Guide+3](https://arxiv.org/abs/2210.03629?utm_source=chatgpt.com)

**Where you see this:**

- **LangChain Agents / LangGraph**

   - `AgentExecutor` with tools is basically ReAct: the LLM emits either a “final answer” or a tool call; tools return observations; loop continues.

   - LangGraph just makes the loop explicit as a graph with a “should I call tools again?” node.

   - **LangChain Agents** (`ReActAgent`, tool-using agents).

- **LlamaIndex Agents** (query engines with tool calls through reasoning).

- **OpenAI function/tool calling agents (Agents SDK, Swarm, etc.)**

   - Model emits `tool_calls` as actions; the runtime executes them and feeds back tool results; the model updates its “thoughts” in subsequent turns.

- **AutoGen (Microsoft)**

   - A “worker” agent often produces thoughts and decides when to call tools / ask other agents; the host loop is still ReAct-ish.

- **CrewAI / multi-agent orchestrators**

   - Each agent tends to use a ReAct-like inner loop (tools + reasoning), even if the outer loop is multi-agent.

You can think of “standard tool-calling assistant” as a **slightly more structured ReAct loop**.

---

### B. Plan-and-Execute (PaE)

From research & blog patterns like “ReWOO”, PaE separates a **planner** from an **executor**; the planner emits a task list; executor steps through it. [Turing+2Analytics Vidhya+2](https://www.turing.com/resources/ai-agent-frameworks?utm_source=chatgpt.com)

**Where you see this:**

- **LangChain**

   - `PlanAndExecute` and similar “planner + tools” agents.

- **LangGraph**

   - Common pattern: Planner node → tasks persisted to graph state → Worker node iterates tasks (often with tools).

- **AutoGPT / BabyAGI-style loops**

   - Those “task list managers” are almost canonical Plan-and-Execute: maintain a backlog, pop one, run, create new tasks.

- **Semantic Kernel**

   - Uses “planners” that generate a function-call sequence (skills) and then execute them separately.

---

### C. Reflex / “toolformer-style” loops

Minimalistic: each step, the model just answers or calls a tool, with no explicit multi-step plan. Often used in production because it’s simple and easy to bound.

**Where you see this:**

- **Vanilla OpenAI / Anthropic tool-calling assistants**

   - Prompt: “Call tools if needed; otherwise answer.” The runtime loop is: LLM → maybe tool → LLM → done.

- **Framework defaults**

   - Many frameworks’ “simple agent” or “chat with tools” examples are this reflex loop (no long-term planning or memory).

- **Toolformer** (Meta’s paper: training LMs to use tools in a single forward pass).

- Many **“single-call with tools”** production assistants (e.g., typical chat-with-tools pattern in LangChain, OpenAI function calling, Vercel AI SDK, etc.).

- **Haystack Agents** in simple tool-calling configurations.

---

### D. Chain-of-Thought, Self-Ask, Least-to-Most

These are mostly **prompt-level reasoning strategies**, but frameworks often encode them in reusable components. [arXiv+](https://arxiv.org/abs/2305.10601?utm_source=chatgpt.com)[2ibm.com](2ibm.com)[\+2](https://arxiv.org/abs/2305.10601?utm_source=chatgpt.com)

**Where you see this:**

- **CoT everywhere** – simply enabling CoT in prompts of LangChain, Semantic Kernel, LlamaIndex, etc.

- **Self-Ask / question decomposition**

   - Some frameworks provide a “question decomposition” chain or node that breaks a task into sub-questions, but still runs inside another loop (e.g., LangGraph nodes that do “generate sub-queries”).

- **Least-to-Most**

   - Implemented as “solve easier subproblems sequentially” flows in LangGraph or AutoGen multi-step scripts.

- Many **RAG frameworks** that use “question decomposition” (like LlamaIndex query decomposition, LangChain’s “map-reduce” QA and multi-step question decomposition).

---

### E. Self-Reflective loops (Reflexion, self-critique, etc.)

Reflexion adds **episodic memory + self-critique** to an agent: after each episode, the agent generates a reflection and stores it, then uses that in future reasoning. [ACM Digital Library+4arXiv+4OpenReview+4](https://arxiv.org/abs/2303.11366?utm_source=chatgpt.com)

**Where you see this:**

- **Reflexion replicated in frameworks**

   - Repos show sample implementations with AutoGen and LangChain. [GitHub+1](https://github.com/noahshinn/reflexion?utm_source=chatgpt.com)

- **Self-critique patterns**

   - Many frameworks offer “router → generator → reviewer → finalizer” pipelines, where a second LLM reviews and suggests corrections.

   - **LangChain** (e.g., a chain where an LLM critiques another’s output).

   - **LlamaIndex** evaluators and self-reflection flows.

- **Constitutional AI style**

   - Some multi-message “critic and revise” loops inspired by Anthropic’s Constitutional AI show up as templates.

---

### F. Multi-agent loops (debate, committee, roles)

Here the loop is across agents, not just time: multiple agents exchange messages, sometimes with a supervising “judge”.

**Where you see this:**

- **AutoGen**

   - Designed around multi-agent chat loops: user proxy ↔ assistant ↔ tools, or more complex teams.

- **ChatDev**, **MetaGPT**, other “software company in a box” multi-agent systems.

- **CrewAI, Swarm-like frameworks**

   - You define roles (researcher, coder, reviewer). The runtime cycles between them until a goal condition is met.

- **LangGraph**

   - Graph nodes that are separate agents, each called in a loop with explicit edges and termination conditions.

---

### G. Hierarchical controller/worker

A special case of multi-agent: **manager** decomposes tasks, **workers** execute, **manager** aggregates.

**Where you see this:**

- **CrewAI**

   - Explicit concepts of “manager agent” and “specialist agents.”

- **AutoGen**

   - Manager agent that coordinates multiple assistants and tools.

- **LangGraph / LlamaIndex agents**

   - Graph/task nodes acting as manager; leaf nodes as workers (e.g., RAG worker, code worker).

- **BabyAGI** (modern forms) with task manager + executor + prioritizer.

- **SWE-agent-like** setups (manager LLM orchestrating tools + sub-routines).

- **MetaGPT** again (CEO → CTO → Developer roles)

---

### H. Retrieval-augmented loops (RAG-centric, ReWOO, etc.)

The loop or graph is centered on retrieval → reasoning → maybe more retrieval, not just one retrieval call. [arXiv+2Turing+2](https://arxiv.org/abs/2305.10601?utm_source=chatgpt.com)

**Where you see this:**

- **LlamaIndex, LangChain RAG agents**

   - Query node → retriever → re-ranker → answer node; may iterate if answer confidence is low.

- **LangGraph**

   - RAG graphs where a “planner” chooses which retriever/tool to use; the loop continues until enough context is gathered.

---

### I. Verification & correction loops

Verifier/critic loops: a separate model (or the same model in another role) checks and corrects.

**Where you see this:**

- **AutoGen**

   - Common pattern: Coder agent + Critic (or Tester) agent. Tester runs code or checks correctness.

- **LangGraph / LangChain**

   - “Retry” nodes: verifier node checks constraints; if failed, loop back to generator node with feedback.

- **Semantic Kernel**

   - Pipelines that include step-level validation skills.

---

### J. Simulation-based loops (Tree-of-Thoughts, Graph-of-Thoughts)

Tree-of-Thoughts (ToT) explicitly explores multiple reasoning paths and prunes/branches them. [NeurIPS Proceedings+3arXiv+3GitHub+3](https://arxiv.org/abs/2305.10601?utm_source=chatgpt.com)

**Where you see this:**

- **Research / experimental code** (e.g., Tree-of-Thoughts repo, or custom LangGraph graphs).

- **Search-based reasoning in LangGraph**

   - People implement ToT- or beam-search-like flows: each “state” is a node in graph state, edges represent expansions, a controller picks which node to expand next.