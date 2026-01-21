# Reor

 LLM assisted note taking with RAG capabilities

> #### Private & local AI personal knowledge management app for high entropy thinkers.
>
> <https://reorproject.org/>\
> <https://github.com/reorproject/reor>\
> <https://www.reorproject.org/docs>

> **Reor** is an AI-powered desktop note-taking app: it automatically links related notes, answers questions on your notes and provides semantic search. Everything is stored locally and you can edit your notes with an Obsidian-like markdown editor.
>
> The hypothesis of the project is that AI tools for thought should run models locally *by default*. Reor stands on the shoulders of the giants ++[Ollama](https://github.com/ollama/ollama)++, ++[Transformers.js](https://github.com/xenova/transformers.js)++ & ++[LanceDB](https://github.com/lancedb/lancedb)++ to enable both LLMs and embedding models to run locally:
>
> 1. Every note you write is chunked and embedded into an internal vector database.
>
> 2. Related notes are connected automatically via vector similarity.
>
> 3. LLM-powered Q&A does RAG on your corpus of notes.
>
> 4. Everything can be searched semantically.