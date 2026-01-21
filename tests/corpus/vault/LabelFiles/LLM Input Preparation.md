---
tags:
  - document 📑
Document Type:
  - Technical
Topic:
  - ai
  - development
Subject:
  - Concept
---
# LLM Input Preparation

# Codebases and docs

## Library Documentation Compression

For efficiently “teaching” ([In-Context Learning.md](./In-Context%20Learning.md)) an LLM about some libraries that you are using in your project, to ensure it creates the right code and answers your questions accurately (without hallucinations).

Related: [LLM Coding Assistants.md](./LLM%20Coding%20Assistants.md)

### llm-docs

[Dicklesworthstone/llm-docs](https://github.com/Dicklesworthstone/llm-docs)

> LLM-Docs provides condensed, optimized documentation specifically tailored for efficient consumption by Large Language Models (LLMs).

> The documentation optimization process involves:
>
> 1. **Collection**: Gathering comprehensive documentation for popular open-source libraries.
>
> 2. **Distillation**: Employing advanced LLMs, such as Claude 3.7, to meticulously analyze, interpret, and rewrite the documentation into a more compact, clear, and LLM-friendly format.
>
> 3. **Organization**: Structuring the documentation consistently and logically across various programming languages and libraries.

## Codebase Understanding apps

### codeqai

> Generate datasets from code for finetuning, search your codebase semantically or chat with your code from cli. Keep the vector database superfast up to date to the latest code changes. 100% local support without any dataleaks.

> <https://github.com/fynnfluegge/codeqai>

Python, Ollama, LangChain, tree sitter, Streamlit, vector db, CodeLlama

### Cody

Uses RAG on your codebase, via [SCIP](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/5b5d3c91-b479-40e7-873b-c6a064b74983#39fd33c4-5f9b-464f-ab37-fae768987d55), ranks with modified [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) bag of words. 

- [How Cody understands your codebase | Sourcegraph Blog](https://sourcegraph.com/blog/how-cody-understands-your-codebase?ref=blog.lancedb.com)

Requires Sourcegraph account, cannot run locally since v5.7 :(

### Aider

Aider has a repository map component.

\[Documentation\](<https://aider.chat/docs/repomap.html>)

\[Building a better repository map with tree sitter\](<https://aider.chat/2023/10/22/repomap.html>)

## Codebase understanding Prompts

Related: My [Prompt Library.md](./Prompt%20Library.md)

<https://github.com/Pythagora-io/gpt-pilot/tree/main/core/prompts/code-monkey>

## Source code indexing

### SCIP Code Intelligence Protocol

open source by Sourcegraph

<https://github.com/sourcegraph/scip>

### Tools using SCIP

#### code-charter

<https://github.com/CRJFisher/code-charter>

> transforming complex code into diagrams that distill key user-centric patterns

VSCode extension, supports python, requires Ollama

## Codebase embedding

Using embeddings as a strategy becomes more difficult as source code volume increases, vector database size scales with source code size. Sourcegraph themselves switched from embeddings to ICL, but they are at a huge scale.

Github topics:

- <https://github.com/topics/code-understanding>

- <https://github.com/topics/code-intelligence>

- <https://github.com/topics/code-generation> (big)

### Unoplat Code confluence

Open source code intelligence, AKA exactly-what-i-need, github repo analysis

Python, requires Github PAT

<https://github.com/unoplat/unoplat-code-confluence>

<https://docs.unoplat.io>

<https://docs.unoplat.io/docs/quickstart/how-to-run>

### Source code-specific models

#### CodeLlama (llm)

<https://huggingface.co/codellama>

<https://ai.meta.com/blog/code-llama-large-language-model-coding/>

From Meta

Fine-tuning CodeLLama: <https://github.com/okuvshynov/slowllama>

#### Salesforce T5 (llm, embedding)

ex: `codet5p-110m-embedding`

<https://huggingface.co/Salesforce/codet5p-110m-embedding>

Pre-trained on github open source, strict permissive, published in 2023

Paper: CodeT5: Open Code LLMs for Code Understanding and Generation

[arxiv.org/abs/2305.07922](arxiv.org/abs/2305.07922)

#### **CodeBERTa (embedding)**

#### **GraphCodeBERT (embedding)**

#### Voyage AI (embedding, llm)

Hosted/API product

<https://docs.voyageai.com/docs/embeddings>

`voyage-code-3`

> Optimized for **code** retrieval. 

> <https://blog.voyageai.com/2024/12/04/voyage-code-3/>

#### OpenAI

Could just use theirs, suggested by some

### Papers

#### GALLa

> Graph Aligned Large Language Models for Improved Source Code Understanding

<https://github.com/codefuse-ai/GALLa>

From CodeFuse AI ([huggingface](https://huggingface.co/codefuse-ai))

### Resources

- OpenAI Cookbook for code indexing: [jupyter notebook](https://github.com/openai/openai-cookbook/blob/main/examples/Code_search_using_embeddings.ipynb?ref=blog.lancedb.com)

#### Guides

- <https://docs.continue.dev/customize/tutorials/custom-code-rag> Instructions on how to set up RAG for a codebase. Suggests `voyage-code-3` for embedding, LanceDB for the vector DB, and a recursive AST-based chunker.

- Instruction series by LanceDB

   - <https://blog.lancedb.com/rag-codebase-1/>

   - <https://blog.lancedb.com/building-rag-on-codebases-part-2/>

   - Repo: [https://github.com/sankalp1999/code_qa](https://github.com/sankalp1999/code_qa?ref=blog.lancedb.com)

   - Much lower level of detail than Continue.dev

   - Uses recursive AST-based chunking via tree-sitter and LanceDB

- [Vector search on codebase using LLAMA2/ Gemini pro and Chromadb](https://medium.com/@raiharsh88/vector-search-on-codebase-using-llama2-and-chromadb-cc6c0ab8bc63)

   - Uses a code embedding model checkpoint (Salesforce/codet5p-110m) to tokenize and embed a code snippet and store it in ChromaDB, in like ten LoC, using hf’s `transformers.AutoModel` and `AutoTokenizer`

   - Ranks the results using Gemini Pro API

#### Code samples

- RAGBase4Code: <https://github.com/zzhou292/RAGBase4Code/tree/main>

   - *A advanced RAG framework, by Json Zhou, designed to enhance AI applications with multi-source knowledge retrieval. Specifically designed for offline LLM4CODE, aiming to provide codebase understanding offline and securely.*

   - Python. Uses `microsoft/codebert-base` for embedding, Ollama and Langchain

- Composeio’s `CodeMap` implementation: <https://github.com/ComposioHQ/composio/blob/8a7bac58810e5be3e661b24c830c53cf5ea2f2d8/python/composio/tools/local/codeanalysis/actions/create_codemap.py>

### How to approach embedding source code

[source](https://discuss.huggingface.co/t/codebase-embedding/137026/2?u=threecheese) (hf forums)

1. **Code Splitting:** Don’t embed entire files. Split your code into smaller chunks (e.g., functions, classes, or even smaller logical blocks). This improves retrieval accuracy.

2. **Embedding Models:** For code, consider these options:

- **Sentence Transformers (e.g., `all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1`):**Good general-purpose embeddings, often a solid starting point.

- **Code-Specific Models (e.g., CodeBERTa, GraphCodeBERT):** These are trained on code and often perform better for code-related tasks. Sentence Transformers also offers code specific models.

- **OpenAI Embeddings (if budget allows):** Very high quality but come with usage costs.

1. **Directory Structure:** You don’t directly “feed” the directory structure to the embedding model. Instead:

- Include file paths/names in the metadata of each code chunk. This helps with context.

- Consider creating a “summary” embedding for each file or directory. This can be used for a first-pass filtering before retrieving more granular chunks.

1. **Vector Database:** Use a vector database (e.g., FAISS, Chroma, Weaviate, Pinecone) to store and efficiently query your embeddings.

2. **Chunking Strategy:** Consider splitting code based on Abstract Syntax Trees (ASTs). Libraries like `ast` in Python can help with this. This ensures you’re embedding logical code units.

### LangChain

LangChain Source Code document loader: <https://python.langchain.com/docs/integrations/document_loaders/source_code/>

Uses tree-sitter to parse source code in various languages, for splitting or embedding

## Repository mappers

Generates a prompt/summary of some codebase to be used for [In-Context Learning.md](./In-Context%20Learning.md).

**Research sources:**

- <https://news.ycombinator.com/item?id=41482793>

- <https://www.reddit.com/r/Rag/comments/1gtacn6/rag_for_codebases/>

### ummon

> Unlock code insights with knowledge graphs: Connect code to concepts, query with ease, empower AI assistance

> Ummon is a code analysis tool that builds knowledge graphs from codebases to enhance understanding, improve AI assistance, and enable sophisticated querying. It creates connections between code entities (functions, classes, modules) and domain concepts, making it easier to reason about complex software systems.

> <https://github.com/Nayshins/ummon>

rust

### smoosh (python lib)

> Snapshot an entire repo or directory as plaintext on the clipboard and paste to your favorite AI tool!

> <https://github.com/J-McNamara/smoosh>

### files-to-prompt

Python

Using: [files-to-prompt](https://github.com/simonw/files-to-prompt) (from the GOAT simonw)

> Concatenate a directory full of files into a single prompt for use with LLMs

<https://github.com/simonw/files-to-prompt>

### ingest

> Parse files (e.g. code repos) and websites to clipboard or a file for ingestions by AI / LLMs\
> <https://github.com/sammcj/ingest>

> Ingest includes a feature to estimate VRAM requirements and check model compatibility

golang

### Repomix

> Repomix (formerly Repopack) is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) or other AI tools like Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok, and more.\
> <https://github.com/yamadashy/repomix>\
> <https://repomix.com>

Typescript cli, web version, has MCP server, various output options

> Repomix generates a single file with clear separators between different parts of your codebase.\
> To enhance AI comprehension, the output file begins with an AI-oriented explanation, making it easier for AI models to understand the context and structure of the packed repository.

> If you're using Python, you might want to check out `Gitingest`, which is better suited for Python ecosystem and data science workflows: ++[https://github.com/cyclotruc/gitingest](https://github.com/cyclotruc/gitingest)++

### code2prompt

> A CLI tool to convert your codebase into a single LLM prompt with source tree, prompt templating, and token counting.
>
> You can run this tool on the entire directory and it would generate a well-formatted Markdown prompt detailing the source tree structure, and all the code.
>
> <https://github.com/mufeedvh/code2prompt>

5k stars, updated as of Mar 13, 2025

rust cli, python sdk

### git-ingest

> Turn any Git repository into a prompt-friendly text ingest for LLMs.

<https://github.com/cyclotruc/gitingest>

Has browser plugins

Python library

### repo2file

Python script, shmeh

> Dump selected files from your repo into single file to easily use in LLMs (Claude, Openai, etc..)

> <https://github.com/artkulak/repo2file>

### repogather

Python library, uses OpenAI for ranking

<https://github.com/gr-b/repogather>

> Easily copy all relevant source files in a repository to clipboard. For use in LLM code understanding and generation workflows

> repogather is a command-line tool that copies all relevant files (with their relative paths) in a repository to the clipboard. It is intended to be used in LLM code understanding or code generation workflows. It uses gpt-4o-mini (configurable) to decide file relevance, but can also be used without an LLM to return all files, with non-AI filters (such as excluding tests or config files).

### onefilellm

> Specify a github or local repo, github pull request, arXiv or Sci-Hub paper, Youtube transcript or documentation URL on the web and scrape into a text file and clipboard for easier LLM ingestion

<https://github.com/jimmc414/onefilellm>

Python, requires github PAT

Supports the following input options:

- Local file path (e.g., C:\\documents\\report.pdf)

- Local directory path (e.g., C:\\projects\\research) -> (files of selected filetypes segmented into one flat text file)

- GitHub repository URL (e.g., ++<https://github.com/jimmc414/onefilellm>++) -> (Repo files of selected filetypes segmented into one flat text file)

- GitHub pull request URL (e.g., ++[dear-github/dear-github#102](https://github.com/dear-github/dear-github/pull/102)++) -> (Pull request diff detail and comments and entire repository content concatenated into one flat text file)

- GitHub issue URL (e.g., ++[isaacs/github#1191](https://github.com/isaacs/github/issues/1191)++) -> (Issue details, comments, and entire repository content concatenated into one flat text file)

- ArXiv paper URL (e.g., ++<https://arxiv.org/abs/2401.14295>++) -> (Full paper PDF to text file)

- YouTube video URL (e.g., ++<https://www.youtube.com/watch?v=KZ_NlnmPQYk>++) -> (Video transcript to text file)

- Webpage URL (e.g., ++<https://llm.datasette.io/en/stable/>++) -> (To scrape pages to x depth in segmented text file)

- Sci-Hub Paper DOI (Digital Object Identifier of Sci-Hub hosted paper) (e.g., 10.1053/j.ajkd.2017.08.002) -> (Full Sci-Hub paper PDF to text file)

- Sci-Hub Paper PMID (PubMed Identifier of Sci-Hub hosted paper) (e.g., 29203127) -> (Full Sci-Hub paper PDF to text file)

### gh repo download

> download and view the contents of a GitHub repository or a ZIP file as a single text file

> <https://github.com/dmwyatt/gh_repo_download>

Does what it says on the tin

python and javascript

# Web content

[onefilellm](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/5b5d3c91-b479-40e7-873b-c6a064b74983#94c09d80-b973-4274-9f94-33b51cb73d18) ingests URLs, YT transcripts etc

## Libraries

### Extractus

<https://github.com/extractus>

> ++[feed-extractor](https://github.com/extractus/feed-extractor)++: extract & normalize RSS/ATOM/JSON feed
>
> ++[article-extractor](https://github.com/extractus/article-extractor)++: extract main article from given URL
>
> ++[oembed-extractor](https://github.com/extractus/oembed-extractor)++: extract oEmbed data from supported providers

### others

-  <https://github.com/romansky/dom-to-semantic-markdown>

   - | preserves the semantic structure of web content, extracts essential metadata, and reduces token usage compared to raw HTML, making it easier for LLMs to understand and process information

### Crawlers

## Hosted

### Jina.ai

<https://jina.ai/reader/>

> Convert a URL to LLM-friendly input, by simply adding `[r.jina.ai](r.jina.ai)` in front.

Public API (1 million tokens free), or self-hostable

<https://github.com/jina-ai/reader>

### APIfy

 <https://apify.com/>

self-host: [Crawlee](https://github.com/apify/crawlee)—A web scraping and browser automation library for Node.js to build reliable crawlers

### Firecrawl

<https://www.firecrawl.dev/>

self-host: <https://github.com/mendableai/firecrawl>

> Turn entire websites into LLM-ready markdown or structured data. Scrape, crawl and extract with a single API.

### Browserbase

> A web browser for your AI

<https://docs.browserbase.com/features/session-live-view>

Qwen and tokenizing a set of files for context ingestion: You might want to use the file markers that the model outputs while being loaded by ollama:

lm_load_print_meta: general.name     = Qwen2.5 7B Instruct 1M
    llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
    llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
    llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
    llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
    llm_load_print_meta: LF token         = 148848 'ÄĬ'
    llm_load_print_meta: FIM PRE token    = 151659 '<|fim_prefix|>'
    llm_load_print_meta: FIM SUF token    = 151661 '<|fim_suffix|>'
    llm_load_print_meta: FIM MID token    = 151660 '<|fim_middle|>'
    llm_load_print_meta: FIM PAD token    = 151662 '<|fim_pad|>'
    llm_load_print_meta: FIM REP token    = 151663 '<|repo_name|>'
    llm_load_print_meta: FIM SEP token    = 151664 '<|file_sep|>'
    llm_load_print_meta: EOG token        = 151643 '<|endoftext|>'
    llm_load_print_meta: EOG token        = 151645 '<|im_end|>'
    llm_load_print_meta: EOG token        = 151662 '<|fim_pad|>'
    llm_load_print_meta: EOG token        = 151663 '<|repo_name|>'
    llm_load_print_meta: EOG token        = 151664 '<|file_sep|>'
    llm_load_print_meta: max token length = 256

<https://news.ycombinator.com/item?id=42832838>

# inbox

# Codebase packing tools for prompts

<https://github.com/mufeedvh/code2prompt>

<https://github.com/bodo-run/yek>

<https://news.ycombinator.com/item?id=42753302>



[shaneholloman / codemapper](https://github.com/shaneholloman/codemapper)

Perfect for AI Prompts. CodeMapper is a python script that creates a comprehensive Markdown document representing the structure and contents of any given code base

### [docling-project / docling](https://github.com/docling-project/docling)

Get your documents ready for gen AI