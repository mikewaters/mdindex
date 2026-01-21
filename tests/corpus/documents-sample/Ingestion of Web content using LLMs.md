# Ingestion of Web content using LLMs

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

### [Jina.ai](Jina.ai)

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

Qwen and tokenizing a set of files for context ingestion: You might want to use the file markers that the model outputs while being loaded by ollama:  lm_load_print_meta: [general.name](general.name)     = Qwen2.5 7B Instruct 1M     llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'     llm_load_print_meta: EOS token        = 151645 '<|im_end|>'     llm_load_print_meta: EOT token        = 151645 '<|im_end|>'     llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'     llm_load_print_meta: LF token         = 148848 'ÄĬ'     llm_load_print_meta: FIM PRE token    = 151659 '<|fim_prefix|>'     llm_load_print_meta: FIM SUF token    = 151661 '<|fim_suffix|>'     llm_load_print_meta: FIM MID token    = 151660 '<|fim_middle|>'     llm_load_print_meta: FIM PAD token    = 151662 '<|fim_pad|>'     llm_load_print_meta: FIM REP token    = 151663 '<|repo_name|>'     llm_load_print_meta: FIM SEP token    = 151664 '<|file_sep|>'     llm_load_print_meta: EOG token        = 151643 '<|endoftext|>'     llm_load_print_meta: EOG token        = 151645 '<|im_end|>'     llm_load_print_meta: EOG token        = 151662 '<|fim_pad|>'     llm_load_print_meta: EOG token        = 151663 '<|repo_name|>'     llm_load_print_meta: EOG token        = 151664 '<|file_sep|>'     llm_load_print_meta: max token length = 256

<https://news.ycombinator.com/item?id=42832838>