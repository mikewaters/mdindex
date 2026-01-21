---
tags:
  - document 📑
---
# Entity labeling and recognition in text

> non-generative or *discriminative* tasks, such as classification

> Named Entity Recognition (NER)
>
> NER attempts to find a label for each entity in a sentence, such as a person, location, or organization.

> Token Classification 

### Fine tuning token classification

<https://huggingface.co/docs/transformers/en/tasks/token_classification>

### Using LLMs

#### Example prompt for LLMs:

<https://nsavage.substack.com/p/from-vector-search-to-entity-processing>

#### Using llamaindex and streamlit 

… in a simple app to upload a file, extract terms, and save them in a vector store:

<https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a/terms_definitions_tutorial/>

### GLiNER

GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.

<https://github.com/urchade/GLiNER>

### BERT

> its *encoder-only architecture* makes it ideal for the kinds of real-world problems that come up every day, like retrieval (such as for RAG), classification (such as content moderation), and entity extraction (such as for privacy and regulatory compliance).

source: **[Finally, a Replacement for BERT](https://huggingface.co/blog/modernbert)**

There’s a modern version called Modern BERT from Answer.ai: <https://huggingface.co/blog/modernbert>

<https://news.ycombinator.com/item?id=42463315>

> Apple has supported BERT models in their SDKs for Apple developers for years

Library:

<https://www.labellerr.com/blog/top-7-nlp-libraries-for-nlp-development/>