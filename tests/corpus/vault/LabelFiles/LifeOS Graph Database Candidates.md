# LifeOS Graph Database Candidates

Ingest models:

- vectorizes when data is loaded, for querying later

   - Weaviate

- vectorizes and LLM-ifies during data load

   - Graphiti - not a database

   - Cognee

Storage models:

Structure models:

- Structured

- Unstructured

- Multi-model

Graph models:

- Property graph

- Structured property graph

## Resources

Great article about the ways to define an ontology and how to select a graph database (mentions a lot of the same things I am looking at in this note): 

[GoodData Developers - From RAG to GraphRAG: Knowledge Graphs, Ontologies and Smarter AI by Marcelo G. Almiron](https://medium.com/gooddata-developers/from-rag-to-graphrag-knowledge-graphs-ontologies-and-smarter-ai-01854d9fe7c3)

“Virtual Ontology”: Using Claude Code, a database, a sketched out database schema, and a sketched-out (-ish) ontology description, iterate towards a fleshed out mapping of database structure to ontology: 

[GitHub - virtual-ontology (mcfitzgerald)](https://github.com/mcfitzgerald/virtual-ontology)

[Medium - Whither Ontologies? Palantir-lite with Claude Code by Michael Fitzgerald](https://medium.com/@michael.craig.fitzgerald/whither-ontologies-d871bd3a8098)

> The virtual ontology approach:
>
> 1. **Keep data in place** - Query existing SQL databases directly
>
> 2. **Define business concepts** - Lightweight ontology (TBox/RBox) maps concepts to schema
>
> 3. **Leverage agent capabilities** - Natural language REPLs handle the translation
>
> 4. **Learn from usage** - Capture successful patterns for reuse

## Candidates

### SurrealDB

- Has embedded mode (wasm), only works in js/ts/browser

- Vector store, graph, relational, full-text, time-series

- Query via GraphQL or SurrealQL

### KuzuDB

- Embedded mode, works in Python

- Immature vector support

- Structured Property Graph

- Query via Cypher

### Weaviate

- GraphQL

- Mature vector, immature graph

### Gel

“Relational graph”

- Ontology support

### RDFLib

## My Constraints

### What I Need

- LLM-adjacency, so I can apply classification tasks at ingest time

- A database that can be used by various language runtimes (Python, js/ts, maybe Swift)

### What I Want

- A database that can be embedded in Python app

### What I am neutral on

- Search capabilties

### What I do not want

- A java application

- Something that isnt open source

<https://terminusdb.org/docs/documents-explanation/>

Really good comment on HN about usability

<https://pygraphistry.readthedocs.io/en/latest/10min.html>

<https://pypi.org/project/graphistry/>

\> rapidly wrangle graphs, whether for analysis, data science, ETL, visualization, or AI. Imagine an ETL pipeline or notebook flow or web app where data comes from files, elastic search, databricks, and neo4j, and you need to do more on-the-fly graph stuff with it.

<https://news.ycombinator.com/item?id=45560036>

GFQL allows you to perform graph queries directly on your dataframes

<https://pygraphistry.readthedocs.io/en/latest/gfql/about.html>

The Kuzudb Fork

<https://github.com/Kineviz/bighorn>