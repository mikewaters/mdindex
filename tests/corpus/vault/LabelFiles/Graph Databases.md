# Graph Databases

Purpose: [LifeOS Graph Database Candidates.md](./LifeOS%20Graph%20Database%20Candidates.md)

### Weaviate

Has embedded mode, but it looks kinda hacky. 

Looks really cool though, fits what I need for the future

Has a great article about how its perfect for representing [Defining the LifeOS Ontology - Notes.md](./Defining%20the%20LifeOS%20Ontology%20-%20Notes.md)

[Weaviate - Taxonomies, Ontologies, and Schemas: How Do They Relate to Weaviate?](https://medium.com/semi-technologies/taxonomies-ontologies-and-schemas-how-do-they-relate-to-weaviate-9f76739fc695)

Query via GraphQL

### SurrealDB

Multi-model

Has an embedded mode, but it only works in WASM and so it is unavailable to Python

Query via GraphQL or SurrealQL

[SurrealDB - The ultimate multi-model database](https://surrealdb.com/)

>  unified, multi-model database purpose-built for AI systems. It combines structured and unstructured data (including vector search, graph traversal, relational queries, full-text search, document storage, and time-series data)

### TigerGraph

> As the world’s first and only Native Parallel Graph (NPG) system, TigerGraph is a complete, distributed, graph analytics platform supporting web-scale data analytics in real time

[TigerGraph - Documentation: Explore](https://docs.tigergraph.com/home/)

[pyTigerGraph - pyTigerGraph](https://docs.tigergraph.com/pytigergraph/1.8/intro/)

[TigerGraph - Graph Data Science Library](https://docs.tigergraph.com/graph-ml/3.10/intro/)

All *sorts* of data science algorithms, like classification, similarity

This is like Postgres level serious for graphs

### Kuzudb

Seems simplistic but has an embedded python version

Support BM25 embedding models for search, not mature in the vector area

Structured property graph model

### Postgres

Can [abuse the pgrouter extension](https://supabase.com/blog/pgrouting-postgres-graph-database)

Apache has an extension AGE: 

[Medium - PostgreSQL Goes Multi-Model: Graph, Vector, and SQL by Sixing Huang](https://dgg32.medium.com/postgresql-goes-multi-model-graph-vector-and-sql-5f27dbc04835)

It allows you to structure as a graph and make Cypher queries

### FalcorDB

[FalkorDB - Home Docs](https://docs.falkordb.com/)

GraphRAG

Property Graph Database model

> What is `FalkorDB`?

> - FalkorDB is an `open-source database management system` that specializes in graph database technology.
>
> - FalkorDB allows you to represent and store data in nodes and edges, making it ideal for handling connected data and relationships.
>
> - FalkorDB Supports OpenCypher query language with proprietary extensions, making it easy to interact with and query your graph data.
>
> - With FalkorDB, you can achieve high-performance `graph traversals and queries`, suitable for production-level systems.

### Memgraph

Streaming version of neo4j, in c++

[Memgraph - Open-source graph database, tuned for dynamic analytics environments. Easy to adopt, scale and own.](https://github.com/memgraph/memgraph)

### [Flur.ee](Flur.ee)

Very new

> Fluree is a graph database that:
>
> - Supports semantic web standards
>
> - Has deep cryptographic integration, providing a fine-grained security model
>
> - Supports seamless horizontal scaling
>
> - Provides a pluggable storage interface, letting you pick the storage layer most appropriate for your application
>
> …  a *JSON-LD (JSON for Linked-Data) database* which is a version of an *RDF (Resource Description Framework) graph database*

<https://next.developers.flur.ee/docs/learn/foundations/architecture-overview/>

JSON-LD database built on a ledger, and so comes at this from a security control and collaboration-on-data perspective

### ArangoDB

Licensing makes me feel icky

### Networkx

Not a database, but can be used for analysis of graph data

### Neo4j

[Neo4j - LangChain](https://python.langchain.com/docs/integrations/providers/neo4j/)

I am biased

### NebulaGraph

<https://github.com/vesoft-inc/nebula>

open source

cloud hosting

### DGraph

<https://github.com/hypermodeinc/dgraph>

<https://docs.hypermode.com/dgraph/self-managed/overview>

open source

cloud hosting

### OntoText GraphDB

<https://www.ontotext.com/products/graphdb/>

Features

- RDF database

- Ontology features

- Not open source, has open source components ([github org](https://github.com/Ontotext-AD))

- Has a [free license](https://graphdb.ontotext.com/documentation/10.8/set-up-your-license.html)

- Supports docker

- Java ew

### Gel

Query: EdgeQL

Built on Postgres, used to be called EdgeDB

Define a schema for the ontology: <https://docs.geldata.com/reference/datamodel>

Example of gel being used in an app: 

[Gel Blog - The missing piece in the FastAPI + Pydantic AI agentic stack](https://www.geldata.com/blog/the-missing-piece-in-the-fastapi-pydantic-ai-agentic-stack)

Docs:

<https://docs.geldata.com/reference/using/python/api/codegen>

[Gel - Gel supercharges Postgres with a modern data model, graph queries, Auth & AI solutions, and much more.](https://github.com/geldata/gel?tab=readme-ov-file)

> Gel is a next-generation [graph-relational database](https://www.geldata.com/blog/the-graph-relational-database-defined) designed as a spiritual successor to the relational database.

COOL

> Gel is not a graph database: the data is stored and queried using relational database techniques. Unlike most graph databases, Gel maintains a strict schema.
>
> Gel is not a document database, but inserting and querying hierarchical document-like data is trivial.
>
> Gel is not a traditional object database, despite the classification, it is not an implementation of OOP persistence.

## RDF Stores

### LinkML

<https://linkml.io/linkml/intro/overview.html>

> LinkML is a flexible modeling language that allows you to author [schemas](https://w3id.org/linkml/SchemaDefinition) (“models”) in YAML that describe the structure of your data. 
>
> 

Integrated with a lot of cool ontology things, see Ontology docs.

### Ontotext GraphDB

[Ontotext GraphDB](https://graphdb.ontotext.com/) is a graph database and knowledge discovery tool compliant with RDF and SPARQL.

<https://github.com/Ontotext-AD/graphdb-docker>

Licensing rubs me the wrong way, like ArangoDB

Java, ew



## Python Libraries

### DocArray

<https://docs.docarray.org/#coming-from-pydantic>

## [RAG Frameworks.md](./RAG%20Frameworks.md)

## Episodic Memory

### Graphiti

Not a database, but has examples using Pydantic model ingestion (!). Sits on top of another graph DB like FalkorDB, Neo4j, KuzuDB

Is a “[Graph RAG.md](./Graph%20RAG.md) ++” application technically

Does LLM inference at ingest time

Did some shady shit with Zep Memory oen source, they close-sourced it. Its an [Agent-based Frameworks!Libraries.md](./Agent-based%20Frameworks!Libraries.md) like Letta

Fits what I might need in the future

### Mem0

<https://github.com/mem0ai/mem0>

has graph memory via the usuals - Neo4j, KuzuDB etc

### Cognee

<https://docs.cognee.ai/getting-started/introduction>

Sits on top of KuzuDB, as well as LanceDB and SQLite (it has a vector, triple/graph, and relational store)

> Cognee provides four main operations that users interact with:
>
> - **[Add](https://docs.cognee.ai/core-concepts/main-operations/add)** — Ingest and prepare data for processing, handling various file formats and data sources
>
> - **[Cognify](https://docs.cognee.ai/core-concepts/main-operations/cognify)** — Create knowledge graphs from processed data through cognitive processing and entity extraction
>
> - **[Memify](https://docs.cognee.ai/core-concepts/main-operations/memify)** — Optional semantic enrichment of the graph for enhanced understanding *(coming soon)*
>
> - **[Search](https://docs.cognee.ai/core-concepts/main-operations/search)** — Query and retrieve information using semantic similarity, graph traversal, or hybrid approaches

You can use Cognee to convert a relational database to.a graph structure - magic?  <https://blog.kuzudb.com/post/cognee-kuzu-relational-data-to-knowledge-graph/>

<https://gitlab-org.gitlab.io/rust/knowledge-graph/getting-started/usage/>