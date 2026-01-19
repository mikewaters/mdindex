## Migrate from substrate
- Add a Resources table, even URLs and stuff.  It will contain the URI, and generate a Document which could be the actual document content, or in the case of a url the cached web page, or the highlights etc. The Resource will be cached, not the document, so we can iterate on things like “collect resource type X”.
An obsidian resource would have the obsidian:// uri etc 
- I can have scripts for raindrop, heptabase, obsidian etc, rather than build a comprehensive CLI.
- obsidian et al can be in a Substrate integrations library, or sub module of pmd, “from pmd.integrations.obsidian import VaultCollector;”. Decision lies in if integrations like obsidian are needed outside of indexing.
- rename pmd to catalog
- rename Source Collection to Dataset

## Tasks
- review the other substrates components
- migrate external config to Pydantic so cli, api, and Python clients can use the same validated structures
- remove _index_via_legacy and anything like it
- move LLMs to dspy
- use a better chunker wtpsplit sat-61-sm
- move all my codebases into one repository 
- add a resources “host” for filtering by local stuff
- skip indexing LLM slop: pipeline get it, what is it, what should I do with it
- have a separate index for community detection, where the index is not ontology-aligned; instead it goes ham on entity extraction etc. Maybe this is a separate graph, using standard graphrag library
- -is_llm_availabel should rais a runtimeerror if false
- fix "re-deifned exports" across the codebase.  Example: workflows.contracts
- remove the metadata module, its just a workaround atm
———
 
## Decisions
- need a Resource type, which will generate one or more documents
- Resources will be cached in their original form with the addition of metadata if it can’t be an intrinsic part of the cached thing (like it’s created and modified dates)
- a Document is a thing to be indexed; a Resource is the representation of a thing in the world.
- Substrate is an app that uses various libraries that handle domains like data catalog hosting and ontology definition/storage
- Substrate should support local first for all features but optionally use centrally hosted data, hosted indexing/classification, use external capabilities like OpenAi, or be hosted and externally accessible by api itself.
- Substrate should be usable as a tool by a coding agent

———

## Roadmap
### Local, then centralized
- Collection happens locally, and documents are copied to the local object store
- Indexing is done against the cache
- Databases are local
Next step would be to centralize the object store and the database. 

### Multiple contexts
Single context per repo for now, next would be multiple database support and per-context initialization.

### Data stores
Need to be able to iterate on data stores, without rewriting codebase. And so this will need to be configuration-driven.

### Pipelines
Need to introduce a more robust pipeline which can be assembled in code, graphed/visualized, and monitored during runtime.

### Experiments
Want to experiment with different RAG frameworks, databases (graph and relational) and vector stores.
1. Duckdb-vss over sqlite-vec
2. SurrealDB
3. DAGSter or Llamaindex workflows

———

## Substrate Architecture
### Terminology
- Resource: refers to a document, url, or text chunk. Each resource has an unique uri 
### Classes
- Entity
- Concept
- Fact (Identity, role, constraints)
- Preference
- Concern ("Context": Activity, Effort, Goal, Problem etc)
- Memory
- Decisions
Some of these can have temporal dimensions, like creation and lifetime
### Interfaces
#### Scripts
Obsidian importer accepts a vault path, uses the obsidian source type and configures my ontology. 

### Domains
#### The Ontology
The Ontology associates content with relevance and meaning.

Responsible for:
- maintaining entity and topic data stores; should this include claim, fact, decision etc?
- providing classification primitives
- support ontology improvement
- can be configured for each context (life, work, interest etc)
- 
#### Data Catalog
The Data Catalog records all resources collected by the end user. It assists with collection and surfacing of content, aligned with the users Ontology. 

Responsible for:
- providing search/retrieval and collection primitives
- indexing content
- storing and serving content copies
- content and index freshness 
- framework for content app understanding
- base classes for filesystem et al collection

Not responsible for:
- maintaining the ontology
- content classification

Depends on:
- consumes the ontology, delegating content classification
- 
Interfaces:
- content browser and UI for operation
- http and Python api

#### Context (cross-cutting)
System provides base classes/protocols and registries, as well as common classes like ObsidianVaultCollector which can be subclasses or parameterized.
System provides base classes for Entity and Resource, consuming app (of which Substrate is the only one) defines the context and this is used to build out the database schema and populate it.
Separate contexts are not mixed, there should be strict separation at the database level.

User can define:
- content-source-app classes for their data catalog; includes which files to collect, how different file types should be indexed, how to translate metadata into their ontology
- classes for the various resource types they are interested in
- classes for the entity types they are interested in
- cues to inform entity linking, like areas of interest

Depends on:
- provides classes to build out the ontology
- provides classes for the apps they use
- 

———

## Features
### Ontology Engineering and Labeling
Improve the ontology given a document which has ontologically-significant content.

- Identify new entities, concepts, topics etc
- identify entities that we aren’t interested in and can exclude from future classification
- Improve classification ability when encountering text that isn’t properly classified in the way it should have been
 
### File this for me
Classifies the input text, resource, or document to perform a store operation or suggestion.
#### use case: Save text fragment to existing document
Given a blob of text, or a url, find the document that should contain that.
#### use case: Ensure this document I’m going to store has the right classification
Given some text that won’t be appended to another document, allow a user to provide hints to assist correct classification.
#### use case: integration with user tools
- create or update existing documents in some other tool - an improvement to the use cases above 
- update document metadata in other tools, for example update obsidian frontmatter when an obsidian document is classified

### Search/Retrieval
#### Use case: Find resources related to an entity
Many resources may be related to an entity, but the end user should not need to associate that resource with the entity - the system should do so, and provide the ability to surface those resources easily.

#### use case: where’s my project for X
My code projects are indexed for text 

#### use case: task lists (Todoist, ClickUp)
#### use case: logbook 
#### use case: tip of my tongue

### Distillation
#### use case: collections
Given a list of known-interesting flagged topics, allow a user to associate a new or existing resource with a topic with low effort.
#### use case: magnetic folders
Given a known-interesting flagged topic, aggressively collect related resources.

#### use case: topic distillation
Pre-computed search results in virtual document form that link text chunks or documents together using a single Topic leaf node.
Depends-on:: Epic: Storage and classification of sub-document chunks
I should not need to explicitly create “collections” for everything; classification (topic modeling and entity resolution) at the chunk level (h3 etc) should be able to create enough “related content chunks” to distill NEW virtual documents representing some leaf node topic. If I save a block of text talking about “chunking source code”, and there are N other text chunks discussing same, I shouldn’t even need to query for this; there should be a topic that links to content chunks.

———
