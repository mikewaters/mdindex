## Substrate Architecture
### Domains
#### The Ontology
Responsible for:
- maintaining entity and topic data stores
- providing classification primitives
- support ontology improvement
- can be configured for each context (life, work, interest etc)
- 
#### Data Catalog

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

#### Context (Life)
User can define:
- content source app classes for their data catalog
- can configure how each source should be indexed
- 

Depends on:
- provides classes to build out the ontology
- provides classes for the apps they use
- 

## Tasks
- migrate external config to Pydantic so cli, api, and Python clients can use the same validated structures
- 


- remove _index_via_legacy and anything like it
- move LLMs to dspy
- use a better chunker wtpsplit sat-61-sm