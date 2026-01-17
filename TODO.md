## Substrate Architecture
### Domains
#### The Ontology
Responsible for:
- maintaining entity and topic data stores
- providing classification primitives
#### Data Catalog

Responsible for:
- providing search/retrieval and collection primitives
- indexing content
- storing and serving content copies
- content and index freshness 
- framework for content app understanding

Not responsible for:
- maintaining the ontology
- content classification

Depends on:
- the ontology
- 
Interfaces:
- content browser 
#### User
User can define:
- content source app classes
- how each source should be indexed
- 
 
## Tasks

- remove _index_via_legacy and anything like it
- move LLMs to dspy
- use a better chunker wtpsplit sat-61-sm