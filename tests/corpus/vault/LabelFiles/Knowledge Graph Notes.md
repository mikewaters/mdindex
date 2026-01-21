---
tags:
  - document 📑
---
# Knowledge Graph Notes

Interesting:

<https://github.com/modelcontextprotocol/servers/blob/main/src/memory/README.md>

A KG store might have the discussed structure:

Entity → Relation → Observation

Many times when I am capturing a thought, I’m making an observation about the entity (other times I may be creating or changing an entity or relation). When storing thoughts that are observations, the system should extract the entity or entities automatically.



### Just try asking Claude to make one

### Having an LLM construct a KG schema

- <https://blog.scottlogic.com/2024/05/16/navigating-knowledge-graphs-creating-cypher-queries-with-llms.html>

- <https://python.langchain.com/docs/how_to/graph_constructing/>

## Example

![image 11.png](./Knowledge%20Graph%20Notes-assets/image%2011.png)

Example cypher statement:

```json
CREATE (adam:User {name: 'Adam'}), (pernilla:User {name: 'Pernilla'}),   (david:User {name: 'David'}), (adam)-[:FRIEND]->(pernilla), (pernilla)-[:FRIEND]->(david)

```

Impl of a graph for code repos

<https://gitlab.com/gitlab-org/rust/knowledge-graph/-/blob/main/crates/database/src/graph/relationship.rs?ref_type=heads>