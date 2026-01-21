# Linked Data and Ontologies (RDF et al)

## Explanation of RDF and OWL

First, realize that you want RDF, because that’s at the top of the chain - defining the lowest-level primitives like `rdf:type` and `rdf:resource`. Its like the ontology for defining ontologies. “rdf” is the “RDF syntax namespace” (`<http://www.w3.org/1999/02/22-rdf-syntax-ns#>`).

Then, we have “RDFs” which is the RDF Schema (`<http://www.w3.org/2000/01/rdf-schema#>`); this defines things like `subclassOf` 

The next layer is OWL (Web Ontology Language) which defines things useful for modeling domains, like `owl:Class`, `owl:DataProeprty` , `owl:NamedIndividual`etc.

The steps above are explained nicely here: 

[OntoGPT - Starting with OWL](https://monarch-initiative.github.io/ontogpt/start_with_owl/)

Going further, than OWL, we find ontologies for specific domains - there are other common ontologies that reuse OWL to build for like information stuff, and are used widely.

Some widely used ontologies are:

- [FOAF](http://xmlns.com/foaf/spec/) (friend of a friend) “ linking people and information across the web”

- [SKOS](https://www.w3.org/TR/skos-reference/#concepts) - Simple Knowledge Organization System “sharing and linking knowledge organization systems via the web;”

- Dublincore Metadata Terms ([DCMI](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/))

## Upper Ontologies

<https://en.wikipedia.org/wiki/Upper_ontology>

DublinCore is an upper ontology; they distribute DCMI (see below)