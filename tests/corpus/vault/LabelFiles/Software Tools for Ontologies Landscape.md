---
tags:
  - document 📑
  - landscape
---
# Software Tools for Ontologies Landscape

## Research - Github topics

- [Ontology engineering](https://github.com/topics/ontology-engineering)

- [ontologies](https://github.com/topics/ontologies)

## Examples of using ontologies in python

### “Building Ontologies with Python” article

> Let’s try building an ontology of Starbucks coffees and their characteristics. We will then use this to classify new, unknown coffees.

Comes with Jupiter notebooks

<https://paul-bruffett.medium.com/building-ontologies-with-python-84238d6eee52>

### CurateGPT

> LLM-driven curation assist tool

> CurateGPT is a prototype web application and framework for performing general purpose AI-guided curation and curation-related operations over *collections* of objects.

> <https://github.com/monarch-initiative/curategpt>

Looks like some guy’s passion project

Demo: <https://curategpt.io>

# Tools for creating ontologies

## OntoLearner

> Ontology Learning plays a crucial role in dynamically building and maintaining ontologies, which are essential for intelligent applications in knowledge graphs, information retrieval, question answering, and more. OntoLearner uses a LLMs4OL paradigm tasks to provide the capability to use LLMs for ontology learning tasks. The goal is to transition from raw text or structured corpora to formal knowledge representations such as classes, properties, and axioms.

[OntoLearner - Quickstart 0.1.0 documentation](https://ontolearner.readthedocs.io/quickstart.html#ontologizer)

-  supports multiple ontology formats (OWL, RDF, XML, TTL)

- python

### Features

- transforms ontologies into programmatically accessible Python objects

- learns a new ontology from data:

   - Term Typing

   - Taxonomy Discovery

   - Non-Taxonomic Relationship Extraction

- extracting ontological terms and types directly from raw text

- generate synthatic data for evaluating the task of term and type extraction from natural language text. It will generate a text corpus of documents aligned with a given ontology.

# Tools for mapping data to ontologies

## [OntoLearner](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/36aaf7f5-7907-476f-8fa8-ba4ad426911b#7795f4fd-aa1b-4198-b0f8-f3c56b20a90b)

## [OntoText Graphdb](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/1a9829f5-16db-4c10-86b1-fbbf048ea495#1d247180-d702-4122-bfe6-8bbcadfb9796)



# Tools for working within ontologies

## Ontology Access Kit (OAK)

> Ontology Access Toolkit (OAK) is a Python library for common [Ontology](https://incatools.github.io/ontology-access-kit/glossary.html#term-Ontology) operations over a variety of Adapters.

[oaklib documentation - Introduction](https://incatools.github.io/ontology-access-kit/introduction.html)

- python

### Features

- basic features of an [Ontology Element](https://incatools.github.io/ontology-access-kit/glossary.html#term-Ontology-Element), such as its [Label](https://incatools.github.io/ontology-access-kit/glossary.html#term-Label), [Definition](https://incatools.github.io/ontology-access-kit/glossary.html#term-Definition), [Relationships](https://incatools.github.io/ontology-access-kit/glossary.html#term-Relationship), or [Synonyms](https://incatools.github.io/ontology-access-kit/glossary.html#term-Synonym).

- Search an ontology for a term.

- Apply modifications to terms, including adding, deleting, or updating

- numerous specialized operations, such as [Graph Traversal](https://incatools.github.io/ontology-access-kit/glossary.html#term-Graph-Traversal), or Axiom processing, Ontology Alignment, or :term\`Text Annotation\`.

## DeepOnto

Tools and language models for working with ontologies, either by mapping classes/concepts within an ontology or across multiple ontologies.

Not for modeling content, more for ontologies themselves.

[DeepOnto - DeepOnto](https://krr-oxford.github.io/DeepOnto/)

> A package for ontology engineering with deep learning.

- Works with [OWL](https://en.wikipedia.org/wiki/Web_Ontology_Language)\-based ontologies.

- Python

- Has a language model: [Hugging Face - OntoLAMA · Datasets (krr-oxford)](https://huggingface.co/datasets/krr-oxford/OntoLAMA) **OntoLAMA: LAnguage Model Analysis for Ontology Subsumption Inference**

### Features

- **Ontology Reasoning** (`[OntologyReasoner](https://krr-oxford.github.io/DeepOnto/deeponto/onto/reasoning/#deeponto.onto.ontology.OntologyReasoner)`): Each instance of DeepOnto has a reasoner as its attribute. It is used for conducting reasoning activities, such as obtaining inferred subsumers and subsumees, as well as checking entailment and consistency. 

- **Ontology Verbalisation**: Verbalising concept expressions is very useful for models that take textual inputs. While the named concepts can be verbalised simply using their names (or labels), complex concepts that involve logical operators require a more sophisticated algorithm. See [verbalising ontology concepts](https://krr-oxford.github.io/DeepOnto/verbaliser).

- **Ontology Alignment**

   - Subsumption Prediction/Concept Mapping: Finding inheritance relationships between classes in an ontology

   - Ontology Mapping/Matching: [DeepOnto - Ontology Matching with BERTMap Family](https://krr-oxford.github.io/DeepOnto/bertmap/)

   - **Ontology Projection** (`[OntologyProjector](https://krr-oxford.github.io/DeepOnto/deeponto/onto/projection/#deeponto.onto.projection.OntologyProjector)`): The projection algorithm adopted in the OWL2Vec\* ontology embeddings is implemented here, which is to transform an ontology's TBox into a set of RDF triples. The relevant code is modified from the mOWL library.

- **Ontology Pruning**: Extracting a sub-ontology from an input ontology.

- **Ontology Normalisation** (`[OntologyNormaliser](https://krr-oxford.github.io/DeepOnto/deeponto/onto/normalisation/#deeponto.onto.normalisation.OntologyNormaliser)`): The implemented EL normalisation is also modified from the mOWL library, which is used to transform TBox axioms into normalised forms to support, e.g., geometric ontology embeddings.

- **Ontology Taxonomy** (`[OntologyTaxonomy](https://krr-oxford.github.io/DeepOnto/deeponto/onto/taxonomy/#deeponto.onto.taxonomy.OntologyTaxonomy)`): The taxonomy extracted from an ontology is a directed acyclic graph for the subsumption hierarchy, which is often used to support graph-based deep learning applications.

## OntolAligner

> Ontology Alignment (OA) is fundamental for achieving semantic interoperability across diverse knowledge systems

[OntoAligner - Documentation](https://ontoaligner.readthedocs.io/index.html)

<https://github.com/sciknoworg/OntoAligner>

- python

### Features

- **Ontology Alignment**

## OwlApy

> **Owlapy** is an open-source Python library designed for representing and manipulating OWL 2 ontologies, offering a robust foundation for knowledge graph and class expression learning projects in machine learning.

> <https://dice-group.github.io/owlapy/usage/main.html>

> <https://deepwiki.com/dice-group/owlapy>

- Python

## OwlReady

> Owlready2 is a module for ontology-oriented programming in Python 3. It can manage ontologies and knwoledge graphs, and includes an optimized RDF/OWL quadstore.

<https://pypi.org/project/owlready2/>

- python

### Features

- *Manipulates ontology classes, instances and properties transparently, as if they were normal Python objects*

- *Perform automatic classification of classes and instances, using the HermiT or Pellet reasoner (included)*

- *Native support for optimized SPARQL queries*

- *Finally, Owlready2 can also be used as an ORM (Object-Relational mapper) – as a graph/object database, it beats Neo4J, MongoDB, SQLObject and SQLAlchemy in terms of performances*

# Example codebases that match text to an ontology

## Text2Term

> #### A tool for mapping free-text descriptions of (biomedical) entities to ontology terms

> #### <https://github.com/rsgoncalves/text2term>

```bash
import text2term 
df = text2term.map_terms(source_terms=["asthma", "acute bronchitis"], 
                         target_ontology="http://purl.obolibrary.org/obo/mondo.owl")
```


