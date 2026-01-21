# Finding Ontologies for LifeOS - Journal

- [ ] Review the ontologies in this note

Oct 1, 2025

## From data.world

KOS (Knowledge Organization System) defines a structured model that describes many types of cataloged entities. It includes a core model as well as a variety of extensions tailored for different technologies

---

**Semantic model**

The **Semantic Model** is represented by three key files: **the Ontology file, the Mappings file, and the Index file.**

- The **Ontology file** includes entities known as **Concepts** (e.g., customer or order), characteristics of these entities called **Attributes** (e.g., customer ID, first name, last name), and the **Relationships** that connect these entities (e.g., an order was made by a customer). These elements are collectively known as CARS.

- The **Mappings file** links the ontology to specific data sources, such as indicating that customer data is located in the customer table.

   Mappings within the system can be either direct or complex. Direct mappings are straightforward one-to-one correspondences between concepts and data, while complex mappings may involve additional clauses or conditions to specify relationships more intricately.

- The **Index file**, is a system file, which helps in vectorizing this information, storing it in a specialized knowledge graph vector. It is crucial for the system operation.

   These three files live within a **Project** in [data.world](data.world). It is associated with your particular AI context engine application.

---

> Identify the CARs in the question: Concepts, Attributes and Relationships (CARs)  that are associated with Concepts.



## Interesting Vocabularies

### RDA

<https://www.rdaregistry.info/termList/>

### COAR

<https://vocabularies.coar-repositories.org/resource_types/>

> The Resource Type vocabulary defines concepts to identify the genre of a resource. Such resources, like publications, research data, audio and video objects, are typically deposited in institutional and thematic repositories or published in ejournals.

### Open Vocabulary

[Open.vocab.org](Open.vocab.org)

Hundreds of useful terms, used by DBPedia and many others

### XSD Datatypes

<https://www.w3.org/TR/xmlschema11-2/>

### DCAT

> DCAT is an RDF vocabulary for representing data catalogs.

> <https://www.w3.org/TR/vocab-dcat-3/>

### DCMI Metadata Terms

> **Metadata**, literally "data about data" -- specifically, **descriptive metadata** -- is structured data about anything that can be named, such as Web pages, books, journal articles, images, songs, products, processes, people (and their activities), research data, concepts, and services. 
>
> **Dublin Core™ metadata** … is metadata designed for interoperability on the basis of **Semantic Web** or **Linked Data**principles. 

<https://www.dublincore.org/specifications/dublin-core/dcmi-terms/>

## Interesting Ontologies

### SKOS

> Using SKOS, **[concepts](https://www.w3.org/TR/skos-reference/#concepts)** can be identified using URIs, **[labeled](https://www.w3.org/TR/skos-reference/#labels)** with lexical strings in one or more natural languages, assigned **[notations](https://www.w3.org/TR/skos-reference/#notations)** (lexical codes), **[documented](https://www.w3.org/TR/skos-reference/#notes)** with various types of note, **[linked to other concepts](https://www.w3.org/TR/skos-reference/#semantic-relations)** and organized into informal hierarchies and association networks, aggregated into **[concept schemes](https://www.w3.org/TR/skos-reference/#schemes)**, grouped into labeled and/or ordered **[collections](https://www.w3.org/TR/skos-reference/#collections)**, and**[mapped](https://www.w3.org/TR/skos-reference/#mapping)** to concepts in other schemes.

**I think I am going to use this**



### DBPedia

[DBpedia - Ontology](https://dbpedia.org/ontology/)

>  this ontology corresponds to a new release of the DBpedia data set which contains instance data extracted from the different language versions of Wikipedia

DBPedia ontology classes in a nice list: <https://mappings.dbpedia.org/server/ontology/classes/>

CRAAAZY

### Asset Description

ADMS <https://www.w3.org/TR/vocab-adms/>

### PROV/PROV-O

<https://www.w3.org/ns/prov>

<https://www.w3.org/TR/prov-o/>

Provenance classes, like wasGeneratedBy, [wasDerivedFrom](https://www.w3.org/ns/prov#wasDerivedFrom), wasAttributedTo

### **Academic Research ontologies**

I am finding that academic research stuff is doing what i want, and way waaaay more.

### BIBO for bibliographies

Very simple, might work.

[Ontospy - http://purl.org/ontology/bibo/](https://dcmi.github.io/bibo/)

> The Bibliographic Ontology describes\
> bibliographic things on the semantic Web in RDF. This ontology can be\
> used as a citation ontology, as a document classification ontology, or\
> simply as a way to describe any kind of document in RDF. It has been\
> inspired by many existing document description metadata formats, and\
> can be used as a common ground for converting other bibliographic data\
> sources.

### FRBR for bibliographies

This one is really cool, it handles the modern world where pelople are mixing all their art together, and you might have ideas that come from other ideas and want to maintain that lineage. A little too much for my usage though.

[ Functional Requirements for Bibliographic Records](https://vocab.org/frbr/core.html)

Building on top of FRBR: <https://sparontologies.github.io/fabio/current/fabio.html>

## Ontologies for Dataset Hosting!

They have ontologies expressing systems information for dataset retrieval, which I would need as part of knowledge management:

### PROV-O

[prov:hadPrimarySource](https://www.w3.org/TR/prov-o/#hadPrimarySource) etc; See above

### DCAT

> DCAT is an RDF vocabulary for representing data catalogs.

<https://www.w3.org/TR/vocab-dcat-3/>

### Describing Linked Datasets with the VoID Vocabulary

> The Vocabulary of Interlinked Datasets (VoID) is concerned with *metadata about RDF datasets*. It is an RDF Schema vocabulary that provides terms and patterns for describing RDF datasets, and is intended as a bridge between the publishers and users of RDF data. VoiD descriptions can be used in many situations, ranging from data discovery to cataloging and archiving of datasets, but most importantly it helps users find the right data for their tasks.

<https://www.w3.org/TR/void/>

### SPARQL 1.1 Service Description

<https://www.w3.org/TR/sparql11-service-description/>

### a CSV Ontology?

<https://www.w3.org/ns/csvw>

>  creating Metadata descriptions for Tabular Data

## Interesting Composed Ontologies

An example of an ontology composed mostly of a bunch of others: 

> SemSur, the Semantic Survey Ontology, is a core ontology for describing individual research problems, approaches, implementations and evaluations in a structured, comparable way.
>
> <https://saidfathalla.github.io/SemSur/doc/> 

# Looking for reusable stuff online



## Searching for “vocabulary” terms

Google-fu: `rdf vocabulary for  …`

## Search resources

- [Linked Open Vocabularies](https://lov.linkeddata.es/dataset/lov/vocabs) - Vocabs search

- [Linked Open Vocabularies](https://lov.linkeddata.es/dataset/lov/terms) - Terms search