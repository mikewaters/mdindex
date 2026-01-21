# What is an Ontology

## Definition

Ontology: a shared conceptualization of the objects, concepts, and other entities that are assumed to exist in a particular domain, that is formally specified.

> From the perspective of computer science, an ontology has been defined as a shared conceptualization (of the “objects, concepts, and other entities that are assumed to exist” in a particular domain) that is formally specified ([Gruber, 1995](https://nap.nationalacademies.org/read/26464/chapter/5#chapter03_pz83-17), p. 908; [Gruber, 1993](https://nap.nationalacademies.org/read/26464/chapter/5#chapter03_pz83-16); see also [Studer et al., 1998](https://nap.nationalacademies.org/read/26464/chapter/5#chapter03_pz85-10)). 
>
> This definition emerged from a study by the U.S. Defense Advanced Research Projects Agency (DARPA) of how knowledge could be shared across computer systems ([Neches et al., 1991](https://nap.nationalacademies.org/read/26464/chapter/5#chapter03_pz84-16)). In the early 1990s, participants in this DARPA initiative argued that artificial intelligence (AI) would require the use of standard ontologies to ground content-specific agreements for the sharing and reuse of knowledge among software systems.

> in the context of computer and information science, an ontology refers to a specification of entities within a domain, which **loosely parallels the philosophical definition of ontology as the science or study of existence** (e.g., Stanford Encyclopedia of Philosophy[2](https://nap.nationalacademies.org/read/26464/chapter/5#chapter03_pz64-5)).

> source: 
>
> [National Academies Press - Understanding Ontologies in the Behavioral Sciences: Accelerating Research and the Spread of Knowledge](https://nap.nationalacademies.org/read/26464/chapter/5)

### Complexity

Ontologies exist within a spectrum of “semantic complexity”:

> classification systems designed for ontological purposes (the specification of definitions and relationships) may include weak semantics (such as a simple taxonomy that specifies only class—subclass relationships) or strong semantics (such as formal representation in a logic that allows developers to specify the properties of entities and constraints on those properties).[4](https://nap.nationalacademies.org/read/26464/chapter/5#chapter03_pz67-6)
>
> source: [Understanding Ontologies in the Behavioral Sciences](https://nap.nationalacademies.org/read/26464/chapter/5#chapter03_pz67-2)

This continuum might look something like:

```bash
Weak Semantics <----                                         ----> Strong Semantics
Lists <-> Thesaurus <-> Weak Hierarchy <-> Taxonomy <-> Formal Logic Representation
```

For this reason, I think it is OK for my ontology to be flexible and imperfect.

### Contrast with Taxonomy

> Where simple taxonomies are organized in terms of the basic is_a relation only, ontologies are organized also by other relations, such as part_of (‘*parthood*’)

+ ### Philosophical, Formal (Top-level), and Material (Domain) Ontologies

   > A formal ontology is domain neutral. It contains just those most general terms—such\
   > as “object” and “process”—which apply in all scientific disciplines whatsoever. Thus it\
   > corresponds to the sort of ontological interest we identified above as predominating\
   > among philosophers. A material (or “domain”) ontology is domain specific. It contains\
   > terms—such as “cell,” or “carburetor”—which apply only in a subset of disciplines.

   > Each domain ontology consists of a taxonomy (a hierarchy structured by the **is_a** relation) together with other **relations such as part_of, contained_in, adjacent_to, has_agent, preceded_by**, and so forth, along with definitions and axioms governing how its terms and relations are to be understood. A domain ontology is thus a taxonomy that has been enhanced to include more information about the universals, classes, and relations that it represents.

   #### **Philosophical Ontology**

   - The study of what is, of the kinds and structures of objects, properties, events, processes,\
      and relations in every area of reality (metaphysics). Results in ontologies, descriptions, or\
      theories of what exists, as representational artifacts.

   - Has roots in ancient Greece in the work of philosophers such as Parmenides, Heraclitus,\
      Plato, and Aristotle

   - Example: the Porphyrian Tree

   #### **Material or Domain Ontology**

   - A structured representation of the entities and relations existing within a particular domain\
      of reality such as medicine, geography, ecology, or law

   - A graph-theoretic structure whose nodes are linked by the subtype relation (thereby forming\
      a taxonomy) and by other relations

   - Goal: to support knowledge sharing and reuse

   - Examples: Gene Ontology (GO), Foundational Model of Anatomy (FMA), Environment\
      Ontology (EnvO), Chemical Entities of Biological Interest (ChEBI), and many others.

   #### **Formal or Top-Level Ontology**

   - Upper-level ontology that assists in making communication between and among domain\
      ontologies possible by providing a common ontological architecture

   - Goal: the calibration of interoperable domain ontologies into larger networks

   - Examples: Basic Formal Ontology (BFO), Descriptive Ontology for Linguistic and Cognitive\
      Engineering (DOLCE), Standard Upper Merged Ontology (SUMO)

### Resources

- [National Academies Press - Understanding Ontologies in the Behavioral Sciences: Accelerating Research and the Spread of Knowledge](https://nap.nationalacademies.org/read/26464/chapter/5)

- [Building-Ontologies-with-Basic-Formal-Ontology.pdf](../Card%20Library/Building-Ontologies-with-Basic-Formal-Ontology.pdf) (BFO)

### Examples

[Examples of Ontologies.md](./Examples%20of%20Ontologies.md)

[Examples of Taxonomies.md](./Examples%20of%20Taxonomies.md)

## Relations

> the set of relations in an ontology describes the [semantics](https://en.wikipedia.org/wiki/Semantics "Semantics") of the domain: that is, its various [semantic relations](https://en.wiktionary.org/wiki/Wiktionary:Semantic_relations "wikt:Wiktionary:Semantic relations"), such as [synonymy](https://en.wikipedia.org/wiki/Synonym "Synonym"), [hyponymy and hypernymy](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy "Hyponymy and hypernymy"), [coordinate](https://en.wiktionary.org/wiki/coordinate_term "wikt:coordinate term") relation, and others. The set of used relation types (classes of relations) and their subsumption hierarchy describe the expression power of the language in which the ontology is expressed.

### Is-a (Inheritance)

> An important type of relation is the [subsumption](https://en.wikipedia.org/wiki/Hierarchy#Subsumptive_containment_hierarchy "Hierarchy") relation (*is-a-[superclass](https://en.wikipedia.org/wiki/Superclass\_\\(knowledge_representation\\) "Superclass (knowledge representation)")\-of*, the converse of *[is-a](https://en.wikipedia.org/wiki/Is-a "Is-a")*, *is-a-subtype-of* or *is-a-[subclass](https://en.wikipedia.org/wiki/Subclass\_\\(knowledge_representation\\) "Subclass (knowledge representation)")\-of*). This defines which objects are classified by which class. For example, we have already seen that the class Ford Explorer *is-a-subclass-of* 4-Wheel Drive Car, which in turn *is-a-subclass-of* Car.
>
> The addition of the is-a-subclass-of relationships creates a [taxonomy](https://en.wikipedia.org/wiki/Taxonomy\_\\(general\\) "Taxonomy (general)"); a tree-like structure (or, more generally, a [partially ordered set](https://en.wikipedia.org/wiki/Partially_ordered_set "Partially ordered set")) that clearly depicts how objects relate to one another. In such a structure, each object is the 'child' of a 'parent class'

### Part-of (Composition)

> Another common type of relations is the [mereology](https://en.wikipedia.org/wiki/Mereology "Mereology") relation, written as *part-of*, that represents how objects combine to form composite objects. For example, if we extended our example ontology to include concepts like Steering Wheel, we would say that a "Steering Wheel is-by-definition-a-part-of-a Ford Explorer" since a steering wheel is always one of the components of a Ford Explorer. If we introduce meronymy relationships to our ontology, the hierarchy that emerges is no longer able to be held in a simple tree-like structure since now members can appear under more than one parent or branch. Instead this new structure that emerges is known as a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph "Directed acyclic graph").