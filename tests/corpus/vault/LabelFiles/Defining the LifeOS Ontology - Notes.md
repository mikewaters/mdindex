# Defining the LifeOS Ontology - Notes



> this is similar to how semantic triples or RDF (Resource Description Framework) structures work. In these systems, the linking verb part is typically called a "predicate" or "relationship."
>
> So if we were to break down these examples structurally:
>
> - Subject + "is a" + Object (predicate expressing type/classification)
>
> - Subject + "is related to" + Object (predicate expressing general relationship)
>
> - Subject + "is a child of" + Object (predicate expressing hierarchical relationship)
>
> - Subject + "is a parent of" + Object (predicate expressing hierarchical relationship)
>
> - Subject + "should be added to" + Object (predicate expressing an action/directive)
>
> These predicates act as the connective tissue that defines how the subject and object relate to each other. Each one represents a different type of relationship or action between the terms it connects.

## Structure

### Statement

Subject —> Predicate —> Object

subjectType (more a *data* type: URL, Note, …)

predicateType (a *relationship* type: causal, temporal, …)

objectType (an *entity* type, part of some ontology, belonging to a domain)

Subject and Object types carry metadata


