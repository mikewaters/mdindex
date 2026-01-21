# Ontology Service

> what is this from?:
>
> OODA loop: observe orient decide act

Is a:: Solution

Part of:: [LifeOS Strategy.md](./LifeOS%20Strategy.md) —> [PoC System (³) Assemblage.md](./PoC%20System%20\(³\)%20Assemblage.md)

This is essentially the same as [Registry Service.md](./Registry%20Service.md) [Resource Service.md](./Resource%20Service.md) [Relationship Service.md](./Relationship%20Service.md) [LifeData Service.md](./LifeData%20Service.md)

- [ ] Todo: merge other “services” from system3 with this one in my system3/lifeos solution definition stuff

## Dependencies

1. I need to have data in the registry system before this can be useful.

## Features

In these use cases, the term “Item” refers to a text string that I have decided is meaningful (interesting, important, relevant etc). This text string may refer to a Topic, Subject, Concept, Project, Goal etc, and thus the item is part of the Registry and is incorporated into my ontology (in that it carries its type descriptor, from some domain, at a minimum). 

The Registry is just a list of items, their types in the ontology, related resources, and other related items. Items in the Registry include Subjects or Topics of note, Knowledge Management Concepts, named Efforts, personal Goals etc.

The term ”Resource” refers to some document that is associated with one or more items. Resources belong to the Information Domain, and are thusly associated with my ontology. Resources may relate to many items, for example a Project, a Concept, and one or many Topics (which themselves belong in a hierarchy/taxonomy). Resources may be Notes from Heptabase or Obsidian, bookmarks in Raindrop, pages in Evernote etc. 

### Search the Registry for an item

Digital version of “I know I put this somewhere…”

Problems addressed:

- Where am I writing about X, or collecting data/researching for later

- What term did I use for X?

- Where am I documenting the Objective|Solution|Problem|Vision|Project etc Y, where can I find it to add info, or read info

- I am using the wrong precise term for a thing, and so can’t find where it is

User stories:

1. Find named item (exact match)

2. Find similar item

3. Typeahead matching

Returns a list of matches and their type in the ontology.

### Locate a specific Resource for an item

Given all artifacts are attached to their correct terms and tags, and each artifact has a type within the Information Domain, and each artifact has an URL, I can extract that information given the item so I can use a resource 

Problems addressed:

- Where are the prompts for Project P

- Which URLs did I use for researching R?

- List all resources for Objective O

### Find Related Items

## Implementations

### Mobile

Simple lookup for when I am on my phone. Expo Go looks like an easy way to do this. Research in Perplexity

### Service

Something composable for other solution- this is the foundation of everything else.


