# Redesign of Abstractions

## Feature: Add dataset abstraction
Lives in the store module
Add new Dataset model (dataclass). This is 1-1 with SourceCollection, and handles creation or update of the SourceCollection, retrieving and caching, and persisting individual Resources. Dataset supports an indexing operation once this has been completed, which for now can call IndexingService.


Add new Resource model (sqlalchemy) database model. This represents a cached document or url, with all the source's attributes like modified data etc.  Reflects the "clean" source material. Will need to move DocumentCacher to the Resource level.
A Resource needs to know when it was last loaded (and how) and indexed (and how)

Add a Resources table, even URLs and stuff.  It will contain the URI, and generate a Document which could be the actual document content, or in the case of a url the cached web page, or the highlights etc. The Resource will be cached, not the document, so we can iterate on things like “collect resource type X”.
An obsidian resource would have the obsidian:// uri etc 
Resources will be cached in their original form with the addition of metadata if it can’t be an intrinsic part of the cached thing (like it’s created and modified dates)
- a Document is a thing to be indexed; a Resource is the representation of a thing in the world.

- Dataset creates collection, documents, all that stuff. it calls the repos.
- 

Move loader svc to extraction

Things that are static:
- database
- collections (rename to dataset)
- 

```python



```