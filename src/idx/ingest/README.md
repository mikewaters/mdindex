# Ingestion

## Obsidian ingestion
### Data flow
- Create and persist dataset
- Read whole documents, get metadata and relationships, splitting off frontmatter into metadata (JSON utf-8 format)
- Persist documents and frontmatter to the database and to FTS index
- Chunk the documents smartly-ish
- Generate chunk-level embeddings and persist them in vector db

### Metadata extraction
A few types of metadata are collected from an Obsidian note; basic file-level, and frontmatter.
1. File-level metadata
* includes folder path, note name, ctime and in-vault wikilinks
* unresolveable wikilinks are stripped out (ex: partial vault import, dead links etc)

2. Frontmatter
* the yaml text is stripped from the documents before they are persisted
* frontmatter-based metadata is stored in its own JSON key, as-is
* frontmatter metadata is mapped to a single format based on the source (in this case, Obsidian)

### Metadata post-processing
- Frontmatter is analyzed and mapped based on user's ontology
- Everything is given an URI on the users ontology
