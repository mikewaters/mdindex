# Runbook for Humans

**Agents should ignore this file**

### Changing the embedding model
If the embedding model for vector indexing is changed, all chunks will need to be reprocessed.  To properly re-embed with a new model, you must:

1. Clear pipeline cache (so docs aren't skipped)
rm -rf ~/.idx/cache/pipeline_storage/{dataset_name}/

2. Clear vector store (so old embeddings are gone)
rm -rf ~/.idx/vector_store/

3. Re-ingest (force=True not even needed now since cache is gone)