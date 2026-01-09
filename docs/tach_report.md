
[ Usages of 'src/pmd/sources' ]
src/pmd/cli/commands/index.py:8: Import 'src.pmd.sources.get_default_registry'
src/pmd/services/indexing.py:17: Import 'src.pmd.sources.DocumentReference'
src/pmd/services/indexing.py:18: Import 'src.pmd.sources.DocumentSource'
src/pmd/services/indexing.py:19: Import 'src.pmd.sources.FetchResult'
src/pmd/services/indexing.py:20: Import 'src.pmd.sources.SourceFetchError'
src/pmd/services/indexing.py:21: Import 'src.pmd.sources.SourceListError'
src/pmd/services/indexing.py:22: Import 'src.pmd.sources.SourceRegistry'
src/pmd/services/indexing.py:23: Import 'src.pmd.sources.get_default_registry'
src/pmd/services/indexing.py:26: Import 'src.pmd.sources.metadata.get_default_profile_registry'
src/pmd/services/indexing.py:311: Import 'src.pmd.sources.FetchResult'
-------------------------------

[ External Dependencies in 'pmd/sources' ]
src/pmd/sources/content/filesystem.py:25: Import 'pmd.metadata.ExtractedMetadata' from package 'pmd'
src/pmd/sources/content/base.py:13: Import 'pmd.core.exceptions.PMDError' from package 'pmd'
src/pmd/sources/content/base.py:14: Import 'pmd.metadata.ExtractedMetadata' from package 'pmd'
src/pmd/sources/metadata/registry.py:15: Import 'pmd.sources.metadata.drafts._detect_drafts_content' from package 'pmd'
src/pmd/sources/metadata/registry.py:16: Import 'pmd.sources.metadata.obsidian._detect_obsidian_content' from package 'pmd'
src/pmd/sources/metadata/drafts.py:6: Import 'pmd.metadata.ExtractedMetadata' from package 'pmd'
src/pmd/sources/metadata/drafts.py:7: Import 'pmd.metadata.extract_inline_tags' from package 'pmd'
src/pmd/sources/metadata/drafts.py:8: Import 'pmd.metadata.extract_tags_from_field' from package 'pmd'
src/pmd/sources/metadata/drafts.py:9: Import 'pmd.metadata.parse_frontmatter' from package 'pmd'
src/pmd/sources/metadata/obsidian.py:6: Import 'pmd.metadata.ExtractedMetadata' from package 'pmd'
src/pmd/sources/metadata/obsidian.py:7: Import 'pmd.metadata.extract_inline_tags' from package 'pmd'
src/pmd/sources/metadata/obsidian.py:8: Import 'pmd.metadata.extract_tags_from_field' from package 'pmd'
src/pmd/sources/metadata/obsidian.py:9: Import 'pmd.metadata.parse_frontmatter' from package 'pmd'
src/pmd/sources/metadata/base.py:8: Import 'pmd.metadata.ExtractedMetadata' from package 'pmd'
src/pmd/sources/metadata/base.py:9: Import 'pmd.metadata.extract_inline_tags' from package 'pmd'
src/pmd/sources/metadata/base.py:10: Import 'pmd.metadata.extract_tags_from_field' from package 'pmd'
src/pmd/sources/metadata/base.py:11: Import 'pmd.metadata.parse_frontmatter' from package 'pmd'
-------------------------------

[ Usages of 'src/pmd/core' ]
src/pmd/cli/commands/collection.py:7: Import 'src.pmd.core.config.Config'
src/pmd/cli/commands/collection.py:8: Import 'src.pmd.core.exceptions.CollectionExistsError'
src/pmd/cli/commands/collection.py:8: Import 'src.pmd.core.exceptions.CollectionNotFoundError'
src/pmd/cli/commands/index.py:5: Import 'src.pmd.core.config.Config'
src/pmd/cli/commands/index.py:6: Import 'src.pmd.core.exceptions.CollectionNotFoundError'
src/pmd/cli/commands/search.py:11: Import 'src.pmd.core.config.Config'
src/pmd/cli/commands/status.py:5: Import 'src.pmd.core.config.Config'
src/pmd/cli/main.py:9: Import 'src.pmd.core.config.Config'
src/pmd/cli/main.py:183: Import 'src.pmd.core.instrumentation.configure_phoenix_tracing'
src/pmd/llm/base.py:5: Import 'src.pmd.core.types.EmbeddingResult'
src/pmd/llm/base.py:5: Import 'src.pmd.core.types.RerankResult'
src/pmd/llm/embeddings.py:7: Import 'src.pmd.core.config.Config'
src/pmd/llm/embeddings.py:8: Import 'src.pmd.core.types.Chunk'
src/pmd/llm/factory.py:5: Import 'src.pmd.core.config.Config'
src/pmd/llm/lm_studio.py:8: Import 'src.pmd.core.config.LMStudioConfig'
src/pmd/llm/lm_studio.py:9: Import 'src.pmd.core.types.EmbeddingResult'
src/pmd/llm/lm_studio.py:9: Import 'src.pmd.core.types.RerankDocumentResult'
src/pmd/llm/lm_studio.py:9: Import 'src.pmd.core.types.RerankResult'
src/pmd/llm/mlx_provider.py:12: Import 'src.pmd.core.config.MLXConfig'
src/pmd/llm/mlx_provider.py:13: Import 'src.pmd.core.types.EmbeddingResult'
src/pmd/llm/mlx_provider.py:13: Import 'src.pmd.core.types.RerankDocumentResult'
src/pmd/llm/mlx_provider.py:13: Import 'src.pmd.core.types.RerankResult'
src/pmd/llm/mlx_provider.py:152: Import 'src.pmd.core.instrumentation.traced_mlx_embed'
src/pmd/llm/mlx_provider.py:256: Import 'src.pmd.core.instrumentation.traced_mlx_generate'
src/pmd/llm/openrouter.py:8: Import 'src.pmd.core.config.OpenRouterConfig'
src/pmd/llm/openrouter.py:9: Import 'src.pmd.core.types.EmbeddingResult'
src/pmd/llm/openrouter.py:9: Import 'src.pmd.core.types.RerankDocumentResult'
src/pmd/llm/openrouter.py:9: Import 'src.pmd.core.types.RerankResult'
src/pmd/llm/reranker.py:28: Import 'src.pmd.core.types.RankedResult'
src/pmd/llm/reranker.py:28: Import 'src.pmd.core.types.RerankDocumentResult'
src/pmd/llm/reranker.py:28: Import 'src.pmd.core.types.RerankResult'
src/pmd/mcp/server.py:3: Import 'src.pmd.core.config.Config'
src/pmd/mcp/server.py:4: Import 'src.pmd.core.exceptions.CollectionNotFoundError'
src/pmd/search/chunking.py:5: Import 'src.pmd.core.config.ChunkConfig'
src/pmd/search/chunking.py:6: Import 'src.pmd.core.types.Chunk'
src/pmd/search/fusion.py:9: Import 'src.pmd.core.types.RankedResult'
src/pmd/search/fusion.py:9: Import 'src.pmd.core.types.SearchResult'
src/pmd/search/fusion.py:9: Import 'src.pmd.core.types.SearchSource'
src/pmd/search/pipeline.py:48: Import 'src.pmd.core.types.RankedResult'
src/pmd/search/pipeline.py:48: Import 'src.pmd.core.types.SearchResult'
src/pmd/search/scoring.py:57: Import 'src.pmd.core.types.RankedResult'
src/pmd/search/scoring.py:57: Import 'src.pmd.core.types.RerankDocumentResult'
src/pmd/services/container.py:9: Import 'src.pmd.core.config.Config'
src/pmd/services/indexing.py:13: Import 'src.pmd.core.exceptions.CollectionNotFoundError'
src/pmd/services/indexing.py:14: Import 'src.pmd.core.types.Collection'
src/pmd/services/search.py:9: Import 'src.pmd.core.types.RankedResult'
src/pmd/services/search.py:9: Import 'src.pmd.core.types.SearchResult'
src/pmd/services/status.py:9: Import 'src.pmd.core.types.IndexStatus'
src/pmd/store/collections.py:10: Import 'src.pmd.core.exceptions.CollectionExistsError'
src/pmd/store/collections.py:10: Import 'src.pmd.core.exceptions.CollectionNotFoundError'
src/pmd/store/collections.py:11: Import 'src.pmd.core.types.Collection'
src/pmd/store/database.py:11: Import 'src.pmd.core.exceptions.DatabaseError'
src/pmd/store/documents.py:7: Import 'src.pmd.core.exceptions.DocumentNotFoundError'
src/pmd/store/documents.py:8: Import 'src.pmd.core.types.DocumentResult'
src/pmd/store/embeddings.py:9: Import 'src.pmd.core.types.SearchResult'
src/pmd/store/embeddings.py:9: Import 'src.pmd.core.types.SearchSource'
src/pmd/store/search.py:36: Import 'src.pmd.core.types.SearchResult'
src/pmd/store/search.py:36: Import 'src.pmd.core.types.SearchSource'
src/pmd/store/vector_search.py:7: Import 'src.pmd.core.types.SearchResult'
-------------------------------

[ Usages of 'src/pmd/llm' ]
src/pmd/services/container.py:10: Import 'src.pmd.llm.create_llm_provider'
src/pmd/services/container.py:11: Import 'src.pmd.llm.embeddings.EmbeddingGenerator'
src/pmd/services/container.py:12: Import 'src.pmd.llm.query_expansion.QueryExpander'
src/pmd/services/container.py:13: Import 'src.pmd.llm.reranker.DocumentReranker'
-------------------------------


[ Usages of 'src/pmd/metadata' ]
src/pmd/services/indexing.py:25: Import 'src.pmd.metadata.ExtractedMetadata'
-------------------------------

[ Usages of 'src/pmd/search' ]
src/pmd/llm/embeddings.py:9: Import 'src.pmd.search.chunking.chunk_document'
src/pmd/llm/embeddings.py:10: Import 'src.pmd.search.text.is_indexable'
src/pmd/services/indexing.py:15: Import 'src.pmd.search.text.is_indexable'
src/pmd/services/search.py:10: Import 'src.pmd.search.pipeline.HybridSearchPipeline'
src/pmd/services/search.py:10: Import 'src.pmd.search.pipeline.SearchPipelineConfig'
-------------------------------

[ External Dependencies in 'pmd/search' ]
src/pmd/search/pipeline.py:51: Import 'pmd.metadata.Ontology' from package 'pmd'
src/pmd/search/fusion.py:7: Import 'loguru.logger' from package 'loguru'
src/pmd/search/metadata/retrieval.py:13: Import 'loguru.logger' from package 'loguru'
src/pmd/search/metadata/retrieval.py:15: Import 'pmd.core.types.SearchResult' from package 'pmd'
src/pmd/search/metadata/retrieval.py:15: Import 'pmd.core.types.SearchSource' from package 'pmd'
src/pmd/search/metadata/inference.py:225: Import 'pmd.metadata.load_default_aliases' from package 'pmd'
-------------------------------

[ Usages of 'src/pmd/services' ]
src/pmd/cli/commands/index.py:7: Import 'src.pmd.services.ServiceContainer'
src/pmd/cli/commands/search.py:12: Import 'src.pmd.services.ServiceContainer'
src/pmd/cli/commands/status.py:6: Import 'src.pmd.services.ServiceContainer'
src/pmd/mcp/server.py:5: Import 'src.pmd.services.ServiceContainer'
-------------------------------

[ Usages of 'src/pmd/store' ]
src/pmd/cli/commands/collection.py:9: Import 'src.pmd.store.collections.CollectionRepository'
src/pmd/cli/commands/collection.py:10: Import 'src.pmd.store.database.Database'
src/pmd/llm/embeddings.py:11: Import 'src.pmd.store.database.Database'
src/pmd/llm/embeddings.py:12: Import 'src.pmd.store.embeddings.EmbeddingRepository'
src/pmd/search/pipeline.py:49: Import 'src.pmd.store.search.FTS5SearchRepository'
src/pmd/services/container.py:14: Import 'src.pmd.store.collections.CollectionRepository'
src/pmd/services/container.py:15: Import 'src.pmd.store.database.Database'
src/pmd/services/container.py:16: Import 'src.pmd.store.documents.DocumentRepository'
src/pmd/services/container.py:17: Import 'src.pmd.store.embeddings.EmbeddingRepository'
src/pmd/services/container.py:18: Import 'src.pmd.store.search.FTS5SearchRepository'
src/pmd/services/indexing.py:27: Import 'src.pmd.store.source_metadata.SourceMetadata'
src/pmd/services/indexing.py:27: Import 'src.pmd.store.source_metadata.SourceMetadataRepository'
src/pmd/services/indexing.py:28: Import 'src.pmd.store.document_metadata.DocumentMetadataRepository'
src/pmd/services/indexing.py:151: Import 'src.pmd.store.source_metadata.SourceMetadataRepository'
src/pmd/services/indexing.py:571: Import 'src.pmd.store.document_metadata.StoredDocumentMetadata'
-------------------------------

[ Usages of 'src/pmd/utils' ]
src/pmd/services/indexing.py:214: Import 'src.pmd.utils.hashing.sha256_hash'
src/pmd/store/documents.py:9: Import 'src.pmd.utils.hashing.sha256_hash'

