# Test Cases

Test cases for pmd (personal metadata) search system.

## Search Pipeline (tests/unit/search/)

### Pipeline Contract Tests (tests/unit/search/test_pipeline_contracts.py)

Contract tests for HybridSearchPipeline using in-memory fakes. These tests verify
pipeline behavior without database or LLM infrastructure dependencies.

#### Basic Search Flow

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_fts_only_search | Returns FTS results when only text_searcher provided | Pass |
| test_fts_and_vector_search | Combines FTS and vector results via RRF | Pass |
| test_collection_filter_applied | Collection ID passed through to searchers | Pass |

#### RRF Fusion

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_documents_found_by_multiple_sources_ranked_higher | Docs in multiple lists get boosted | Pass |
| test_rrf_weights_affect_ranking | Different FTS/vec weights affect ranking | Pass |

#### Metadata Boost

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_metadata_boost_disabled_by_default | Not applied unless enabled | Pass |
| test_metadata_boost_increases_scores | Matching tags boost scores | Pass |
| test_metadata_boost_with_no_query_tags | No boost when no tags inferred | Pass |

#### Reranking

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_reranking_disabled_by_default | Not applied unless enabled | Pass |
| test_reranking_reorders_results | Reranker scores affect ordering | Pass |

#### Query Expansion

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_expansion_disabled_by_default | Not used unless enabled | Pass |
| test_expansion_adds_results | Expanded queries contribute results | Pass |

#### Tag Retrieval

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_tag_retrieval_disabled_by_default | Not included in RRF unless enabled | Pass |
| test_tag_retrieval_adds_to_rrf | Tag results included in RRF when enabled | Pass |

#### Score Normalization

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_scores_normalized_by_default | Scores normalized to 0-1 range | Pass |
| test_normalization_can_be_disabled | Normalization disabled via config | Pass |

#### Error Handling

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_empty_results | Handles empty result lists gracefully | Pass |
| test_min_score_filter | Results below min_score filtered | Pass |
| test_limit_respected | Result count respects limit | Pass |

#### Configuration Defaults

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_default_weights | Default weight values are reasonable | Pass |
| test_default_features_disabled | Optional features disabled by default | Pass |
| test_default_normalization_enabled | Score normalization enabled by default | Pass |

### Pipeline Metadata Boost Integration

Tests for `_apply_metadata_boost` method in HybridSearchPipeline.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_metadata_boost_disabled_by_default | Metadata boost disabled when not configured | Pass |
| test_metadata_boost_requires_both_matcher_and_repo | Requires both matcher and repo | Pass |
| test_apply_metadata_boost_returns_unchanged_without_matcher | Returns unchanged without matcher | Pass |
| test_apply_metadata_boost_with_no_query_tags | Returns unchanged when no tags inferred | Pass |
| test_apply_metadata_boost_boosts_matching_docs | Boosts and reorders matching docs | Pass |
| test_apply_metadata_boost_handles_errors_gracefully | Returns original on error | Pass |
| test_default_metadata_boost_disabled | Config defaults to disabled | Pass |
| test_custom_metadata_boost_config | Accepts custom boost config | Pass |

### Pipeline Ontology Integration

Tests for ontology-aware boosting in HybridSearchPipeline.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_ontology_disabled_by_default | Ontology None when not provided | Pass |
| test_ontology_cleared_when_metadata_boost_disabled | Ontology cleared when boost disabled | Pass |
| test_ontology_preserved_when_metadata_boost_enabled | Ontology preserved when enabled | Pass |
| test_uses_v2_boost_when_ontology_provided | Uses apply_metadata_boost_v2 with ontology | Pass |
| test_uses_v1_boost_without_ontology | Uses v1 boost when no ontology | Pass |
| test_ontology_boost_with_custom_config | Custom boost config works with ontology | Pass |
| test_ontology_boost_reorders_by_boosted_score | Reorders by boosted scores | Pass |

### Ontology (tests/unit/search/test_ontology.py)

Tests for Ontology class hierarchical tag matching.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_init_with_adjacency | Initialize with adjacency dict | Pass |
| test_has_tag_parent | Recognize parent tags | Pass |
| test_has_tag_child | Recognize child tags | Pass |
| test_has_tag_unknown | Return False for unknown tags | Pass |
| test_get_description | Return description for known tags | Pass |
| test_get_description_unknown | Return None for unknown tags | Pass |
| test_all_tags | Return all known tags | Pass |
| test_empty_ontology | Handle empty adjacency | Pass |
| test_direct_parent | Find direct parent | Pass |
| test_multiple_ancestors | Find multiple ancestor levels | Pass |
| test_no_ancestors | Root tags have no ancestors | Pass |
| test_max_hops_limit | Respect max_hops limit | Pass |
| test_max_hops_zero | max_hops=0 returns empty | Pass |
| test_unknown_tag_ancestors | Unknown tags return empty | Pass |
| test_direct_children | Return direct children | Pass |
| test_leaf_has_no_children | Leaf tags have no children | Pass |
| test_unknown_tag_children | Unknown tags return empty | Pass |
| test_all_descendants | Return all descendants | Pass |
| test_direct_children_only | max_depth=1 returns direct children | Pass |
| test_leaf_has_no_descendants | Leaf tags have no descendants | Pass |
| test_single_tag_exact_match | Single tag gets weight 1.0 | Pass |
| test_expands_to_parent | Expand to parent with reduced weight | Pass |
| test_expands_multiple_levels | Multiple levels with decreasing weights | Pass |
| test_multiple_tags | Expand multiple query tags | Pass |
| test_accepts_set | Accept set of tags | Pass |
| test_max_hops_limit | Respect max_hops in expansion | Pass |
| test_no_duplicate_weights | Keep highest weight for overlaps | Pass |
| test_exact_match_overrides_expansion | Exact match wins over expansion | Pass |
| test_empty_tags | Empty returns empty dict | Pass |
| test_unknown_tag_no_expansion | Unknown tags still get weight 1.0 | Pass |
| test_custom_parent_weight | Use custom parent_weight | Pass |
| test_deep_hierarchy | Handle deeply nested hierarchies | Pass |
| test_wide_hierarchy | Handle wide hierarchies (100 children) | Pass |
| test_tag_names_with_special_characters | Handle special characters | Pass |
| test_parent_weight_zero | parent_weight=0 gives 0 to ancestors | Pass |
| test_parent_weight_one | parent_weight=1.0 gives full weight | Pass |

### Metadata Scoring (tests/unit/search/test_metadata_scoring.py)

Tests for apply_metadata_boost and apply_metadata_boost_v2 functions.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_no_boost_without_query_tags | No boost when no query tags | Pass |
| test_boosts_matching_document | Boost documents with matching tags | Pass |
| test_multiple_tag_matches | Multiple matches increase boost | Pass |
| test_respects_max_boost | Respects max_boost cap | Pass |
| test_does_not_boost_unknown_doc | No boost for unknown docs | Pass |
| test_reorders_by_boosted_score | Reorders results by boosted score | Pass |
| test_preserves_original_score_info | Preserves original score in info | Pass |
| test_custom_config | Accepts custom config | Pass |
| test_empty_results_list | Handles empty results | Pass |
| test_single_result | Handles single result | Pass |
| test_all_docs_match | All docs match scenario | Pass |
| test_no_docs_match | No docs match scenario | Pass |
| test_partial_matches | Partial tag overlap | Pass |
| test_min_tags_for_boost | Respects min_tags_for_boost | Pass |
| test_zero_scores | Handles zero scores correctly | Pass |
| test_high_boost_factor | Handles high boost factors | Pass |

### Metadata Scoring V2 (Weighted)

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_v2_empty_query_tags | No boost with empty query tags | Pass |
| test_v2_weighted_boost_exact_match | Exact match (weight 1.0) boost | Pass |
| test_v2_weighted_boost_partial_weight | Partial weight boost (0.5) | Pass |
| test_v2_cumulative_weights | Multiple tag weights accumulate | Pass |
| test_v2_respects_max_boost | Respects max_boost cap | Pass |
| test_v2_unknown_doc_no_boost | No boost for unknown docs | Pass |
| test_v2_reorders_by_boosted_score | Reorders by boosted score | Pass |
| test_v2_preserves_matching_tag_weights | Preserves matching tag weights | Pass |
| test_v2_handles_empty_results | Handles empty results | Pass |

### Tag Inference (tests/unit/search/test_inference.py)

Tests for LexicalTagMatcher query-time tag inference.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_exact_match | Match exact tag names | Pass |
| test_alias_match | Match via aliases | Pass |
| test_prefix_match | Match via prefix (when enabled) | Pass |
| test_multiple_matches | Multiple tag matches | Pass |
| test_no_matches | No matches scenario | Pass |
| test_case_insensitive | Case insensitive matching | Pass |
| test_get_matching_tags | Returns set of matched tags | Pass |
| test_register_tags | Register tags correctly | Pass |
| test_register_alias | Register aliases correctly | Pass |
| test_clear | Clear registered tags and aliases | Pass |
## Source Registry (tests/unit/sources/test_registry.py)

### SourceRegistry

Tests for SourceRegistry class that maps source_type strings to factory functions.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_register_and_create | Register factory and create source | Pass |
| test_create_unknown_type_raises | Unknown source_type raises ValueError | Pass |
| test_create_unknown_type_lists_available | Error message lists available types | Pass |
| test_overwrite_registration_replaces_factory | Overwriting replaces factory | Pass |
| test_unregister_removes_factory | Unregister removes factory | Pass |
| test_unregister_unknown_returns_false | Unregister unknown type returns False | Pass |
| test_is_registered | is_registered returns correct status | Pass |
| test_registered_types_sorted | registered_types returns sorted list | Pass |
| test_empty_registry_create_raises | Empty registry raises ValueError | Pass |
| test_filesystem_factory_registered_by_default | Default registry has filesystem | Pass |
| test_create_filesystem_source_from_collection | Create FileSystemSource from Collection | Pass |
| test_create_filesystem_source_with_source_config | FileSystemSource respects config | Pass |
| test_reset_default_registry | reset_default_registry clears singleton | Pass |
| test_get_default_registry_singleton | get_default_registry returns same instance | Pass |
| test_null_source_type_defaults_to_filesystem | collection.source_type=None uses filesystem | Pass |

## Loading Service (tests/unit/services/test_loading.py)

### LoadingService

Tests for LoadingService that handles document retrieval and preparation.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_load_eager_returns_all_documents | Eager mode returns complete list | Pass |
| test_load_stream_yields_documents | Stream mode yields as iterator | Pass |
| test_skip_unmodified_document | Respects check_modified returning False | Pass |
| test_skip_unchanged_content_hash | Skips when content hash matches | Pass |
| test_force_reloads_all | force=True ignores change detection | Pass |
| test_extracts_title_from_content | Title extraction fallback works | Pass |
| test_enumerated_paths_complete | All paths in enumerated_paths even if skipped | Pass |
| test_errors_captured_not_raised | Fetch errors added to errors list | Pass |
| test_resolves_source_from_collection | Source created from registry when None | Pass |
| test_collection_not_found_raises | Raises CollectionNotFoundError for unknown | Pass |

## LlamaIndex Loader Adapter (tests/unit/services/test_loading_llamaindex.py)

### URI Construction

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_uri_from_metadata_path | Uses metadata[uri_key] when present | Pass |
| test_uri_from_id | Falls back to id_ when no path | Pass |
| test_uri_from_hash | Falls back to content hash when no id | Pass |
| test_hash_includes_namespace | Hash fallback includes namespace for uniqueness | Pass |
| test_custom_uri_key | Uses custom uri_key when specified | Pass |

### Metadata Augmentation

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_metadata_augmentation | Extracted metadata preserved, LlamaIndex in _llamaindex | Pass |
| test_empty_llamaindex_metadata | Handles empty LlamaIndex metadata gracefully | Pass |

### Multi-Document Handling

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_single_doc_default | Raises ValueError if multiple docs and allow_multiple=False | Pass |
| test_allow_multiple_true | Accepts multiple docs when allow_multiple=True | Pass |
| test_duplicate_uri_rejected | Raises ValueError on duplicate URIs | Pass |

### Content and Title

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_content_type_propagated | content_type appears in LoadedDocument | Pass |
| test_title_from_metadata | Uses title from metadata when present | Pass |
| test_title_from_heading | Extracts title from markdown heading | Pass |
| test_title_fallback_to_path | Falls back to path when no title or heading | Pass |

### Content Extraction

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_content_from_text_attribute | Extracts content from .text attribute | Pass |
| test_content_from_get_content_method | Falls back to get_content() method | Pass |

### LoadingService Integration

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_collection_id_injected | LoadingService injects collection_id correctly | Pass |
| test_collection_not_found | Raises CollectionNotFoundError for unknown collection | Pass |
| test_enumerated_paths_populated | EagerLoadResult has enumerated_paths populated | Pass |

### Load Kwargs

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_load_kwargs_passed | load_kwargs are passed to loader.load_data() | Pass |

## Indexing Service (tests/unit/services/test_indexing.py)

### Backfill Metadata

Tests for `backfill_metadata` method, specifically exercising source_config JSON parsing
and Collection object creation (lines 709-724 in indexing.py).

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_backfill_metadata_empty_database | Returns zeros for empty database | Pass |
| test_backfill_metadata_extracts_tags | Extracts metadata from documents | Pass |
| test_backfill_metadata_skips_empty_body | Skips documents with empty content | Pass |
| test_backfill_metadata_with_source_config | Parses source_config JSON | Pass |
| test_backfill_metadata_with_null_source_config | Handles NULL source_config | Pass |
| test_backfill_metadata_with_metadata_profile_in_config | Uses metadata_profile from config | Pass |
| test_backfill_metadata_collection_filter | Filters by collection_name | Pass |
| test_backfill_metadata_force_reextracts | force=True re-extracts existing metadata | Pass |
| test_backfill_metadata_handles_extraction_errors | Captures errors and continues | Pass |
| test_backfill_metadata_with_empty_source_config_string | Handles empty string source_config | Pass |


## Metadata Integration (tests/unit/metadata/test_metadata_integration.py)

### Deprecation Shims

Tests that old import paths raise deprecation warnings.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_sources_metadata_shim_warns | pmd.sources.metadata raises DeprecationWarning | Pass |
| test_search_metadata_shim_warns | pmd.search.metadata raises DeprecationWarning | Pass |
| test_store_document_metadata_shim_warns | pmd.store.document_metadata raises DeprecationWarning | Pass |

### Cross-Module Interactions

Tests that types flow correctly between metadata submodules.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_extraction_produces_expected_metadata_type | Profiles return ExtractedMetadata | Pass |
| test_query_inference_with_model_aliases | Query uses aliases from model | Pass |
| test_stored_metadata_compatible_with_extracted | StoredDocumentMetadata from ExtractedMetadata | Pass |
| test_ontology_expansion_with_query_scoring | Ontology expands tags for scoring | Pass |
| test_registry_provides_correct_profile_types | Registry returns valid profiles | Pass |

### Public API Consistency

Tests that all expected types are exported from pmd.metadata.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_all_model_types_exported | Model types exported | Pass |
| test_all_extraction_types_exported | Extraction types exported | Pass |
| test_all_query_types_exported | Query types exported | Pass |
| test_store_types_exported | Store types exported | Pass |
| test_subpackage_imports_work | Direct subpackage imports work | Pass |


## Database Migrations (tests/unit/store/test_migrations.py)

### MigrationRunner

Tests for the versioned migration runner using SQLite PRAGMA user_version.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_fresh_database_starts_at_version_zero | New database has version 0 | Pass |
| test_run_applies_pending_migrations | run() applies all pending migrations | Pass |
| test_run_is_idempotent | Running twice is safe (no-op second time) | Pass |
| test_version_persists_across_connections | Version survives reconnect | Pass |
| test_get_migrations_returns_sorted_list | Migrations sorted by version | Pass |
| test_get_pending_migrations_filters_by_version | Only returns unapplied migrations | Pass |
| test_set_version_updates_user_version | Sets PRAGMA user_version | Pass |
| test_initial_migration_creates_tables | Creates all required tables | Pass |

### Database Integration

Tests for Database class migration integration.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_database_connect_runs_migrations | connect() runs migrations automatically | Pass |
| test_database_connect_is_idempotent | Reconnecting to existing DB works | Pass |
| test_database_preserves_data_across_migrations | Data not lost during migration | Pass |

### Upgrade Scenarios

Tests simulating database upgrades.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_upgrade_from_version_zero | DB at v0 upgrades to latest | Pass |
| test_already_migrated_database_no_op | Already current = no migration runs | Pass |
| test_migration_repr | Migration has useful repr | Pass |


## Application Module (tests/unit/app/)

### Application Class (tests/unit/app/test_application.py)

Tests for the Application composition root and lifecycle management.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_application_init | Application initializes with provided dependencies | Pass |
| test_application_db_property | db property returns database instance | Pass |
| test_application_config_property | config property returns configuration | Pass |
| test_application_vec_available | vec_available reflects db.vec_available | Pass |
| test_application_close_with_llm | close() closes LLM provider and database | Pass |
| test_application_close_without_llm | close() works when LLM provider is None | Pass |
| test_application_context_manager | Application works as async context manager | Pass |

### create_application Factory

Tests for the create_application factory function.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_create_application_returns_application | Returns configured Application | Pass |
| test_create_application_connects_database | Connects to database on creation | Pass |
| test_create_application_creates_services | Creates indexing, search, status services | Pass |
| test_create_application_handles_llm_provider_error | Handles LLM provider creation errors | Pass |
| test_create_application_cleans_up_on_context_exit | Context manager cleans up resources | Pass |
| test_create_application_services_have_dependencies | Services have proper dependencies | Pass |
| test_create_application_wires_loading_service | Application includes loading service | Pass |

## Loader + Indexer Integration (tests/integration/test_loader_indexer_flow.py)

### Full Flow Tests

Integration tests for LoadingService + IndexingService working together.

| Test Case | Description | Status |
|-----------|-------------|--------|
| test_full_index_flow_via_loader | Index collection through Application uses loader | Pass |
| test_stale_document_cleanup | Removed files are marked inactive after reindex | Pass |
| test_incremental_indexing | Only changed documents are reloaded | Pass |
| test_loader_accessible_on_application | Application exposes loading service | Pass |
