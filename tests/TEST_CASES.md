# Test Cases

Test cases for pmd (personal metadata) search system.

## Search Pipeline (tests/unit/search/)

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

