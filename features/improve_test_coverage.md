# Test coverage gaps

src/pmd/metadata/aliases.py
src/pmd/sources/metadata/
src/pmd/search/metadata/

## src/pmd/metadata/ontology.py

### OntologyNode

**Test: OntologyNode initialization with minimal fields**
- Given: A tag string "ml"
- When: Creating an OntologyNode with only the tag
- Then: The node has the tag, empty children list, and empty description

**Test: OntologyNode initialization with all fields**
- Given: A tag "ml", children ["ml/supervised", "ml/unsupervised"], and description "Machine Learning"
- When: Creating an OntologyNode with all fields
- Then: All fields are set correctly

**Test: OntologyNode default factory for children**
- Given: Two OntologyNode instances created without explicit children
- When: Modifying the children list of one node
- Then: The other node's children list remains unaffected (ensuring separate list instances)

### Ontology.__init__

**Test: Ontology initialization with empty adjacency**
- Given: An empty adjacency dictionary
- When: Creating an Ontology
- Then: The ontology is created with no tags and default parent_weight of 0.7

**Test: Ontology initialization with custom parent_weight**
- Given: An adjacency dictionary and parent_weight of 0.5
- When: Creating an Ontology with parent_weight=0.5
- Then: The ontology's parent_weight is 0.5

**Test: Ontology initialization builds parent map**
- Given: An adjacency with "ml" having children ["ml/supervised"]
- When: Creating an Ontology
- Then: The _parent_map contains "ml/supervised" -> "ml"

### Ontology._build_parent_map

**Test: Build parent map with single-level hierarchy**
- Given: An ontology with "ml" having children ["ml/supervised", "ml/unsupervised"]
- When: Building the parent map
- Then: Both children map to "ml" as parent

**Test: Build parent map with multi-level hierarchy**
- Given: An ontology with "ml" -> ["ml/supervised"] and "ml/supervised" -> ["ml/supervised/regression"]
- When: Building the parent map
- Then: "ml/supervised" maps to "ml", and "ml/supervised/regression" maps to "ml/supervised"

**Test: Build parent map with missing children key**
- Given: An adjacency entry without a "children" key
- When: Building the parent map
- Then: No error occurs and the entry is skipped

### Ontology.get_ancestors

**Test: Get ancestors of leaf node**
- Given: A hierarchy "ml" -> "ml/supervised" -> "ml/supervised/regression"
- When: Getting ancestors of "ml/supervised/regression"
- Then: Returns ["ml/supervised", "ml"]

**Test: Get ancestors of root node**
- Given: A hierarchy with "ml" as root
- When: Getting ancestors of "ml"
- Then: Returns empty list

**Test: Get ancestors with max_hops limit**
- Given: A hierarchy "a" -> "b" -> "c" -> "d"
- When: Getting ancestors of "d" with max_hops=2
- Then: Returns ["c", "b"] (stops before "a")

**Test: Get ancestors of unknown tag**
- Given: An ontology without tag "unknown"
- When: Getting ancestors of "unknown"
- Then: Returns empty list

**Test: Get ancestors with max_hops=0**
- Given: A hierarchy with multiple levels
- When: Getting ancestors with max_hops=0
- Then: Returns empty list

### Ontology.get_children

**Test: Get children of parent node**
- Given: An ontology with "ml" having children ["ml/supervised", "ml/unsupervised"]
- When: Getting children of "ml"
- Then: Returns ["ml/supervised", "ml/unsupervised"]

**Test: Get children of leaf node**
- Given: An ontology where "ml/supervised" has no children
- When: Getting children of "ml/supervised"
- Then: Returns empty list

**Test: Get children of unknown tag**
- Given: An ontology without tag "unknown"
- When: Getting children of "unknown"
- Then: Returns empty list

### Ontology.get_descendants

**Test: Get descendants of parent with single level**
- Given: An ontology with "ml" having children ["ml/supervised", "ml/unsupervised"]
- When: Getting descendants of "ml"
- Then: Returns ["ml/supervised", "ml/unsupervised"]

**Test: Get descendants of parent with multiple levels**
- Given: A hierarchy "ml" -> ["ml/supervised"] -> ["ml/supervised/regression", "ml/supervised/classification"]
- When: Getting descendants of "ml"
- Then: Returns all descendants in breadth-first order

**Test: Get descendants with max_depth limit**
- Given: A 4-level deep hierarchy
- When: Getting descendants with max_depth=2
- Then: Returns only descendants up to 2 levels deep

**Test: Get descendants of leaf node**
- Given: An ontology where "ml/supervised/regression" has no children
- When: Getting descendants
- Then: Returns empty list

**Test: Get descendants with max_depth=0**
- Given: A hierarchy with multiple levels
- When: Getting descendants with max_depth=0
- Then: Returns empty list

### Ontology.has_tag

**Test: Check existence of parent tag**
- Given: An ontology with "ml" as parent
- When: Checking if "ml" exists
- Then: Returns True

**Test: Check existence of child tag**
- Given: An ontology with "ml/supervised" as child
- When: Checking if "ml/supervised" exists
- Then: Returns True

**Test: Check existence of unknown tag**
- Given: An ontology without tag "unknown"
- When: Checking if "unknown" exists
- Then: Returns False

### Ontology.get_description

**Test: Get description of tag with description**
- Given: An ontology with "ml" having description "Machine Learning"
- When: Getting description of "ml"
- Then: Returns "Machine Learning"

**Test: Get description of tag without description**
- Given: An ontology with "ml" having no description field
- When: Getting description of "ml"
- Then: Returns None

**Test: Get description of unknown tag**
- Given: An ontology without tag "unknown"
- When: Getting description of "unknown"
- Then: Returns None

### Ontology.expand_for_matching

**Test: Expand single leaf tag**
- Given: A hierarchy "ml" -> "ml/supervised" -> "ml/supervised/regression" with parent_weight=0.7
- When: Expanding ["ml/supervised/regression"]
- Then: Returns {"ml/supervised/regression": 1.0, "ml/supervised": 0.7, "ml": 0.49}

**Test: Expand root tag**
- Given: An ontology with "ml" as root
- When: Expanding ["ml"]
- Then: Returns {"ml": 1.0}

**Test: Expand multiple tags with shared ancestors**
- Given: "ml" -> ["ml/supervised", "ml/unsupervised"]
- When: Expanding ["ml/supervised", "ml/unsupervised"]
- Then: "ml" appears once with weight 0.7 (highest from either child)

**Test: Expand with max_hops limit**
- Given: A 4-level hierarchy with parent_weight=0.7
- When: Expanding leaf tag with max_hops=1
- Then: Only includes the tag and its immediate parent

**Test: Expand unknown tag**
- Given: An ontology without tag "unknown"
- When: Expanding ["unknown"]
- Then: Returns {"unknown": 1.0} (no ancestors added)

**Test: Expand with custom parent_weight**
- Given: An ontology with parent_weight=0.5 and hierarchy "a" -> "b"
- When: Expanding ["b"]
- Then: Returns {"b": 1.0, "a": 0.5}

**Test: Expand preserves highest weight when tags overlap**
- Given: Tags "a/b" (weight 1.0) and "a" (weight 1.0 explicit)
- When: Expanding ["a/b", "a"]
- Then: "a" has weight 1.0 (not reduced to 0.7 from ancestor expansion)

**Test: Expand with empty tag list**
- Given: Any ontology
- When: Expanding []
- Then: Returns {}

**Test: Expand with set input**
- Given: A hierarchy "ml" -> "ml/supervised"
- When: Expanding with a set {"ml/supervised"}
- Then: Returns correct expanded dictionary (tests set input handling)

### Ontology.all_tags

**Test: Get all tags from multi-level hierarchy**
- Given: An ontology with "ml" -> ["ml/supervised"] -> ["ml/supervised/regression"]
- When: Getting all tags
- Then: Returns set containing all three tags

**Test: Get all tags from empty ontology**
- Given: An empty ontology
- When: Getting all tags
- Then: Returns empty set

**Test: Get all tags includes both parents and leaf children**
- Given: An ontology with parents in adjacency and leaves only in parent_map
- When: Getting all tags
- Then: Returns set containing both parent and leaf tags

### load_ontology

**Test: Load ontology from valid JSON file with wrapped format**
- Given: A JSON file with {"tags": {...}, "parent_weight": 0.8}
- When: Loading the ontology
- Then: Returns Ontology with correct adjacency and parent_weight=0.8

**Test: Load ontology from valid JSON file with flat format**
- Given: A JSON file with flat tag structure (no "tags" wrapper)
- When: Loading the ontology
- Then: Returns Ontology with correct adjacency and default parent_weight=0.7

**Test: Load ontology with Path object**
- Given: A Path object pointing to valid JSON file
- When: Loading the ontology
- Then: Returns correct Ontology instance

**Test: Load ontology with string path**
- Given: A string path to valid JSON file
- When: Loading the ontology
- Then: Returns correct Ontology instance

**Test: Load ontology from non-existent file**
- Given: A path to a file that doesn't exist
- When: Loading the ontology
- Then: Raises FileNotFoundError

**Test: Load ontology from invalid JSON file**
- Given: A file with malformed JSON
- When: Loading the ontology
- Then: Raises json.JSONDecodeError

### load_default_ontology

**Test: Load default ontology when file exists**
- Given: A tag_ontology.json file exists in metadata/data/
- When: Loading the default ontology
- Then: Returns Ontology loaded from that file

**Test: Load default ontology when file doesn't exist**
- Given: No tag_ontology.json file in metadata/data/
- When: Loading the default ontology
- Then: Returns Ontology with fallback built-in hierarchy (python, machine-learning)

**Test: Default ontology fallback has correct structure**
- Given: No ontology file exists
- When: Loading the default ontology and checking structure
- Then: The fallback ontology has "python" and "machine-learning" parents with proper children

**Test: Default ontology fallback has descriptions**
- Given: No ontology file exists
- When: Loading the default ontology and getting descriptions
- Then: Parent tags have descriptions
