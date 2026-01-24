**Problem**: ObsidianReader has some gaps and we cannot use it to ingest local Obsidian vaults
**Decision**: We will create a subclass (thin wrapper) of SimpleDirectoryReader that embodies the below behaviors of ObsidianReader.

## 1) Vault traversal rules (directory walking semantics)

### 1.1 Recursive walk / traversal mechanism

**ObsidianReader behavior:** walks the vault using `os.walk(..., followlinks=False)` (explicit) ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** discovers files via `self.fs.walk(..., topdown=True, maxdepth=depth)` with `depth = 1000 if recursive else 1` (so recursion is supported, but via fsspec) ([LlamaIndex][2])

**Decision:**
Do nothing

### 1.2 Skip hidden directories

**ObsidianReader behavior:** filters `dirnames[:] = [d for d in dirnames if not d.startswith(".")]` during walk ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** has `exclude_hidden=True` by default, and its `is_hidden()` checks *any path part* starting with `"."` (not just immediate directories); hidden files/paths are skipped during `_add_files` ([LlamaIndex][2])

**Decision:**
If the client provides `exclude_hidden=False`, raise a Not Implemented exception



### 1.3 Markdown-only surface area

**ObsidianReader behavior:** processes only filenames ending with `.md` ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** reads “any files it finds” unless constrained; supports a `required_exts` filter (suffix whitelist). ([LlamaIndex][2])

**Decision:**
Just set `required_exts=[".md"]` or enforce `.md` in your subclass. 

---

## 2) Safety checks (hardlink + path containment)

### 2.1 Hardlink detection + skip

**ObsidianReader behavior:** calls `is_hardlink(...)`, prints warning, skips file ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** no equivalent hardlink check in `_add_files` / `load_file`—it just walks and opens files, applying hidden/empty/ext/exclude rules. ([LlamaIndex][2])

**Decision:**

* Need per-file `stat`/inode link-count checks (platform nuance) and a policy decision for fsspec backends (some won’t expose link metadata cleanly).
* Limit to local vaults for now, raise Not Implemented exception if a non-local source is used.

### 2.2 Resolved path containment guard (skip “outside vault”)

**ObsidianReader behavior:** resolves each file path and ensures it starts with resolved vault root; otherwise skip with warning ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** does not do a “resolved path is under root” guard; it relies on the walk results + exclude filters. With `input_files` you can even pass arbitrary files explicitly. ([LlamaIndex][2])

**Decision:**

* For a local vault, simple `Path.resolve()` + prefix/`relative_to` checks are straightforward (like ObsidianReader). ([LlamaIndex][1])
* Limit to local vaults for now, raise Not Implemented exception if a non-local source is used.


---

## 3) Markdown loading strategy (delegating to MarkdownReader)

### 3.1 Always use `MarkdownReader` for `.md`

**ObsidianReader behavior:** explicitly `MarkdownReader().load_data(Path(filepath))` ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** picks a reader by suffix if the suffix is supported or present in `file_extractor`; otherwise falls back to “read as text.” ([LlamaIndex][2])

**Decision:**

* Use a MarkdownReader by default, our subclass should not allow it to be overridden.
* Ensure `.md` maps to the same reader you want by injecting/overriding `file_extractor[".md"]` (or equivalent) so behavior matches. ([LlamaIndex][2])

### 3.2 Multiple Documents per file

**ObsidianReader behavior:** iterates `for i, doc in enumerate(md_docs)` and `docs.extend(md_docs)` ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** supports multiple docs from a reader too: it appends `docs` returned from `reader.load_data(...)`. It also supports `filename_as_id` and will assign `"{input_file}_part_{i}"` IDs across returned docs. ([LlamaIndex][2])

**Decision:**

* You already get the “one file → many Documents” behavior as long as the `.md` reader returns many. ([LlamaIndex][2])

---

## 4) Per-note metadata enrichment (Obsidian-style fields)

### 4.1 Add `file_name`, `folder_path`, `folder_name`, `note_name`

**ObsidianReader behavior:** sets those four metadata keys per returned doc; `folder_name` is `parent.relative_to(vault_root)` with fallback. ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** supports a `file_metadata` callback that receives filename and returns metadata; also has default metadata via `_DefaultFileMetadataFunc` and `get_resource_info` (file_path, size, timestamps). ([LlamaIndex][2])

**Decision:**

* Implement via `file_metadata` (or a post-load metadata hook) using the vault root.
* Ensure `folder_name` is *relative-to-root* consistently for all docs, including edge cases (same fallback as ObsidianReader). ([LlamaIndex][1])

**One “gotcha” to be aware of:** SimpleDirectoryReader also maintains “excluded metadata keys” lists (e.g., it excludes file_name, file_size, timestamps from embedding/LLM metadata by default). Our metadata must be available downstream, and so we will adjust those exclusions. ([LlamaIndex][2])

---

## 5) Wikilink extraction (Obsidian semantics)

### 5.1 Extract wikilinks + alias stripping + unique targets

**ObsidianReader behavior:** regex `\[\[([^\]]+)\]\]`, split on `|` and keep left side, `list(set(...))` for uniqueness; stored as `metadata["wikilinks"]`. ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** no built-in concept of wikilinks; it just loads file text + metadata. ([LlamaIndex][2])

**Decision:**

* Pure text post-processing (per-Document) after loading. Must match the ObsidianReader implementation

---

## 6) Backlinks graph construction (vault-global, two-pass)

### 6.1 Build backlinks map and annotate all docs

**ObsidianReader behavior:**

* During per-note processing: for each wikilink, `backlinks_map.setdefault(link, []).append(note_name)` ([LlamaIndex][1])
* After all files: sets `doc.metadata["backlinks"] = backlinks_map.get(note_name, [])` ([LlamaIndex][1])

**SimpleDirectoryReader behavior:** no graph/global pass; it loads each file independently. ([LlamaIndex][2])

**Decision:**

* vault-global accumulator + finalization pass.
* Do not implement streaming (`iter_data`) semantics, because backlinks are “reverse edges” that can’t be finalized until all forward links are known. ([LlamaIndex][1])

---

## 7) Optional task extraction (and optional removal from text)

### 7.1 Extract tasks to metadata

**ObsidianReader behavior:** regex for task lines, stores list in `metadata["tasks"]`. ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** none built-in. ([LlamaIndex][2])

**Decision:**

* Straight per-document regex pass.

### 7.2 Remove task lines from text (optional)

**ObsidianReader behavior:** if enabled, rewrites the `Document` text to `cleaned_text` by creating a new `Document(...)` for that index in `md_docs`. ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** no built-in transforms, but you can post-process returned `Document.text`. ([LlamaIndex][2])

**Decision:**

* Match implementation; ObsidianReader removes tasks after loading docs (and after any splitting done inside `MarkdownReader`). ([LlamaIndex][1])

---

## 8) Error handling policy (best-effort vs strict)

### 8.1 “Continue on file error” semantics

**ObsidianReader behavior:** wraps per-file processing in `try/except Exception`, prints, continues. ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** `load_file` catches exceptions; if `raise_on_error` is False it prints “Failed to load… Skipping…” and returns `[]`; if True it raises. ([LlamaIndex][2])

**Decision:**

* Match implementation of ObsidianReader
---

## 9) Output format convenience (LangChain conversion)

### 9.1 `load_langchain_documents()`

**ObsidianReader behavior:** wrapper that calls `load_data()` then `to_langchain_format()` for each doc. ([LlamaIndex][1])
**SimpleDirectoryReader behavior:** doesn’t define this wrapper in the API reference shown; it provides `load_data`, `aload_data`, and `iter_data`. ([LlamaIndex][2])

**Decision:**

* Simple utility method in your subclass.

[1]: https://developers.llamaindex.ai/python/framework-api-reference/readers/obsidian/ "Obsidian - LlamaIndex"
[2]: https://developers.llamaindex.ai/python/framework-api-reference/readers/simple_directory_reader/ "Simple directory reader - LlamaIndex"

---

# Technical requirements

1. New abstraction: llama_index.readers.obsidian.SimpleObsidianReader`
2. All code for `SimpleObsidianReader` should exist in `simple.py`
3. All tests for `ObsidianReader` should be replicated and must pass