
## 6. Data Flow Diagrams

This section illustrates how data flows between Python classes for key operations.

### 6.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      cli/main.py                                     │    │
│  │                    create_parser()                                   │    │
│  │                        main()                                        │    │
│  └───────────────────────────┬─────────────────────────────────────────┘    │
│                              │                                               │
│  ┌───────────┬───────────┬───┴────┬───────────┬───────────┬──────────┐     │
│  │ search.py │collection │document│ context.py│  index.py │status.py │     │
│  │           │    .py    │  .py   │           │           │          │     │
│  └─────┬─────┴─────┬─────┴────┬───┴─────┬─────┴─────┬─────┴────┬─────┘     │
└────────┼───────────┼──────────┼─────────┼───────────┼──────────┼────────────┘
         │           │          │         │           │          │
         ▼           ▼          ▼         ▼           ▼          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core Services                                      │
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐    │
│  │  search/           │  │  llm/              │  │  formatters/       │    │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │  │  ┌──────────────┐  │    │
│  │  │ pipeline.py  │  │  │  │  base.py     │  │  │  │  base.py     │  │    │
│  │  │ Hybrid       │  │  │  │  LLMProvider │  │  │  │  Formatter   │  │    │
│  │  │ SearchPipe   │◄─┼──┼──┤  (ABC)       │  │  │  │  (ABC)       │  │    │
│  │  └──────────────┘  │  │  └──────┬───────┘  │  │  └──────────────┘  │    │
│  │  ┌──────────────┐  │  │         │          │  │  ┌──────────────┐  │    │
│  │  │ fusion.py    │  │  │  ┌──────▼───────┐  │  │  │  json.py     │  │    │
│  │  │ RRF          │  │  │  │  ollama.py   │  │  │  │  csv.py      │  │    │
│  │  └──────────────┘  │  │  │  Ollama      │  │  │  │  xml.py      │  │    │
│  │  ┌──────────────┐  │  │  │  Provider    │  │  │  │  markdown.py │  │    │
│  │  │ scoring.py   │  │  │  └──────────────┘  │  │  └──────────────┘  │    │
│  │  │ blend_scores │  │  │                    │  │                    │    │
│  │  └──────────────┘  │  │  ┌──────────────┐  │  │                    │    │
│  │  ┌──────────────┐  │  │  │query_expan   │  │  │                    │    │
│  │  │ chunking.py  │  │  │  │  sion.py     │  │  │                    │    │
│  │  └──────────────┘  │  │  └──────────────┘  │  │                    │    │
│  └────────┬───────────┘  └─────────┬──────────┘  └────────────────────┘    │
│           │                        │                                        │
└───────────┼────────────────────────┼────────────────────────────────────────┘
            │                        │
            ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Store Layer                                       │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │  database.py   │  │  search.py     │  │  documents.py  │                 │
│  │  Database      │◄─┤  Search        │  │  Document      │                 │
│  │  (connection)  │  │  Repository    │  │  Repository    │                 │
│  └───────┬────────┘  └────────────────┘  └────────────────┘                 │
│          │           ┌────────────────┐  ┌────────────────┐                 │
│          │           │  collections   │  │  embeddings.py │                 │
│          │           │  .py           │  │  Embedding     │                 │
│          │           │  Collection    │  │  Repository    │                 │
│          │           │  Repository    │  └────────────────┘                 │
│          │           └────────────────┘  ┌────────────────┐                 │
│          │           ┌────────────────┐  │  contexts.py   │                 │
│          │           │  cache.py      │  │  Context       │                 │
│          │           │  Cache         │  │  Repository    │                 │
│          │           │  Repository    │  └────────────────┘                 │
│          │           └────────────────┘                                     │
│          ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │                    SQLite Database                         │              │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌────────────────┐  │              │
│  │  │ content │ │documents │ │collections│ │ documents_fts │  │              │
│  │  └─────────┘ └──────────┘ └─────────┘ └────────────────┘  │              │
│  │  ┌─────────────────┐ ┌────────────┐ ┌──────────────────┐  │              │
│  │  │ content_vectors │ │ vectors_vec│ │   ollama_cache   │  │              │
│  │  └─────────────────┘ └────────────┘ └──────────────────┘  │              │
│  └───────────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            │ HTTP
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         External Services                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Ollama (localhost:11434)                       │   │
│  │   /api/embed   /api/generate   /api/chat   /api/tags   /api/pull     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Collection Add & Document Indexing Flow

This diagram shows what happens when a user runs `pmd collection add /path/to/notes --name mynotes`.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ User: pmd collection add /path/to/notes --name mynotes --mask "**/*.md"      │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        cli/commands/collection.py                            │
│                                                                              │
│  def handle_add(args: Namespace, config: Config):                           │
│      name = args.name or Path(args.path).name                               │
│      path = Path(args.path).resolve()                                       │
│      glob_pattern = args.mask or "**/*.md"                                  │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────────┐    │
│      │ IndexService(db, config)                                         │    │
│      │     .add_collection(name, path, glob_pattern)                    │    │
│      └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           IndexService                                        │
│                    (orchestrates indexing workflow)                          │
│                                                                              │
│  class IndexService:                                                         │
│      def __init__(self, db: Database, config: Config):                      │
│          self.collections = CollectionRepository(db)                        │
│          self.documents = DocumentRepository(db)                            │
│          self.file_scanner = FileScanner()                                  │
│                                                                              │
│      async def add_collection(self, name, path, glob_pattern):              │
│                                                                              │
│          # Step 1: Create collection record                                 │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ collection = self.collections.create(                       │     │
│          │     name=name,                                              │     │
│          │     pwd=str(path),                                          │     │
│          │     glob_pattern=glob_pattern                               │     │
│          │ )                                                           │     │
│          └────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│          # Step 2: Scan filesystem for matching files                       │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ files = self.file_scanner.scan(path, glob_pattern)          │     │
│          │                                                             │     │
│          │ FileScanner.scan():                                         │     │
│          │   for filepath in Path(path).glob(glob_pattern):            │     │
│          │       yield FileInfo(                                       │     │
│          │           path=filepath,                                    │     │
│          │           modified_at=filepath.stat().st_mtime              │     │
│          │       )                                                     │     │
│          └────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│          # Step 3: Index each file                                          │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ for file_info in files:                                     │     │
│          │     await self._index_file(collection.id, file_info)        │     │
│          └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      IndexService._index_file()                              │
│                                                                              │
│  async def _index_file(self, collection_id: int, file_info: FileInfo):      │
│                                                                              │
│      # Step 3a: Read file content                                           │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ content = file_info.path.read_text(encoding='utf-8')            │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│      # Step 3b: Compute content hash (SHA256)                               │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ content_hash = utils.hashing.hash_content(content)              │     │
│      │                                                                 │     │
│      │ def hash_content(content: str) -> str:                          │     │
│      │     return hashlib.sha256(                                      │     │
│      │         content.encode('utf-8')                                 │     │
│      │     ).hexdigest()                                               │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│      # Step 3c: Extract title from content                                  │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ title = utils.text.extract_title(content, file_info.path.name)  │     │
│      │                                                                 │     │
│      │ def extract_title(content: str, filename: str) -> str:          │     │
│      │     # Try first H1 heading: # Title                             │     │
│      │     match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)     │     │
│      │     if match:                                                   │     │
│      │         return match.group(1).strip()                           │     │
│      │     # Fallback to filename without extension                    │     │
│      │     return Path(filename).stem                                  │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│      # Step 3d: Store content (content-addressable)                         │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ self.documents.store_content(content_hash, content)             │     │
│      │                                                                 │     │
│      │ DocumentRepository.store_content():                             │     │
│      │   INSERT OR IGNORE INTO content (hash, doc, created_at)         │     │
│      │   VALUES (?, ?, datetime('now'))                                │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│      # Step 3e: Create document record                                      │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ relative_path = file_info.path.relative_to(collection.pwd)      │     │
│      │                                                                 │     │
│      │ self.documents.upsert(                                          │     │
│      │     collection_id=collection_id,                                │     │
│      │     path=str(relative_path),                                    │     │
│      │     title=title,                                                │     │
│      │     hash=content_hash,                                          │     │
│      │     modified_at=file_info.modified_at                           │     │
│      │ )                                                               │     │
│      │                                                                 │     │
│      │ DocumentRepository.upsert():                                    │     │
│      │   INSERT INTO documents (...) VALUES (...)                      │     │
│      │   ON CONFLICT(collection_id, path) DO UPDATE SET                │     │
│      │       hash=excluded.hash, title=excluded.title, ...             │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│      # Step 3f: Update FTS index                                            │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ self.documents.update_fts(doc_id, relative_path, content)       │     │
│      │                                                                 │     │
│      │ DocumentRepository.update_fts():                                │     │
│      │   DELETE FROM documents_fts WHERE rowid = ?                     │     │
│      │   INSERT INTO documents_fts (rowid, path, body)                 │     │
│      │   VALUES (?, ?, ?)                                              │     │
│      └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Database State After Indexing                      │
│                                                                              │
│  collections:                                                                │
│  ┌────┬──────────┬──────────────────┬────────────┐                          │
│  │ id │   name   │       pwd        │ glob_pat   │                          │
│  ├────┼──────────┼──────────────────┼────────────┤                          │
│  │  1 │ mynotes  │ /path/to/notes   │ **/*.md    │                          │
│  └────┴──────────┴──────────────────┴────────────┘                          │
│                                                                              │
│  content:                                                                    │
│  ┌──────────────────────────────────────┬─────────────────────┐             │
│  │                 hash                  │         doc         │             │
│  ├──────────────────────────────────────┼─────────────────────┤             │
│  │ a1b2c3d4e5f6...                       │ # Meeting Notes...  │             │
│  │ f6e5d4c3b2a1...                       │ # Project Plan...   │             │
│  └──────────────────────────────────────┴─────────────────────┘             │
│                                                                              │
│  documents:                                                                  │
│  ┌────┬───────────────┬──────────────────┬───────────────────────┐          │
│  │ id │ collection_id │      path        │         hash          │          │
│  ├────┼───────────────┼──────────────────┼───────────────────────┤          │
│  │  1 │       1       │ meetings/jan.md  │ a1b2c3d4e5f6...       │          │
│  │  2 │       1       │ projects/plan.md │ f6e5d4c3b2a1...       │          │
│  └────┴───────────────┴──────────────────┴───────────────────────┘          │
│                                                                              │
│  documents_fts (FTS5 virtual table):                                        │
│  ┌───────┬──────────────────┬─────────────────────────────┐                 │
│  │ rowid │      path        │            body             │                 │
│  ├───────┼──────────────────┼─────────────────────────────┤                 │
│  │   1   │ meetings/jan.md  │ # Meeting Notes\n\nTopics...│                 │
│  │   2   │ projects/plan.md │ # Project Plan\n\nPhase 1...│                 │
│  └───────┴──────────────────┴─────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Embedding Generation Flow

This diagram shows what happens when a user runs `pmd embed` to generate vector embeddings.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ User: pmd embed                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         cli/commands/index.py                                │
│                                                                              │
│  async def handle_embed(args: Namespace, config: Config):                   │
│      db = Database(config.db_path)                                          │
│      db.connect()                                                            │
│                                                                              │
│      embedding_service = EmbeddingService(                                  │
│          db=db,                                                              │
│          llm=OllamaProvider(config.ollama.base_url),                        │
│          config=config                                                       │
│      )                                                                       │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────────┐    │
│      │ await embedding_service.embed_all(force=args.force)              │    │
│      └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EmbeddingService                                    │
│                                                                              │
│  class EmbeddingService:                                                     │
│      def __init__(self, db, llm, config):                                   │
│          self.embeddings = EmbeddingRepository(db)                          │
│          self.documents = DocumentRepository(db)                            │
│          self.llm = llm                                                      │
│          self.chunker = DocumentChunker(config.chunk)                       │
│          self.model = config.ollama.embedding_model                         │
│                                                                              │
│      async def embed_all(self, force: bool = False):                        │
│                                                                              │
│          # Step 1: Get documents needing embeddings                         │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ if force:                                                   │     │
│          │     docs = self.documents.get_all_with_content()            │     │
│          │ else:                                                       │     │
│          │     docs = self.documents.get_unembedded(self.model)        │     │
│          │                                                             │     │
│          │ DocumentRepository.get_unembedded():                        │     │
│          │   SELECT d.*, c.doc FROM documents d                        │     │
│          │   JOIN content c ON d.hash = c.hash                         │     │
│          │   WHERE d.hash NOT IN (                                     │     │
│          │       SELECT hash FROM content_vectors WHERE model = ?      │     │
│          │   )                                                         │     │
│          └────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│          # Step 2: Process each document                                    │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ for doc in docs:                                            │     │
│          │     await self._embed_document(doc)                         │     │
│          └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    EmbeddingService._embed_document()                        │
│                                                                              │
│  async def _embed_document(self, doc: DocumentWithContent):                 │
│                                                                              │
│      # Step 2a: Chunk the document                                          │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ chunks = self.chunker.chunk(doc.body)                           │     │
│      │                                                                 │     │
│      │ class DocumentChunker:                                          │     │
│      │     def chunk(self, content: str) -> list[Chunk]:               │     │
│      │         if len(content.encode()) <= self.max_bytes:             │     │
│      │             return [Chunk(text=content, pos=0)]                 │     │
│      │                                                                 │     │
│      │         chunks = []                                             │     │
│      │         pos = 0                                                 │     │
│      │         while pos < len(content):                               │     │
│      │             # Find best split point                             │     │
│      │             end = self._find_split_point(content, pos)          │     │
│      │             chunks.append(Chunk(                                │     │
│      │                 text=content[pos:end],                          │     │
│      │                 pos=pos                                         │     │
│      │             ))                                                  │     │
│      │             pos = end                                           │     │
│      │         return chunks                                           │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│      # Step 2b: Generate embedding for each chunk                           │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ for seq, chunk in enumerate(chunks):                            │     │
│      │                                                                 │     │
│      │     # Format text for embedding model                           │     │
│      │     formatted = f"title: {doc.title} | text: {chunk.text}"      │     │
│      │                                                                 │     │
│      │     # Call Ollama API                                           │     │
│      │     result = await self.llm.embed(                              │     │
│      │         text=formatted,                                         │     │
│      │         model=self.model,                                       │     │
│      │         is_query=False                                          │     │
│      │     )                                                           │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
┌──────────────────────────────────────────────────────────────────────────────┐
│                         OllamaProvider.embed()                               │
│                                                                              │
│  async def embed(self, text, model, is_query=False) -> EmbeddingResult:     │
│                                                                              │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ response = await self._client.post(                             │     │
│      │     f"{self.base_url}/api/embed",                               │     │
│      │     json={                                                      │     │
│      │         "model": model,          # "embeddinggemma"             │     │
│      │         "input": text            # formatted text               │     │
│      │     }                                                           │     │
│      │ )                                                               │     │
│      │                                                                 │     │
│      │ return EmbeddingResult(                                         │     │
│      │     embedding=response.json()["embeddings"][0],  # float[768]   │     │
│      │     model=model                                                 │     │
│      │ )                                                               │     │
│      └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    EmbeddingService._embed_document() (continued)            │
│                                                                              │
│      # Step 2c: Store embedding in database                                 │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ self.embeddings.store(                                          │     │
│      │     hash=doc.hash,                                              │     │
│      │     seq=seq,                                                    │     │
│      │     pos=chunk.pos,                                              │     │
│      │     model=self.model,                                           │     │
│      │     embedding=result.embedding                                  │     │
│      │ )                                                               │     │
│      │                                                                 │     │
│      │ class EmbeddingRepository:                                      │     │
│      │     def store(self, hash, seq, pos, model, embedding):          │     │
│      │         # Store metadata                                        │     │
│      │         INSERT INTO content_vectors                             │     │
│      │             (hash, seq, pos, model, embedded_at)                │     │
│      │         VALUES (?, ?, ?, ?, datetime('now'))                    │     │
│      │                                                                 │     │
│      │         # Store vector (sqlite-vec)                             │     │
│      │         INSERT INTO vectors_vec (hash_seq, embedding)           │     │
│      │         VALUES (?, ?)  # hash_seq = f"{hash}:{seq}"             │     │
│      └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       Database State After Embedding                         │
│                                                                              │
│  content_vectors:                                                            │
│  ┌──────────────────────┬─────┬──────┬───────────────┬─────────────────┐    │
│  │         hash         │ seq │ pos  │     model     │   embedded_at   │    │
│  ├──────────────────────┼─────┼──────┼───────────────┼─────────────────┤    │
│  │ a1b2c3d4e5f6...      │  0  │   0  │ embeddinggemma│ 2025-01-15 10:30│    │
│  │ a1b2c3d4e5f6...      │  1  │ 6144 │ embeddinggemma│ 2025-01-15 10:30│    │
│  │ f6e5d4c3b2a1...      │  0  │   0  │ embeddinggemma│ 2025-01-15 10:31│    │
│  └──────────────────────┴─────┴──────┴───────────────┴─────────────────┘    │
│                                                                              │
│  vectors_vec (sqlite-vec virtual table):                                    │
│  ┌─────────────────────────────┬────────────────────────────────────────┐   │
│  │          hash_seq           │           embedding (float[768])       │   │
│  ├─────────────────────────────┼────────────────────────────────────────┤   │
│  │ a1b2c3d4e5f6...:0           │ [0.0234, -0.1456, 0.0891, ...]         │   │
│  │ a1b2c3d4e5f6...:1           │ [0.0567, -0.0234, 0.1234, ...]         │   │
│  │ f6e5d4c3b2a1...:0           │ [-0.0123, 0.0456, 0.0789, ...]         │   │
│  └─────────────────────────────┴────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Hybrid Search Flow

This diagram shows the complete hybrid search pipeline when a user runs `pmd query "authentication flow"`.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ User: pmd query "authentication flow" -n 5 --collection mynotes              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         cli/commands/search.py                               │
│                                                                              │
│  async def handle_query(args, config):                                      │
│      db = Database(config.db_path)                                          │
│      llm = OllamaProvider(config.ollama.base_url)                           │
│                                                                              │
│      pipeline = HybridSearchPipeline(                                       │
│          search_repo=SearchRepository(db),                                  │
│          llm_provider=llm,                                                   │
│          config=SearchPipelineConfig()                                      │
│      )                                                                       │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────────┐    │
│      │ results = await pipeline.search(                                 │    │
│      │     query="authentication flow",                                 │    │
│      │     limit=5,                                                     │    │
│      │     collection_id=resolve_collection(args.collection)            │    │
│      │ )                                                                │    │
│      └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    HybridSearchPipeline.search()                             │
│                                                                              │
│  async def search(self, query, limit, collection_id, min_score):            │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  STEP 1: Query Expansion                                               ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ expander = QueryExpander(self.llm, "qwen3:0.6b")                │     │
│      │ variations = await expander.expand(query, num_variations=2)     │     │
│      │                                                                 │     │
│      │ # Input:  "authentication flow"                                 │     │
│      │ # Output: ["user login process", "auth workflow steps"]         │     │
│      │                                                                 │     │
│      │ queries = [query] + variations                                  │     │
│      │ # ["authentication flow", "user login process", "auth workflow"]│     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  STEP 2: Parallel FTS + Vector Search                                  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────────┐    │
│      │                    asyncio.gather()                              │    │
│      │                                                                  │    │
│      │  ┌─────────────────────────────────────────────────────────┐    │    │
│      │  │  Query 1: "authentication flow" (original, weight=2.0)  │    │    │
│      │  │  ├─→ search_fts("authentication flow")                  │    │    │
│      │  │  └─→ search_vec("authentication flow")                  │    │    │
│      │  └─────────────────────────────────────────────────────────┘    │    │
│      │  ┌─────────────────────────────────────────────────────────┐    │    │
│      │  │  Query 2: "user login process" (expanded, weight=1.0)   │    │    │
│      │  │  ├─→ search_fts("user login process")                   │    │    │
│      │  │  └─→ search_vec("user login process")                   │    │    │
│      │  └─────────────────────────────────────────────────────────┘    │    │
│      │  ┌─────────────────────────────────────────────────────────┐    │    │
│      │  │  Query 3: "auth workflow steps" (expanded, weight=1.0)  │    │    │
│      │  │  ├─→ search_fts("auth workflow steps")                  │    │    │
│      │  │  └─→ search_vec("auth workflow steps")                  │    │    │
│      │  └─────────────────────────────────────────────────────────┘    │    │
│      └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                          │                           │
            ┌─────────────┘                           └─────────────┐
            ▼                                                       ▼
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│   SearchRepository.search_fts()   │       │   SearchRepository.search_vec()   │
│                                   │       │                                   │
│ def search_fts(query, limit,      │       │ async def search_vec(query,       │
│                collection_id):    │       │          limit, collection_id):   │
│                                   │       │                                   │
│   # Build FTS5 query              │       │   # Get query embedding           │
│   fts_query = build_fts5_query(   │       │   ┌───────────────────────────┐   │
│       query                       │       │   │ embedding = await          │   │
│   )                               │       │   │   self.llm.embed(          │   │
│   # "authenticat"* AND "flow"*    │       │   │     query,                 │   │
│                                   │       │   │     model,                 │   │
│   # Execute BM25 search           │       │   │     is_query=True          │   │
│   ┌───────────────────────────┐   │       │   │   )                        │   │
│   │ SELECT d.*, bm25(fts) as  │   │       │   └───────────────────────────┘   │
│   │   score                   │   │       │                                   │
│   │ FROM documents_fts fts    │   │       │   # KNN vector search             │
│   │ JOIN documents d ON ...   │   │       │   ┌───────────────────────────┐   │
│   │ WHERE fts MATCH ?         │   │       │   │ SELECT v.hash_seq,        │   │
│   │ ORDER BY score            │   │       │   │   vec_distance_cosine(    │   │
│   │ LIMIT ?                   │   │       │   │     v.embedding, ?        │   │
│   └───────────────────────────┘   │       │   │   ) as distance           │   │
│                                   │       │   │ FROM vectors_vec v        │   │
│   # Normalize scores              │       │   │ ORDER BY distance         │   │
│   max_score = results[0].score    │       │   │ LIMIT ?                   │   │
│   for r in results:               │       │   └───────────────────────────┘   │
│       r.score = abs(r.score)      │       │                                   │
│                  / max_score      │       │   # Convert distance to score     │
│       r.source = SearchSource.FTS │       │   for r in results:               │
│                                   │       │       r.score = 1/(1+r.distance)  │
│   return results                  │       │       r.source = SearchSource.VEC │
│                                   │       │                                   │
└───────────────────────────────────┘       └───────────────────────────────────┘
                          │                           │
                          └─────────────┬─────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    HybridSearchPipeline.search() (continued)                 │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  STEP 3: Reciprocal Rank Fusion                                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ # all_results = [                                               │     │
│      │ #   fts_q1, vec_q1,    # original query (weight=2.0)            │     │
│      │ #   fts_q2, vec_q2,    # expansion 1 (weight=1.0)               │     │
│      │ #   fts_q3, vec_q3     # expansion 2 (weight=1.0)               │     │
│      │ # ]                                                             │     │
│      │                                                                 │     │
│      │ fused = reciprocal_rank_fusion(                                 │     │
│      │     all_results,                                                │     │
│      │     k=60,                                                       │     │
│      │     weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0]                      │     │
│      │ )                                                               │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │            fusion.reciprocal_rank_fusion()                      │     │
│      │                                                                 │     │
│      │  scores = defaultdict(float)                                    │     │
│      │                                                                 │     │
│      │  for list_idx, results in enumerate(result_lists):              │     │
│      │      weight = weights[list_idx]                                 │     │
│      │                                                                 │     │
│      │      for rank, result in enumerate(results):                    │     │
│      │          # RRF formula                                          │     │
│      │          rrf = weight / (k + rank + 1)                          │     │
│      │                                                                 │     │
│      │          # Top-rank bonus                                       │     │
│      │          if rank == 0:                                          │     │
│      │              rrf += 0.05                                        │     │
│      │          elif rank <= 2:                                        │     │
│      │              rrf += 0.02                                        │     │
│      │                                                                 │     │
│      │          scores[result.filepath] += rrf                         │     │
│      │                                                                 │     │
│      │  # Sort by fused score, take top 30                             │     │
│      │  return sorted(scores.items(), key=lambda x: -x[1])[:30]        │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  STEP 4: LLM Reranking                                                 ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ reranked = await self.llm.rerank(                               │     │
│      │     query="authentication flow",                                │     │
│      │     documents=candidates[:30],  # top 30 from RRF               │     │
│      │     model="qwen3-reranker"                                      │     │
│      │ )                                                               │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │              OllamaProvider.rerank()                            │     │
│      │                                                                 │     │
│      │  for doc in documents:                                          │     │
│      │      response = await self._client.post(                        │     │
│      │          "/api/chat",                                           │     │
│      │          json={                                                 │     │
│      │              "model": "qwen3-reranker",                         │     │
│      │              "messages": [                                      │     │
│      │                  {"role": "system", "content": SYSTEM_PROMPT},  │     │
│      │                  {"role": "user", "content":                    │     │
│      │                      f"Query: {query}\nDocument: {doc.body}"}   │     │
│      │              ],                                                 │     │
│      │              "options": {"logprobs": True, "num_predict": 1}    │     │
│      │          }                                                      │     │
│      │      )                                                          │     │
│      │                                                                 │     │
│      │      # Parse Yes/No with confidence from logprobs               │     │
│      │      token = response["message"]["content"]  # "Yes" or "No"    │     │
│      │      logprob = response["logprobs"][0]       # e.g., -0.1       │     │
│      │      confidence = exp(logprob)               # e.g., 0.90       │     │
│      │                                                                 │     │
│      │      if token == "Yes":                                         │     │
│      │          score = 0.5 + 0.5 * confidence  # 0.5-1.0              │     │
│      │      else:                                                      │     │
│      │          score = 0.5 * (1 - confidence)  # 0.0-0.5              │     │
│      └────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  STEP 5: Position-Aware Score Blending                                 ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ final = scoring.blend_scores(fused, reranked)                   │     │
│      │                                                                 │     │
│      │ def blend_scores(rrf_results, rerank_results):                  │     │
│      │     for rank, result in enumerate(rrf_results):                 │     │
│      │         rerank_score = rerank_map[result.file]                  │     │
│      │                                                                 │     │
│      │         # Position-aware weighting                              │     │
│      │         if rank < 3:                                            │     │
│      │             rrf_weight = 0.75  # Trust initial ranking          │     │
│      │         elif rank < 10:                                         │     │
│      │             rrf_weight = 0.60                                   │     │
│      │         else:                                                   │     │
│      │             rrf_weight = 0.40  # Trust reranker more            │     │
│      │                                                                 │     │
│      │         final_score = (                                         │     │
│      │             rrf_weight * rrf_score +                            │     │
│      │             (1 - rrf_weight) * rerank_score                     │     │
│      │         )                                                       │     │
│      │                                                                 │     │
│      │     # Re-sort by blended score                                  │     │
│      │     return sorted(results, key=lambda r: -r.score)[:limit]      │     │
│      └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    cli/commands/search.py (continued)                        │
│                                                                              │
│      # Format and display results                                           │
│      ┌────────────────────────────────────────────────────────────────┐     │
│      │ formatter = get_formatter(args.format)  # JSONFormatter, etc.   │     │
│      │ output = formatter.format_many(results)                         │     │
│      │ print(output)                                                   │     │
│      └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Final Output                                    │
│                                                                              │
│  [                                                                           │
│    {                                                                         │
│      "score": 0.847,                                                         │
│      "file": "pmd://mynotes/auth/login-flow.md",                            │
│      "title": "User Authentication Flow",                                   │
│      "context": "Authentication documentation",                             │
│      "snippet": "...the authentication flow begins when a user..."          │
│    },                                                                        │
│    {                                                                         │
│      "score": 0.723,                                                         │
│      "file": "pmd://mynotes/api/endpoints.md",                              │
│      "title": "API Endpoints",                                              │
│      "context": "API reference",                                            │
│      "snippet": "...POST /auth/login initiates the authentication..."       │
│    },                                                                        │
│    ...                                                                       │
│  ]                                                                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.5 Document Retrieval Flow

This diagram shows what happens when a user runs `pmd get` to retrieve a document.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ User: pmd get "pmd://mynotes/auth/login-flow.md" --from-line 10 -l 50       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         cli/commands/document.py                             │
│                                                                              │
│  def handle_get(args, config):                                              │
│      db = Database(config.db_path)                                          │
│      doc_service = DocumentService(db)                                      │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────────┐    │
│      │ result = doc_service.get(                                        │    │
│      │     path="pmd://mynotes/auth/login-flow.md",                     │    │
│      │     from_line=10,                                                │    │
│      │     max_lines=50                                                 │    │
│      │ )                                                                │    │
│      └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            DocumentService                                    │
│                                                                              │
│  class DocumentService:                                                      │
│      def __init__(self, db: Database):                                      │
│          self.documents = DocumentRepository(db)                            │
│          self.contexts = ContextRepository(db)                              │
│          self.virtual_paths = VirtualPathResolver(db)                       │
│                                                                              │
│      def get(self, path, from_line=None, max_lines=None):                   │
│                                                                              │
│          # Step 1: Parse and resolve virtual path                           │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ vpath = self.virtual_paths.parse(path)                      │     │
│          │                                                             │     │
│          │ class VirtualPathResolver:                                  │     │
│          │     def parse(self, path: str) -> VirtualPath | None:       │     │
│          │         if not path.startswith("pmd://"):                   │     │
│          │             return None                                     │     │
│          │         rest = path[6:]  # Remove "pmd://"                  │     │
│          │         parts = rest.split("/", 1)                          │     │
│          │         return VirtualPath(                                 │     │
│          │             collection_name=parts[0],  # "mynotes"          │     │
│          │             path=parts[1]              # "auth/login-flow"  │     │
│          │         )                                                   │     │
│          └────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│          # Step 2: Find document in database                                │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ doc = self.documents.find(vpath)                            │     │
│          │                                                             │     │
│          │ class DocumentRepository:                                   │     │
│          │     def find(self, vpath: VirtualPath) -> DocumentResult:   │     │
│          │         SELECT d.*, c.doc as body, col.name, col.pwd        │     │
│          │         FROM documents d                                    │     │
│          │         JOIN content c ON d.hash = c.hash                   │     │
│          │         JOIN collections col ON d.collection_id = col.id    │     │
│          │         WHERE col.name = ? AND d.path = ?                   │     │
│          │                                                             │     │
│          │         if not found:                                       │     │
│          │             # Try fuzzy matching                            │     │
│          │             suggestions = self._find_similar(vpath)         │     │
│          │             raise DocumentNotFoundError(path, suggestions)  │     │
│          └────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│          # Step 3: Get context for this path                                │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ context = self.contexts.get_for_path(                       │     │
│          │     collection_id=doc.collection_id,                        │     │
│          │     path=vpath.path                                         │     │
│          │ )                                                           │     │
│          │                                                             │     │
│          │ class ContextRepository:                                    │     │
│          │     def get_for_path(self, collection_id, path):            │     │
│          │         # Find longest matching prefix                      │     │
│          │         SELECT context FROM path_contexts                   │     │
│          │         WHERE collection_id = ?                             │     │
│          │           AND ? LIKE path_prefix || '%'                     │     │
│          │         ORDER BY LENGTH(path_prefix) DESC                   │     │
│          │         LIMIT 1                                             │     │
│          └────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│          # Step 4: Extract requested lines                                  │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ if from_line or max_lines:                                  │     │
│          │     body = self._extract_lines(                             │     │
│          │         doc.body,                                           │     │
│          │         from_line or 1,                                     │     │
│          │         max_lines or len(doc.body.splitlines())             │     │
│          │     )                                                       │     │
│          │ else:                                                       │     │
│          │     body = doc.body                                         │     │
│          │                                                             │     │
│          │ def _extract_lines(self, body, start, count):               │     │
│          │     lines = body.splitlines()                               │     │
│          │     return '\n'.join(                                       │     │
│          │         lines[start-1 : start-1+count]                      │     │
│          │     )                                                       │     │
│          └────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│          # Step 5: Build result                                             │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ return DocumentResult(                                      │     │
│          │     filepath=str(Path(doc.pwd) / vpath.path),               │     │
│          │     display_path=str(vpath),                                │     │
│          │     title=doc.title,                                        │     │
│          │     context=context,                                        │     │
│          │     hash=doc.hash,                                          │     │
│          │     collection_id=doc.collection_id,                        │     │
│          │     modified_at=doc.modified_at,                            │     │
│          │     body_length=len(doc.body),                              │     │
│          │     body=body                                               │     │
│          │ )                                                           │     │
│          └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.6 MCP Server Request Flow

This diagram shows how MCP requests flow through the system.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AI Agent (Claude, etc.)                              │
│                                                                              │
│  Tool call: pmd.query(query="authentication", limit=3)                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ stdio (JSON-RPC)
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              mcp/server.py                                   │
│                                                                              │
│  class PMDMCPServer:                                                        │
│      def __init__(self, db_path):                                           │
│          self.server = Server("pmd")                                        │
│          self.db = Database(db_path)                                        │
│                                                                              │
│          # Register handlers                                                │
│          register_tools(self.server, self.db)                               │
│          register_resources(self.server, self.db)                           │
│                                                                              │
│      async def run(self):                                                   │
│          async with stdio_server() as (read, write):                        │
│              await self.server.run(read, write, ...)                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              mcp/tools.py                                    │
│                                                                              │
│  def register_tools(server: Server, db: Database):                          │
│                                                                              │
│      @server.tool()                                                         │
│      async def query(                                                       │
│          query: str,                                                        │
│          limit: int = 5,                                                    │
│          min_score: float = 0.0,                                            │
│          collection: str | None = None                                      │
│      ) -> str:                                                              │
│          """Hybrid search with query expansion and reranking."""            │
│                                                                              │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ # Resolve collection name to ID                             │     │
│          │ collection_id = None                                        │     │
│          │ if collection:                                              │     │
│          │     coll = CollectionRepository(db).get_by_name(collection) │     │
│          │     if not coll:                                            │     │
│          │         return json.dumps({"error": "Collection not found"})│     │
│          │     collection_id = coll.id                                 │     │
│          └────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ # Create pipeline and execute search                        │     │
│          │ pipeline = HybridSearchPipeline(                            │     │
│          │     search_repo=SearchRepository(db),                       │     │
│          │     llm_provider=OllamaProvider(),                          │     │
│          │     config=SearchPipelineConfig()                           │     │
│          │ )                                                           │     │
│          │                                                             │     │
│          │ results = await pipeline.search(                            │     │
│          │     query=query,                                            │     │
│          │     limit=limit,                                            │     │
│          │     collection_id=collection_id,                            │     │
│          │     min_score=min_score                                     │     │
│          │ )                                                           │     │
│          └────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│          ┌────────────────────────────────────────────────────────────┐     │
│          │ # Format results for MCP response                           │     │
│          │ formatter = JSONFormatter(include_body=False)               │     │
│          │ return formatter.format_many(results)                       │     │
│          └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ JSON-RPC response
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AI Agent receives response                           │
│                                                                              │
│  {                                                                           │
│    "content": [                                                              │
│      {                                                                       │
│        "type": "text",                                                       │
│        "text": "[{\"score\": 0.85, \"file\": \"pmd://...\", ...}]"          │
│      }                                                                       │
│    ]                                                                         │
│  }                                                                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.7 Class Dependency Graph

```
                              ┌─────────────┐
                              │   Config    │
                              │  (config.py)│
                              └──────┬──────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │  Database   │          │   Ollama    │          │  Formatters │
    │(database.py)│          │  Provider   │          │   (base,    │
    └──────┬──────┘          │ (ollama.py) │          │  json, csv) │
           │                 └──────┬──────┘          └─────────────┘
           │                        │                        ▲
    ┌──────┴──────────────┬────────┴────────┐               │
    │                     │                 │               │
    ▼                     ▼                 ▼               │
┌─────────┐        ┌───────────┐     ┌───────────┐         │
│Collection│       │  Document │     │ Embedding │         │
│Repository│       │ Repository│     │ Repository│         │
└────┬────┘        └─────┬─────┘     └─────┬─────┘         │
     │                   │                 │               │
     │    ┌──────────────┴─────────────────┘               │
     │    │                                                │
     ▼    ▼                                                │
┌─────────────┐      ┌─────────────┐                       │
│   Search    │      │  Document   │                       │
│ Repository  │      │  Chunker    │                       │
└──────┬──────┘      └─────────────┘                       │
       │                                                   │
       │         ┌─────────────────────────────────────────┘
       │         │
       ▼         ▼
┌─────────────────────┐     ┌─────────────────┐
│  HybridSearch       │     │  QueryExpander  │
│    Pipeline         │────▶│                 │
│  (pipeline.py)      │     └─────────────────┘
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌─────────┐
│  RRF   │   │ Scoring │
│Fusion  │   │(blend)  │
└────────┘   └─────────┘
```

---
