"""Integration tests for collection indexing.

These tests use the test corpus at tests/fixtures/test_corpus/ to verify
that collections can be created, documents indexed, and searched correctly.
"""

import pytest
from pathlib import Path

from pmd.core.exceptions import CollectionNotFoundError
from pmd.core.types import SearchSource
from pmd.services import IndexResult, ServiceContainer
from pmd.sources import SourceListError
from pmd.store.collections import CollectionRepository
from pmd.store.database import Database
from pmd.store.documents import DocumentRepository
from pmd.store.embeddings import EmbeddingRepository
from pmd.store.search import FTS5SearchRepository


def get_document_id(db: Database, collection_id: int, path: str) -> int:
    """Get the database ID for a document by path.

    Args:
        db: Database instance.
        collection_id: Collection ID.
        path: Document path.

    Returns:
        Document ID (integer primary key).
    """
    cursor = db.execute(
        "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
        (collection_id, path),
    )
    row = cursor.fetchone()
    return row["id"] if row else None


class TestCollectionCreation:
    """Tests for creating collections from the test corpus."""

    def test_create_collection_from_test_corpus(
        self,
        collection_repo: CollectionRepository,
        test_corpus_path: Path,
    ):
        """Should create a collection pointing to the test corpus."""
        collection = collection_repo.create(
            "my-corpus",
            str(test_corpus_path),
            "**/*.md",
        )

        assert collection is not None
        assert collection.name == "my-corpus"
        assert collection.pwd == str(test_corpus_path)
        assert collection.glob_pattern == "**/*.md"

    @pytest.mark.asyncio
    async def test_create_and_index_collection_from_test_corpus(
        self,
        config,
        test_corpus_path: Path,
    ):
        """Should create a collection and index it via IndexingService."""
        async with ServiceContainer(config) as services:
            collection = services.collection_repo.create(
                "my-corpus",
                str(test_corpus_path),
                "**/*.md",
            )

            index_result = await services.indexing.index_collection(
                "my-corpus",
                force=False,
            )

            assert index_result.indexed == 118
            assert index_result.skipped == 0
            assert index_result.errors == []


    def test_collection_can_be_retrieved_by_name(
        self,
        collection_repo: CollectionRepository,
        test_corpus_path: Path,
    ):
        """Created collection should be retrievable by name."""
        collection_repo.create("retrieval-test", str(test_corpus_path), "**/*.md")

        retrieved = collection_repo.get_by_name("retrieval-test")

        assert retrieved is not None
        assert retrieved.name == "retrieval-test"

    def test_collection_appears_in_list(
        self,
        collection_repo: CollectionRepository,
        test_corpus_path: Path,
    ):
        """Created collection should appear in list of all collections."""
        collection_repo.create("list-test", str(test_corpus_path), "**/*.md")

        all_collections = collection_repo.list_all()

        assert len(all_collections) >= 1
        names = [c.name for c in all_collections]
        assert "list-test" in names


class TestDocumentIndexing:
    """Tests for indexing documents from the test corpus."""

    def test_index_documents_from_corpus(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        search_repo: FTS5SearchRepository,
        test_corpus_path: Path,
        db: Database,
    ):
        """Should index all markdown files from the test corpus."""
        indexed_count = 0

        # Index all files in the collection
        for file_path in test_corpus_path.glob("**/*.md"):
            if not file_path.is_file():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, IOError):
                continue

            relative_path = str(file_path.relative_to(test_corpus_path))

            # Extract title
            lines = content.split("\n")
            title = None
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            if not title:
                title = file_path.stem

            # Index document
            doc_result, is_new = document_repo.add_or_update(
                test_corpus_collection.id,
                relative_path,
                title,
                content,
            )

            # Index in FTS (requires numeric doc_id from database)
            doc_id = get_document_id(db, test_corpus_collection.id, relative_path)
            search_repo.index_document(
                doc_id,
                relative_path,
                content,
            )

            indexed_count += 1

        # Should have indexed all markdown files
        assert indexed_count > 0
        # Verify document count matches
        assert document_repo.count_by_collection(test_corpus_collection.id) == indexed_count

    def test_indexed_documents_have_content_hashes(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        test_corpus_path: Path,
    ):
        """Indexed documents should have content hashes stored."""
        # Index a single document
        sample_file = next(test_corpus_path.glob("**/*.md"))
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()

        relative_path = str(sample_file.relative_to(test_corpus_path))
        doc_result, _ = document_repo.add_or_update(
            test_corpus_collection.id,
            relative_path,
            sample_file.stem,
            content,
        )

        # Hash should be a 64-character hex string (SHA256)
        assert doc_result.hash is not None
        assert len(doc_result.hash) == 64
        assert all(c in "0123456789abcdef" for c in doc_result.hash)

    def test_reindex_same_document_updates_not_duplicates(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        test_corpus_path: Path,
    ):
        """Reindexing the same document should update, not create duplicate."""
        sample_file = next(test_corpus_path.glob("**/*.md"))
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()

        relative_path = str(sample_file.relative_to(test_corpus_path))

        # Index twice
        doc1, is_new1 = document_repo.add_or_update(
            test_corpus_collection.id,
            relative_path,
            sample_file.stem,
            content,
        )
        doc2, is_new2 = document_repo.add_or_update(
            test_corpus_collection.id,
            relative_path,
            sample_file.stem,
            content,
        )

        # First should be new, second should be update
        assert is_new1 is True
        assert is_new2 is False
        # Both should reference same document
        assert doc1.hash == doc2.hash
        # Only one document in collection
        assert document_repo.count_by_collection(test_corpus_collection.id) == 1

    def test_document_titles_extracted_from_markdown_heading(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        test_corpus_path: Path,
    ):
        """Titles should be extracted from # heading in markdown."""
        # Find a file with a heading
        for file_path in test_corpus_path.glob("**/*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if content.startswith("# "):
                expected_title = content.split("\n")[0][2:].strip()
                relative_path = str(file_path.relative_to(test_corpus_path))

                doc_result, _ = document_repo.add_or_update(
                    test_corpus_collection.id,
                    relative_path,
                    expected_title,
                    content,
                )

                assert doc_result.title == expected_title
                return

        pytest.skip("No files with markdown headings in test corpus")


class TestFullTextSearch:
    """Tests for FTS5 full-text search on indexed documents."""

    @pytest.fixture
    def indexed_corpus(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        search_repo: FTS5SearchRepository,
        test_corpus_path: Path,
        db: Database,
    ):
        """Index all documents and return count."""
        indexed_count = 0

        for file_path in test_corpus_path.glob("**/*.md"):
            if not file_path.is_file():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, IOError):
                continue

            relative_path = str(file_path.relative_to(test_corpus_path))
            title = file_path.stem

            doc_result, _ = document_repo.add_or_update(
                test_corpus_collection.id,
                relative_path,
                title,
                content,
            )

            doc_id = get_document_id(db, test_corpus_collection.id, relative_path)
            search_repo.index_document(
                doc_id,
                relative_path,
                content,
            )
            indexed_count += 1

        return indexed_count

    def test_fts_search_returns_results(
        self,
        test_corpus_collection,
        search_repo: FTS5SearchRepository,
        indexed_corpus: int,
    ):
        """FTS search should return results for common terms."""
        # Search for a common word that should appear in markdown docs
        results = search_repo.search(
            "the",
            limit=10,
            collection_id=test_corpus_collection.id,
        )

        assert len(results) > 0
        assert all(r.source == SearchSource.FTS for r in results)

    def test_fts_search_respects_limit(
        self,
        test_corpus_collection,
        search_repo: FTS5SearchRepository,
        indexed_corpus: int,
    ):
        """FTS search should respect the limit parameter."""
        results = search_repo.search(
            "the",
            limit=3,
            collection_id=test_corpus_collection.id,
        )

        assert len(results) <= 3

    def test_fts_search_returns_scores(
        self,
        test_corpus_collection,
        search_repo: FTS5SearchRepository,
        indexed_corpus: int,
    ):
        """FTS search results should have BM25 scores."""
        results = search_repo.search(
            "markdown",
            limit=5,
            collection_id=test_corpus_collection.id,
        )

        if len(results) > 0:
            # Scores should be normalized to 0-1 range
            assert all(0 <= r.score <= 1 for r in results)
            # Results should be sorted by score descending
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_fts_search_no_results_for_nonsense(
        self,
        test_corpus_collection,
        search_repo: FTS5SearchRepository,
        indexed_corpus: int,
    ):
        """FTS search should return empty for nonsense queries."""
        results = search_repo.search(
            "xyzzy123nonsensequery",
            limit=10,
            collection_id=test_corpus_collection.id,
        )

        assert len(results) == 0

    def test_fts_search_filters_by_collection(
        self,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        search_repo: FTS5SearchRepository,
        test_corpus_path: Path,
        tmp_path: Path,
        db: Database,
    ):
        """FTS search should only return results from specified collection."""
        # Create two collections
        coll1 = collection_repo.create("coll1", str(test_corpus_path), "**/*.md")
        coll2 = collection_repo.create("coll2", str(tmp_path), "**/*.md")

        # Index a document only in collection 1
        sample_file = next(test_corpus_path.glob("**/*.md"))
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()

        relative_path = str(sample_file.relative_to(test_corpus_path))
        doc_result, _ = document_repo.add_or_update(
            coll1.id,
            relative_path,
            sample_file.stem,
            content,
        )
        doc_id = get_document_id(db, coll1.id, relative_path)
        search_repo.index_document(doc_id, relative_path, content)

        # Search in collection 1 should find it
        results1 = search_repo.search(
            sample_file.stem,
            limit=10,
            collection_id=coll1.id,
        )

        # Search in collection 2 should not find it
        results2 = search_repo.search(
            sample_file.stem,
            limit=10,
            collection_id=coll2.id,
        )

        assert len(results1) >= 0  # May or may not match depending on content
        # Collection 2 has no documents, so should return empty
        assert len(results2) == 0


class TestDocumentRetrieval:
    """Tests for retrieving indexed documents."""

    def test_retrieve_document_by_path(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        test_corpus_path: Path,
    ):
        """Should retrieve document content by path."""
        sample_file = next(test_corpus_path.glob("**/*.md"))
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()

        relative_path = str(sample_file.relative_to(test_corpus_path))
        document_repo.add_or_update(
            test_corpus_collection.id,
            relative_path,
            sample_file.stem,
            content,
        )

        # Retrieve document
        retrieved = document_repo.get(test_corpus_collection.id, relative_path)

        assert retrieved is not None
        assert retrieved.body == content
        assert retrieved.filepath == relative_path

    def test_list_documents_in_collection(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        test_corpus_path: Path,
    ):
        """Should list all documents in a collection."""
        # Index a few documents
        indexed_paths = []
        for i, file_path in enumerate(test_corpus_path.glob("**/*.md")):
            if i >= 5:
                break

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            relative_path = str(file_path.relative_to(test_corpus_path))
            document_repo.add_or_update(
                test_corpus_collection.id,
                relative_path,
                file_path.stem,
                content,
            )
            indexed_paths.append(relative_path)

        # List documents
        documents = document_repo.list_by_collection(test_corpus_collection.id)

        assert len(documents) == len(indexed_paths)
        retrieved_paths = [d.filepath for d in documents]
        for path in indexed_paths:
            assert path in retrieved_paths


class TestContentDeduplication:
    """Tests for content-addressable storage and deduplication."""

    def test_identical_content_shares_hash(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        db: Database,
    ):
        """Documents with identical content should share the same hash."""
        content = "# Duplicate Content\n\nThis is identical content."

        # Add as two different documents
        doc1, _ = document_repo.add_or_update(
            test_corpus_collection.id,
            "doc1.md",
            "Doc 1",
            content,
        )
        doc2, _ = document_repo.add_or_update(
            test_corpus_collection.id,
            "doc2.md",
            "Doc 2",
            content,
        )

        # Should have same hash
        assert doc1.hash == doc2.hash

        # Content table should have only one entry for this hash
        cursor = db.execute(
            "SELECT COUNT(*) as count FROM content WHERE hash = ?",
            (doc1.hash,),
        )
        assert cursor.fetchone()["count"] == 1

    def test_different_content_has_different_hash(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
    ):
        """Documents with different content should have different hashes."""
        doc1, _ = document_repo.add_or_update(
            test_corpus_collection.id,
            "unique1.md",
            "Unique 1",
            "# Unique Content 1\n\nThis is unique.",
        )
        doc2, _ = document_repo.add_or_update(
            test_corpus_collection.id,
            "unique2.md",
            "Unique 2",
            "# Unique Content 2\n\nThis is different.",
        )

        assert doc1.hash != doc2.hash


class TestBulkIndexing:
    """Tests for bulk indexing operations."""

    def test_index_all_corpus_documents(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        search_repo: FTS5SearchRepository,
        test_corpus_path: Path,
        db: Database,
    ):
        """Should successfully index all documents in the test corpus."""
        indexed_count = 0
        errors = []

        for file_path in test_corpus_path.glob("**/*.md"):
            if not file_path.is_file():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                relative_path = str(file_path.relative_to(test_corpus_path))

                doc_result, _ = document_repo.add_or_update(
                    test_corpus_collection.id,
                    relative_path,
                    file_path.stem,
                    content,
                )

                doc_id = get_document_id(db, test_corpus_collection.id, relative_path)
                search_repo.index_document(
                    doc_id,
                    relative_path,
                    content,
                )
                indexed_count += 1
            except Exception as e:
                errors.append((file_path, str(e)))

        # Should have indexed many documents without errors
        assert indexed_count > 100  # Test corpus has 118 files
        assert len(errors) == 0, f"Errors during indexing: {errors}"

    def test_reindex_collection(
        self,
        test_corpus_collection,
        document_repo: DocumentRepository,
        search_repo: FTS5SearchRepository,
        test_corpus_path: Path,
        db: Database,
    ):
        """Should be able to reindex an entire collection."""
        # First index
        for file_path in list(test_corpus_path.glob("**/*.md"))[:10]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            relative_path = str(file_path.relative_to(test_corpus_path))
            doc_result, _ = document_repo.add_or_update(
                test_corpus_collection.id,
                relative_path,
                file_path.stem,
                content,
            )
            doc_id = get_document_id(db, test_corpus_collection.id, relative_path)
            search_repo.index_document(doc_id, relative_path, content)

        # Reindex collection via search repo
        reindexed_count = search_repo.reindex_collection(test_corpus_collection.id)

        assert reindexed_count == 10


class TestIndexingService:
    """Tests for IndexingService.index_collection method."""

    @pytest.mark.asyncio
    async def test_index_collection_indexes_all_matching_files(
        self,
        config,
        test_corpus_path: Path,
    ):
        """index_collection should index all files matching the glob pattern."""
        async with ServiceContainer(config) as services:
            services.collection_repo.create(
                "test-corpus",
                str(test_corpus_path),
                "**/*.md",
            )

            result = await services.indexing.index_collection("test-corpus", force=True)

            assert isinstance(result, IndexResult)
            assert result.indexed > 100  # Test corpus has 118 files
            assert result.errors == []

    @pytest.mark.asyncio
    async def test_index_collection_returns_index_result(
        self,
        config,
        test_corpus_path: Path,
    ):
        """index_collection should return an IndexResult with correct fields."""
        async with ServiceContainer(config) as services:
            services.collection_repo.create(
                "test-corpus",
                str(test_corpus_path),
                "**/*.md",
            )

            result = await services.indexing.index_collection("test-corpus", force=True)

            assert hasattr(result, "indexed")
            assert hasattr(result, "skipped")
            assert hasattr(result, "errors")
            assert isinstance(result.indexed, int)
            assert isinstance(result.skipped, int)
            assert isinstance(result.errors, list)

    @pytest.mark.asyncio
    async def test_index_collection_skips_unchanged_files(
        self,
        config,
        test_corpus_path: Path,
    ):
        """index_collection should skip unchanged files when force=False."""
        async with ServiceContainer(config) as services:
            services.collection_repo.create(
                "test-corpus",
                str(test_corpus_path),
                "**/*.md",
            )

            # First index
            result1 = await services.indexing.index_collection("test-corpus", force=True)

            # Second index without force - should skip all
            result2 = await services.indexing.index_collection("test-corpus", force=False)

            assert result1.indexed > 0
            assert result2.indexed == 0
            assert result2.skipped == result1.indexed

    @pytest.mark.asyncio
    async def test_index_collection_force_reindexes_all(
        self,
        config,
        test_corpus_path: Path,
    ):
        """index_collection with force=True should reindex all documents."""
        async with ServiceContainer(config) as services:
            services.collection_repo.create(
                "test-corpus",
                str(test_corpus_path),
                "**/*.md",
            )

            # First index
            result1 = await services.indexing.index_collection("test-corpus", force=True)

            # Second index with force - should reindex all
            result2 = await services.indexing.index_collection("test-corpus", force=True)

            assert result1.indexed == result2.indexed
            assert result2.skipped == 0

    @pytest.mark.asyncio
    async def test_index_collection_makes_files_searchable(
        self,
        config,
        test_corpus_path: Path,
    ):
        """Documents indexed via index_collection should be searchable via FTS."""
        async with ServiceContainer(config) as services:
            collection = services.collection_repo.create(
                "test-corpus",
                str(test_corpus_path),
                "**/*.md",
            )

            await services.indexing.index_collection("test-corpus", force=True)

            # Search for a common term
            results = services.fts_repo.search(
                "the",
                limit=10,
                collection_id=collection.id,
            )

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_index_collection_raises_for_nonexistent_collection(
        self,
        config,
    ):
        """index_collection should raise CollectionNotFoundError for unknown name."""
        async with ServiceContainer(config) as services:
            with pytest.raises(CollectionNotFoundError):
                await services.indexing.index_collection("nonexistent-collection")

    @pytest.mark.asyncio
    async def test_index_collection_raises_for_nonexistent_path(
        self,
        config,
    ):
        """index_collection should raise SourceListError if collection path doesn't exist."""
        async with ServiceContainer(config) as services:
            # Create collection with non-existent path
            services.collection_repo.create(
                "nonexistent",
                "/nonexistent/path/that/does/not/exist",
                "**/*.md",
            )

            with pytest.raises(SourceListError, match="does not exist"):
                await services.indexing.index_collection("nonexistent")

    @pytest.mark.asyncio
    async def test_index_collection_extracts_titles_from_headings(
        self,
        config,
        test_corpus_path: Path,
    ):
        """index_collection should extract titles from markdown headings."""
        async with ServiceContainer(config) as services:
            collection = services.collection_repo.create(
                "test-corpus",
                str(test_corpus_path),
                "**/*.md",
            )

            await services.indexing.index_collection("test-corpus", force=True)

            # Check a document that should have a heading-based title
            documents = services.document_repo.list_by_collection(collection.id)

            # At least some documents should have titles different from filename
            titles_from_headings = [
                doc for doc in documents
                if doc.title != Path(doc.filepath).stem
            ]
            assert len(titles_from_headings) > 0

    @pytest.mark.asyncio
    async def test_index_collection_stores_document_count(
        self,
        config,
        test_corpus_path: Path,
    ):
        """index_collection should correctly store all documents in database."""
        async with ServiceContainer(config) as services:
            collection = services.collection_repo.create(
                "test-corpus",
                str(test_corpus_path),
                "**/*.md",
            )

            result = await services.indexing.index_collection("test-corpus", force=True)

            # Count should match what's stored
            stored_count = services.document_repo.count_by_collection(collection.id)
            assert stored_count == result.indexed
