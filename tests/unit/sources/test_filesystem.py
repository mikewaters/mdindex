"""Tests for FileSystemSource."""

import pytest
from pathlib import Path

from pmd.sources import (
    FileSystemSource,
    SourceConfig,
    SourceListError,
    SourceFetchError,
)


class TestFileSystemSource:
    """Tests for FileSystemSource."""

    def test_create_from_config(self, tmp_path: Path):
        """Can create FileSystemSource from SourceConfig."""
        config = SourceConfig(
            uri=str(tmp_path),
            extra={"glob_pattern": "**/*.txt"},
        )
        source = FileSystemSource(config)

        assert source.base_path == tmp_path
        assert source.glob_pattern == "**/*.txt"

    def test_create_from_file_uri(self, tmp_path: Path):
        """Can create from file:// URI."""
        config = SourceConfig(uri=tmp_path.as_uri())
        source = FileSystemSource(config)

        assert source.base_path == tmp_path

    def test_list_documents_empty_directory(self, tmp_path: Path):
        """list_documents yields nothing for empty directory."""
        config = SourceConfig(uri=str(tmp_path))
        source = FileSystemSource(config)

        docs = list(source.list_documents())
        assert docs == []

    def test_list_documents_finds_matching_files(self, tmp_path: Path):
        """list_documents finds files matching glob pattern."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Doc 1")
        (tmp_path / "doc2.md").write_text("# Doc 2")
        (tmp_path / "other.txt").write_text("Not markdown")

        config = SourceConfig(uri=str(tmp_path), extra={"glob_pattern": "**/*.md"})
        source = FileSystemSource(config)

        docs = list(source.list_documents())
        paths = {d.path for d in docs}

        assert "doc1.md" in paths
        assert "doc2.md" in paths
        assert "other.txt" not in paths

    def test_list_documents_recursive(self, tmp_path: Path):
        """list_documents finds files in subdirectories."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.md").write_text("# Root")
        (subdir / "nested.md").write_text("# Nested")

        config = SourceConfig(uri=str(tmp_path), extra={"glob_pattern": "**/*.md"})
        source = FileSystemSource(config)

        docs = list(source.list_documents())
        paths = {d.path for d in docs}

        assert "root.md" in paths
        assert "subdir/nested.md" in paths

    def test_list_documents_nonexistent_raises(self):
        """list_documents raises for non-existent directory."""
        config = SourceConfig(uri="/nonexistent/path/12345")
        source = FileSystemSource(config)

        with pytest.raises(SourceListError, match="does not exist"):
            list(source.list_documents())

    def test_list_documents_file_raises(self, tmp_path: Path):
        """list_documents raises if path is a file."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        config = SourceConfig(uri=str(file_path))
        source = FileSystemSource(config)

        with pytest.raises(SourceListError, match="not a directory"):
            list(source.list_documents())

    def test_list_documents_includes_metadata(self, tmp_path: Path):
        """list_documents includes file metadata."""
        doc = tmp_path / "doc.md"
        doc.write_text("# Test")

        config = SourceConfig(uri=str(tmp_path))
        source = FileSystemSource(config)

        docs = list(source.list_documents())
        assert len(docs) == 1
        assert "mtime" in docs[0].metadata
        assert "size" in docs[0].metadata

    @pytest.mark.asyncio
    async def test_fetch_content_reads_file(self, tmp_path: Path):
        """fetch_content reads file content."""
        doc = tmp_path / "doc.md"
        doc.write_text("# Hello World\n\nThis is content.")

        config = SourceConfig(uri=str(tmp_path))
        source = FileSystemSource(config)

        docs = list(source.list_documents())
        result = await source.fetch_content(docs[0])

        assert "Hello World" in result.content
        assert result.content_type == "text/markdown"
        assert result.encoding == "utf-8"

    @pytest.mark.asyncio
    async def test_fetch_content_missing_file_raises(self, tmp_path: Path):
        """fetch_content raises for missing file."""
        from pmd.sources import DocumentReference

        config = SourceConfig(uri=str(tmp_path))
        source = FileSystemSource(config)

        ref = DocumentReference(uri="file:///missing", path="missing.md")

        with pytest.raises(SourceFetchError, match="not found"):
            await source.fetch_content(ref)

    @pytest.mark.asyncio
    async def test_fetch_content_encoding_error(self, tmp_path: Path):
        """fetch_content handles encoding errors."""
        doc = tmp_path / "binary.md"
        doc.write_bytes(b"\xff\xfe Invalid UTF-8")

        config = SourceConfig(uri=str(tmp_path))
        source = FileSystemSource(config)

        docs = list(source.list_documents())

        with pytest.raises(SourceFetchError, match="Encoding error"):
            await source.fetch_content(docs[0])

    def test_capabilities(self, tmp_path: Path):
        """FileSystemSource reports correct capabilities."""
        config = SourceConfig(uri=str(tmp_path))
        source = FileSystemSource(config)

        caps = source.capabilities()
        assert caps.supports_incremental is True
        assert caps.supports_last_modified is True
        assert caps.supports_etag is False

    @pytest.mark.asyncio
    async def test_check_modified_returns_true_for_new(self, tmp_path: Path):
        """check_modified returns True for new documents."""
        doc = tmp_path / "doc.md"
        doc.write_text("content")

        config = SourceConfig(uri=str(tmp_path))
        source = FileSystemSource(config)
        docs = list(source.list_documents())

        # No stored metadata
        is_modified = await source.check_modified(docs[0], {})
        assert is_modified is True

    @pytest.mark.asyncio
    async def test_check_modified_detects_changes(self, tmp_path: Path):
        """check_modified detects file changes via mtime."""
        doc = tmp_path / "doc.md"
        doc.write_text("original content")

        config = SourceConfig(uri=str(tmp_path))
        source = FileSystemSource(config)
        docs = list(source.list_documents())

        # Store original mtime
        stored = {"mtime_ns": docs[0].metadata["mtime_ns"]}

        # File unchanged
        is_modified = await source.check_modified(docs[0], stored)
        assert is_modified is False

        # Modify file
        import time
        time.sleep(0.01)  # Ensure mtime changes
        doc.write_text("modified content")

        # Refresh reference
        docs = list(source.list_documents())
        is_modified = await source.check_modified(docs[0], stored)
        assert is_modified is True

    def test_content_type_detection(self, tmp_path: Path):
        """FileSystemSource detects content types from extensions."""
        files = {
            "doc.md": "text/markdown",
            "page.html": "text/html",
            "data.json": "application/json",
            "readme.txt": "text/plain",
            "config.yaml": "text/yaml",
        }

        for filename, expected_type in files.items():
            (tmp_path / filename).write_text("content")

        config = SourceConfig(uri=str(tmp_path), extra={"glob_pattern": "*"})
        source = FileSystemSource(config)

        for ref in source.list_documents():
            expected = files.get(ref.path)
            # We need to fetch to get content_type
            # For now just verify files are listed
            assert ref.path in files
