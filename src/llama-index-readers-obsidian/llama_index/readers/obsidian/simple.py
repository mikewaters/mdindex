"""
Simple Obsidian reader class.

A thin wrapper around SimpleDirectoryReader that adds Obsidian-specific
functionality including wikilink extraction, backlinks graph construction,
metadata enrichment, and optional task extraction.

Each document will contain the following metadata:
- file_name: the name of the markdown file
- folder_path: the full path to the folder containing the file
- folder_name: the relative path to the folder containing the file
- note_name: the name of the note (without the .md extension)
- wikilinks: a list of all wikilinks found in the document
- backlinks: a list of all notes that link to this note

Optionally, tasks can be extracted from the text and stored in metadata.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import MarkdownReader


def is_hardlink(filepath: Path) -> bool:
    """
    Check if a file is a hardlink by checking the number of links to/from it.

    Args:
        filepath: Path to the file.

    Returns:
        True if the file has more than one hard link.
    """
    stat_info = os.stat(filepath)
    return stat_info.st_nlink > 1

def extract_tasks(text: str, should_remove_tasks: bool) -> Tuple[List[str], str]:
    """
    Extract markdown tasks from text.

    A task is a checklist item in markdown, for example:
        - [ ] Do something
        - [x] Completed task

    Args:
        text: Document text to extract tasks from.

    Returns:
        Tuple of (list of task strings, text with task lines removed).
    """
    # Matches lines starting with '-' or '*' followed by a checkbox.
    task_pattern = re.compile(
        r"^\s*[-*]\s*\[\s*(?:x|X| )\s*\]\s*(.*)$", re.MULTILINE
    )
    tasks = task_pattern.findall(text)
    cleaned_text = task_pattern.sub("", text) if should_remove_tasks else text
    return tasks, cleaned_text

def extract_wikilinks(text: str) -> List[str]:
    """
    Extract Obsidian wikilinks from text.

    Matches patterns like:
        - [[Note Name]]
        - [[Note Name|Alias]]

    Args:
        text: Document text to extract wikilinks from.

    Returns:
        List of unique wikilink targets (aliases are stripped).
    """
    pattern = r"\[\[([^\]]+)\]\]"
    matches = re.findall(pattern, text)
    links = []
    for match in matches:
        # If a pipe is present (e.g. [[Note|Alias]]), take only the part before it.
        target = match.split("|")[0].strip()
        links.append(target)
    return list(set(links))

class SimpleObsidianReader(SimpleDirectoryReader):
    """
    Obsidian vault reader built on SimpleDirectoryReader.

    This reader walks an Obsidian vault, loads markdown files using MarkdownReader,
    and enriches documents with Obsidian-specific metadata including wikilinks
    and backlinks.

    Args:
        input_dir: Path to the Obsidian vault.
        extract_tasks: If True, extract tasks from the text and store them in metadata.
        remove_tasks_from_text: If True and extract_tasks is True, remove task lines
            from the main document text.
        exclude_hidden: Must be True (default). Non-hidden traversal is not supported.
        **kwargs: Additional arguments passed to SimpleDirectoryReader (except
            required_exts and file_extractor which are overridden).

    Raises:
        NotImplementedError: If exclude_hidden=False or non-local filesystem is used.
    """

    def __init__(
        self,
        input_dir: Path,
        extract_tasks: bool = False,
        remove_tasks_from_text: bool = False,
        **kwargs: Any,
    ) -> None:

        # Input checks
        if "exclude_hidden" in kwargs and not kwargs["exclude_hidden"]:
            raise NotImplementedError(
                "SimpleObsidianReader only supports exclude_hidden=True."
            )

        # Check for non-local filesystem (fsspec)
        if "fs" in kwargs and kwargs["fs"] is not None:
            raise NotImplementedError(
                "SimpleObsidianReader only supports local filesystems. "
                "Non-local sources via fsspec are not supported."
            )

        # Force markdown-only and use MarkdownReader
        kwargs.pop("required_exts", None)
        kwargs.pop("file_extractor", None)

        self._vault_root = input_dir.resolve()
        self._should_extract_tasks = extract_tasks
        self._should_remove_tasks = remove_tasks_from_text

        super().__init__(
            input_dir=str(input_dir),
            required_exts=[".md"],  # Markdown files only
            exclude_hidden=True,  # Exclude .obsidian and hidden files
            recursive=True,  # Walk subdirectories
            file_extractor={".md": MarkdownReader()},
            file_metadata=self.get_file_metadata,  # Build Obsidian metadata
            filename_as_id=True,  # stable doc IDs are required for caching
            raise_on_error=False, 
            **kwargs,
        )
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Generate Obsidian-specific metadata for a file.

        Includes both Obsidian-specific fields and standard file metadata
        that SimpleDirectoryReader normally provides.

        Args:
            file_path: Path to the markdown file.

        Returns:
            Dictionary containing file metadata for caching and Obsidian features.
        """
        metadata = {}

        file_path_obj = Path(file_path).resolve()
        metadata['file_name'] = file_path_obj.name
        metadata['folder_path'] = str(file_path_obj.parent)
        metadata['note_name'] = file_path_obj.stem

        try:
            folder_name = str(file_path_obj.parent.relative_to(self._vault_root))
            if folder_name == ".":
                folder_name = ""
        except ValueError:
            # Fallback if relative_to fails
            folder_name = str(file_path_obj.parent)

        metadata['folder_name'] = folder_name
        metadata['file_type'] = "text/markdown"
        metadata.update(self.get_resource_info(str(file_path_obj)))

        return metadata

    def _is_safe_file(self, file_path: Path) -> bool:
        """
        Check if a file passes safety checks (not a hardlink, within vault).

        Args:
            file_path: Path to check.

        Returns:
            True if the file is safe to process.
        """
        resolved_path = file_path.resolve()

        # Check for hardlinks
        try:
            if is_hardlink(resolved_path):
                print(
                    f"Warning: Skipping file because it is a hardlink "
                    f"(potential malicious exploit): {file_path}"
                )
                return False
        except OSError as e:
            print(f"Warning: Could not check hardlink status for {file_path}: {e}")
            return False

        # Check path containment
        try:
            resolved_path.relative_to(self._vault_root)
        except ValueError:
            print(f"Warning: Skipping file outside input directory: {file_path}")
            return False

        return True

    def load_data(
        self,
        show_progress: bool = False,
        num_workers: Optional[int] = None,
        **load_kwargs: Any,
    ) -> List[Document]:
        """
        Load documents from the Obsidian vault with full metadata enrichment.

        This method:
        1. Filters files through safety checks (hardlink, path containment)
        2. Loads markdown files via SimpleDirectoryReader
        3. Extracts wikilinks and builds backlinks graph
        4. Optionally extracts tasks

        Args:
            show_progress: If True, show a progress bar.
            num_workers: Number of workers for parallel loading.
            **load_kwargs: Additional arguments.

        Returns:
            List of Document objects with Obsidian metadata.
        """
        # Filter input files through safety checks before loading
        safe_files = []
        for file_path in self.input_files:
            if self._is_safe_file(file_path):
                safe_files.append(file_path)

        # Update input_files to only include safe files
        self.input_files = safe_files

        # Load documents using parent class
        docs = super().load_data(
            show_progress=show_progress,
            num_workers=num_workers,
            **load_kwargs,
        )

        # Build backlinks map: {target_note: [source_note1, source_note2, ...]}
        backlinks_map: Dict[str, List[str]] = {}

        # First pass: extract wikilinks and build backlinks map
        for i, doc in enumerate(docs):
            wikilinks = extract_wikilinks(doc.text)
            doc.metadata["wikilinks"] = wikilinks

            note_name = doc.metadata.get("note_name", "")
            for link in wikilinks:
                backlinks_map.setdefault(link, []).append(note_name)

            # Optionally extract tasks
            if self._should_extract_tasks:
                tasks, cleaned_text = extract_tasks(doc.text, self._should_remove_tasks)
                doc.metadata["tasks"] = tasks
                if self._should_remove_tasks:
                    docs[i] = Document(text=cleaned_text, metadata=doc.metadata)

        # Second pass: assign backlinks 
        for doc in docs:
            note_name = doc.metadata.get("note_name", "")
            doc.metadata["backlinks"] = backlinks_map.get(note_name, [])

        return docs

