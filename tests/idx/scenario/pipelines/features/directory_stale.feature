Feature: Detect stale documents during directory ingestion
  Background:
    Given a temporary database with FTS enabled
    And an ingest pipeline
    And a sample directory with markdown files
    And I have ingested the directory as dataset "test-docs"

  Scenario: Re-ingestion soft-deletes missing documents
    Given the file "readme.md" is deleted from disk
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the ingest result counts are created=0 updated=0 skipped=2 stale=1 failed=0
    And the active dataset document paths are exactly "notes.md,subdir/deep.md"

  Scenario: Stale documents are removed from the FTS index
    Given the file "notes.md" is deleted from disk
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the ingest result counts are created=0 updated=0 skipped=2 stale=1 failed=0
    And the FTS index contains 2 entries
    And searching the FTS index for "notes" returns 0 results

  Scenario: Multiple files can be marked stale at once
    Given the file "readme.md" is deleted from disk
    And the file "notes.md" is deleted from disk
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the ingest result counts are created=0 updated=0 skipped=1 stale=2 failed=0

  Scenario: Renaming a file is treated as delete plus add
    Given the file "readme.md" is renamed to "readme-new.md"
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the ingest result counts are created=1 updated=0 skipped=2 stale=1 failed=0

