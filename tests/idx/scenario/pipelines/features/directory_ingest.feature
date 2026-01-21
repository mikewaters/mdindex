Feature: Ingest markdown from a directory
  Background:
    Given a temporary database with FTS enabled
    And an ingest pipeline
    And a sample directory with markdown files

  Scenario: Ingestion creates a dataset
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the ingest result dataset name is "test-docs"
    And the ingest result dataset id is positive
    And a dataset named "test-docs" exists with source type "directory"

  Scenario: Ingestion creates documents for matching files
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the ingest result counts are created=3 updated=0 skipped=0 stale=0 failed=0
    And the dataset documents are exactly "notes.md,readme.md,subdir/deep.md"

  Scenario: Ingestion updates the FTS index
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the FTS index contains 3 entries
    And searching the FTS index for "readme" returns at least 1 result
    And the FTS search results for "readme" include a path containing "readme.md"

  Scenario: Re-ingestion skips unchanged documents
    Given I have ingested the directory as dataset "test-docs"
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the ingest result counts are created=0 updated=0 skipped=3 stale=0 failed=0

  Scenario: Re-ingestion updates changed documents
    Given I have ingested the directory as dataset "test-docs"
    And I change the file "readme.md" content
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the ingest result counts are created=0 updated=1 skipped=2 stale=0 failed=0

  Scenario: Force mode updates all documents
    Given I have ingested the directory as dataset "test-docs"
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md" in force mode
    Then the ingest result counts are created=0 updated=3 skipped=0 stale=0 failed=0

  Scenario: Exclusion patterns are respected
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md,!**/subdir/**"
    Then the ingest result counts are created=2 updated=0 skipped=0 stale=0 failed=0
    And the dataset documents are exactly "notes.md,readme.md"

  Scenario: Dataset name is normalized
    When I ingest the directory as dataset "My Test Docs!" with patterns "**/*.md"
    Then the ingest result dataset name is "my-test-docs"

  Scenario: Ingestion reuses an existing dataset with the same name
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md" again
    And I ingest the directory as dataset "test-docs" with patterns "**/*.md" again
    Then both ingestions use the same dataset id

  Scenario: IngestResult reports success and timestamps
    When I ingest the directory as dataset "test-docs" with patterns "**/*.md"
    Then the ingest result total processed is 3
    And the ingest result is successful
    And the ingest result has a completed timestamp not earlier than started

  Scenario: Empty directory ingests successfully
    Given an empty directory
    When I ingest the directory as dataset "empty-dataset" with patterns "**/*.md"
    Then the ingest result counts are created=0 updated=0 skipped=0 stale=0 failed=0
    And the ingest result is successful

  Scenario: Missing directory raises an error
    Given a missing directory path
    When I attempt to ingest the directory as dataset "test" with patterns "**/*.md"
    Then a FileNotFoundError is raised

