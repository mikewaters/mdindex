Feature: Ingest an Obsidian vault
  Background:
    Given a temporary database with FTS enabled
    And an ingest pipeline
    And a valid Obsidian vault

  Scenario: Obsidian ingestion creates a dataset
    When I ingest the Obsidian vault as dataset "my-vault"
    Then the ingest result dataset name is "my-vault"
    And the ingest result dataset id is positive
    And a dataset named "my-vault" exists with source type "obsidian"

  Scenario: Obsidian ingestion creates documents
    When I ingest the Obsidian vault as dataset "my-vault"
    Then the obsidian ingest result counts are created=4 updated=0 skipped=0 failed=0

  Scenario: Obsidian ingestion extracts frontmatter metadata
    When I ingest the Obsidian vault as dataset "my-vault"
    Then the document "note1.md" metadata includes tags "work,important"
    And the document "note1.md" metadata includes aliases "First Note"

  Scenario: Obsidian ingestion handles documents without frontmatter
    When I ingest the Obsidian vault as dataset "my-vault"
    Then the document "plain.md" has no tags or aliases in metadata

  Scenario: Obsidian ingestion updates the FTS index
    When I ingest the Obsidian vault as dataset "my-vault"
    Then the FTS index contains 4 entries
    And searching the FTS index for "nested" returns at least 1 result

  Scenario: Obsidian force mode updates all documents
    Given I have ingested the Obsidian vault as dataset "my-vault"
    When I ingest the Obsidian vault as dataset "my-vault" in force mode
    Then the obsidian ingest result counts are created=0 updated=4 skipped=0 failed=0

  Scenario: Invalid vault raises an error
    Given an invalid Obsidian vault
    When I attempt to ingest the Obsidian vault as dataset "test"
    Then a ValueError is raised containing "missing .obsidian"

