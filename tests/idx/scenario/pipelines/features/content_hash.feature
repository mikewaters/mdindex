Feature: Compute content hash
  Scenario: Same content produces the same hash
    Given a content hash calculator
    When I compute the content hash for "Hello, World!"
    And I compute the content hash for "Hello, World!" again
    Then the hashes are equal
    And the hash has 64 hex characters

  Scenario: Different content produces different hashes
    Given a content hash calculator
    When I compute the content hash for "Hello"
    And I compute the content hash for "World" again
    Then the hashes are different
    And the hash has 64 hex characters

  Scenario: Empty string produces a valid hash
    Given a content hash calculator
    When I compute the content hash for ""
    Then the hash has 64 hex characters

  Scenario: Unicode content is hashed correctly
    Given a content hash calculator
    When I compute the content hash for "Hello 世界 "
    Then the hash has 64 hex characters

