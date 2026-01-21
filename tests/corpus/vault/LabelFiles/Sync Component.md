---
tags:
  - type
objectType: "[Component.md](./Component.md)"
domain scope:
  - Information
---
# Sync Component

Is a:: Component

Status:: Proposed

Data in:: System³ Entities

Data out:: System³ Entities

Implementation: TBD

A globally accessible cache layer for storing shared System³ Entity metadata needed to maintain consistency across Components and Workflows.

*This could be the actual Single Point of Truth for this data.*

## Use Cases

- Keep a record of all [Project](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/3392160e-9db0-4b35-a5e2-ed9db1395e4d) and [Thread](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/910d5841-b09f-42b9-b389-a0ea99c4f1fb) names

- Provide information on the [Ontologies](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/faa4cfd7-f21c-45c8-9727-f1a33d62f82f) in use in the system

ref: [Implement a cache store for system³ entities.md](./Implement%20a%20cache%20store%20for%20system³%20entities.md)