---
tags:
  - experiment
---
# LifeOS Architecture Decisions

Principles

localfirst, barefoot developer etc

### Requirements

Correlation IDs, to gather cost and other metrics by user initiated flows

## Target State

### Integrations

- iCloud Drive

- Tana

- ClickUp

- Roam

- Heptabase

### High-Security Requirements

MY target state finds a highly secure implementation; this is not the first increment though. There is no way for personal information to be read by anyone without permission

For each, decide whether i can buy or need to host, and then i can compare stacks.

- certificates

- dns provider

- TLS that is not terminated by anyone else

- container hosting

- end to end encryption from the client, unless they are local/in a trusted network

- don’t rely on a tunnel provider for core services 

   - Tailscale DNS as an example

## Constraints

### Implementation Decisions

- Python/Typescript

- SQLite or Postgres

### Architecture Decisions

#### Data Sync

Certain reference data is globally available, though it may be sensitive. It is encrypted and accessible by client

#### Secure Partitioning

Some functions are bound to certain clients based on their security level

#### Asynchrony/Queuing

Defer incoming http requests, both locally and remotely

#### Replicated Data

Data stores are available on all clients, depending on need. Updates are instant.

#### Object Storage

Objects are stored locally and/or remotely, with synchronization when local and remote

### Architecture Principles

- Follow the **local first** paradigm; it interests me and I believe it will be influential in the future. 

- Use **React** if I have to write more than a little javascript/ **typescript**; this will help me professionally 

- Don’t learn a new language just yet, use what I know best. Once I have something working, I can experiment

### Development

- **Use an LLM tool to generate the code**, as much as possible. This means I can (or must?) make strong architecture decisions up front.

   - This includes (especially) UI code

### Deployment

The aim is to follow barefoot programmer craft principles, implying self-hosting. “Cloud-second”, though it might be faster to use “someone else’s computer” at first.