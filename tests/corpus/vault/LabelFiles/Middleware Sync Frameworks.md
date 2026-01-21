# Middleware Sync Frameworks

### TinyBase

<https://tinybase.org>

Connects and syncs to a bunch of stuff. Splits persistence, merge, and sync concerns on top of a local Store.

Supports very simple UIs via a Tinybase ui package built on React: <https://tinybase.org/guides/building-uis/>

Supports javascript and typescript clients, and ships [Vite templates](https://tinybase.org/guides/the-basics/getting-started/) for various scenarios (DuraboleObject, PartyKit, ElectricSQL, PGLite, Expo/React Native, Turso)

#### Persister

<https://tinybase.org/api/persisters/>

Databases that can be persisted to, either directly or indirectly (via a local Store). A direct store example is a Cloudflare Durable Object, which supports online writes only.  Some persisters only support local stores, and rely on the persister itself to perform sync (like an ElectricSql or PowerSync persister, which manages sync).

#### Mergeable

Some persisters are mergeable, meaning Tinybase will handle their synchronization.

#### Synchronizer

Network layer for performing the merge operations, can be websockets, direct connection etc.

#### Eval

[TinyBase - Architectural Options](https://tinybase.org/guides/the-basics/architectural-options/)

Supports React Native

Deployable via Expo, with examples

Has a simple UI builder

Integrates with:

- ElectricSQL

- Cloudflare DurableObject

- Cloudflare PartyKit

- PowerSync

- Turso

Can sync to, given a sync server running somewhere

- Yjs documents

- Automerge documents

- cr-sqlite

Supports databases:

- Postgres

- Sqlite

- Pglite

### ElectricSQL

Will take new data into the central Postgres and replicate it to the mobile client. Will also support the mobile client making writes that will get back to the central database in[ a variety of ways](https://electric-sql.com/docs/guides/writes) (usually via the client sdk).

- [ ] How do mobile client writes get replicated back to the central database? I think its via the electric API

### 

# 