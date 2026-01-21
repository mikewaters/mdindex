---
tags:
  - active
---
# Database Sync (localfirst)



---

Crsqlite Moribund. Developer went full time with offline replication vendor. Interesting project. Devil in details.

Yjs or crdt backend persistence solutions

- <https://github.com/jamsocket/y-sweet>

- realtime CRDT-based document store, backed by S3.

   ## Sync Platforms

   Typically, some service sits in front of the primary data store, consuming its changes via WAL or other method. Service fans out changes to clients, who have an embedded database of some kind (sqlite, indexeddb, custom). The protocol used can be http or something else, and the service can use crdt or some custom method for conflict resolution. The clients can either speak the protocol directly, and/or can use an SDK. Most of these only support “web tech” SDKs, where the clients are presumed to be browser applications.

   + ### Evaluation Criteria

      - SDK/client languages and runtimes

      - Backend type

      - Source database support

      - Replica persistence type

      - Protocol wire type

      - Synchronization protocol, reactivity

      - Licensing and hosting

      - e2e encryption

      - local-first, or offline supported

      - WASM

      - How can one use it with backend runtimes and have that be reactive on the frontend?

      #### refined criteria

      1. capabilities

      2. audience

      3. architecture

      4. ???

   + ### Sync Implementations

      + [Actyx](https://developer.actyx.com/) Local-First Cooperation based on Event Sourcing

         - durable event stream storage in peer-to-peer network using ++[libp2p](https://github.com/libp2p/rust-libp2p)++ and ++[ipfs-embed](https://github.com/ipfs-rust/ipfs-embed)++

         - mobile fe: android; fe: win/mac/lin desktop

         - be: Rust

         - runtime: docker

         - protocol: protobuf

         - arch: pure p2p, event sourcing

         - Actyx is a **decentralized** event **database**, **streaming** and **processing** engine that allows you to easily build ++[local-first cooperative](https://www.local-first-cooperation.org/)++ app

         - doesn't depend on any central components (servers, databases, or cloud)

         - Small pieces of code — they are called *local-twins* — contain the logic. Any app on any device can summon any twin at any time. Actyx automatically synchronizes the twins within the local network.

         - [Code](https://github.com/Actyx/Actyx)

      + [Any Sync](https://anytype.io) (Anytype) the everything app for those who celebrate trust & autonomy

         - An open-source protocol designed to create high-performance, local-first, peer-to-peer, end-to-end encrypted applications that facilitate seamless collaboration among multiple users and devices

         - Local first

         - Protocol:[ CRDT DAGs](https://tech.anytype.io/any-sync/overview), local p2p via mDNS

         - Security: e2e, creators’ controlled keys; CR is on-device;  [Ed25519](https://en.wikipedia.org/wiki/EdDSA) algorithm for signing

         - File storage in IPLD

         - component arch: four types of nodes: sync, file, consensus and coordinator nodes

         - [Code](https://github.com/anyproto/any-sync)

      + [Convex](https://www.convex.dev) the sync platform that replaces your backend and client state management ($ hosted option) BaaS

         - hosted open-source BaaS

         - Hosted; They do not recommend running this yourself (wtf??)

            > This repository is the wild west open source version

         - Official React/Typescript SDK, as well as wide runtime support ([python](https://github.com/get-convex/convex-py), rust, swift, etc)

         - Convex backend is a combination of rust crates and typescript modules, exposes an HTTP API for clients.

         - Not sure if the non-TS runtimes support all the Functions stuff

         - Local database, optimistic concurrency, Table and Document objects

         - Functions: 

            > Functions run on the backend and are written in JavaScript (or TypeScript). They are automatically available as APIs accessed through [client libraries](https://docs.convex.dev/client/react).

            - **Queries** (reactive, realtime), **Mutations** (transactional), **Actions** (web service calls etc)

         > Convex is the sync platform that replaces your backend and client state management.

         - [Philosophy](https://stack.convex.dev/the-software-defined-database#introducing-convex-components): The Software-defined Database

         - [Docs](https://docs.convex.dev/home), [Code](https://github.com/get-convex/convex-backend)

      + [DriftDB](https://driftdb.com) real-time data backend that runs on the edge

         - hosting in Cloudflare DO or Jamsocket

         - js/react client

         - API Server (Rust)

         - Worker impl in Rust using Cloudflare Durable Object persistence

         - API is websocket or http protocol

         - Not updated recently, unsure of status

         > DriftDB is a real-time data backend that runs on the edge … a keyed, ordered broadcast stream with replays and compaction.

         - Use cases

            > DriftDB is useful any time you wish you could connect directly from one browser to another, or from a non-web-connected backend (e.g. background task) to a browser.

            > DriftDB is intended for use cases where a relatively small number of clients need to share some state over a relatively short period of time. For example, it could be used to build a shared whiteboard, or act as a signaling server for WebRTC. Room state is currently persisted for 24 hours after the last write.

         - [Docs](https://driftdb.com/docs/data-model), [Code](https://github.com/jamsocket/driftdb)

      + [DXOS](https://dxos.org) build real-time, collaborative apps which run entirely on the client, and communicate peer-to-peer, without servers

         - Pivoted SDK effort to[ realtime collaborative editor ](https://dxos.org/composer/)product for now (12/2024)

         - TS/React client SDK, on pause for now; not ready for use

         - ECHO: realtime database with transparent data replication and conflict resolution

            > ECHO (The **E**ventually **C**onsistent **H**ierarchical **O**bject store) is a peer-to-peer graph database written in TypeScript.

         - HALO: keychain for identiity/auth

         - MESH: decentralized p2p protocol via WebRTC

         - [Code](https://github.com/dxos), [Interesting Architectures.md](./Interesting%20Architectures.md)

      + [Electric Clojure](https://github.com/hyperfiddle/electric) full-stack differential dataflow for UI TBD

         - Electric is a new way to build rich, interactive web products that simply have too much interactivity, realtime streaming, and too rich network connections to be able to write all the frontend/backend network plumbing by hand. With Electric, you can compose your client and server expressions directly (i.e. in the same function), and the Electric compiler macros will **infer at compile time the implied frontend/backend boundary** and generate the corresponding full-stack app.

      + [Evolu](https://www.evolu.dev) Local-First Platform Designed for Privacy, Ease of Use, and No Vendor Lock-In

         > Evolu Server is a simple message buffer and storage generic (the same) for all Evolu apps. Only UserId, NodeId, and messages timestamps are visible. Everything else is encrypted.

         - Typescript sdk

         - Ships Typescript backend, or BYO

         - Local SQLite with CRDT-based protocol to backend server which hosts a sqlite that only stores metadata

         - e2e encrypted

         - [Docs](https://www.evolu.dev/docs), [Code](https://github.com/evoluhq/evolu)

      + [Fireproof](https://fireproof.storage) Realtime database. Runs anywhere ($ hosted option)

         - Offers hosting

         - Fireproof is a decentralized realtime database that stores documents using prolly-trees.

         > Fireproof uses immutable data and distributed protocols to offer a new kind of database that:
         >
         > - can be embedded in any page or app, with a flexible data ownership model
         >
         > - can be hosted on any cloud
         >
         > - uses cryptographically verifiable protocols (what plants crave)

         - JS/TS client SDK

         - reactive using WebRTC

         - eventually consistent

         -  encrypted, immutable files for storage

         - Sync via ledger, using content-addressable archive (CAR) files in remote storage backends (including IPFS, PartyKit, S3)

         - CRDT for conflict resolution

         - [Docs](https://use-fireproof.com/docs/welcome), [Code](https://github.com/fireproof-storage/fireproof), [Interesting Architectures.md](./Interesting%20Architectures.md)

      + [Instant](https://www.instantdb.com) a modern Firebase … giving your frontend a real-time database ($ hosted option)

         - Hosting provider

         - Web client SDKs only (js, react etc)

         - Replication from local IndexedDB to postgres via a Clojure backend

         - [Docs](https://www.instantdb.com/docs)

      + [Jazz](https://jazz.tools) an open-source toolkit for building apps with *distributed state** *($ hosted option)

         - ts client

         - websocket protocol

         - local store sqlite or indexeddb

         - <https://jazz.tools/docs/react>

         - <https://github.com/garden-co/jazz>

      + [Kinto](https://kinto-storage.org) generic JSON document store with sharing and synchronisation capabilities from Mozilla

         > *Kinto* is a minimalist JSON storage service with synchronisation and sharing abilities. It is meant to be **easy to use** and **easy to self-host**.
         >
         > *Kinto* is used at Mozilla for Firefox Sync

         - Ships [JS client](https://kintojs.readthedocs.io/)

         - Local store is json documents. 

         - HTTP api

         - persists into postgres using python backend

         - [Docs](https://docs.kinto-storage.org/en/latest/tutorials/synchronisation.html#sync-implementations)

      + [Liveblocks](https://liveblocks.io) platform for adding collaborative editing, comments, and notifications into your application ($ hosted option)

         - Typescript

         - Mostly a components vendor that uses their APIs - commercial

         - Storage: Liveblocks Storage is a realtime sync engine designed for multiplayer creative tools such as Figma, Pitch, and Spline. `LiveList`, `LiveMap`, and `LiveObject` conflict-free data types can be used to build all sorts of multiplayer tools.

         - Yjs: Liveblocks Yjs is a realtime sync engine designed for building collaborative text editors such as Notion and Google Docs.

         - [Docs](https://liveblocks.io/docs), [Code](https://github.com/liveblocks/liveblocks/tree/main)

      + Livestore from jschickling, closed EA atm in Expo

      + [OctoBase](https://github.com/toeverything/OctoBase) light-weight, scalable, data engine written in Rust

         - [AFFiNE](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#1426264c-8933-464b-b60c-1d2619efee7f) data layer

         - CRDT via y-octo, compatible with blocksuite from AFFiNE

         - Storage layer-agnostic CRDT data and object store (sqlite, postgres, s3 wip)

         - Collaboration protocols (websocket, webrtc, libp2p wip)

         - js, java, swift clients (“bindings”)

         - it is immature and may have some lock-in.

      + [PartyKit](https://partykit.io) open source deployment platform for AI agents, multiplayer and local-first apps, games and websites from Cloudflare ($ hosted option)

         - Bought by Cloudflare, hosted in DO

         - web tech

         - Backend is a wasm module hosted in a durable object, in whatever wasm source lang desired (ie not python probably), implementing your “Party” biz logic

         - Can run an [yjs](https://github.com/yjs/yjs) server on the DO, opening up the possibility for a python app on the client side ([ypy](https://github.com/y-crdt/ypy))

      + Phoenix live view  TBD

         - Examples

            - <https://github.com/thisistonydang/liveview-svelte-pwa>

      + [Pocketbase](https://pocketbase.io) Open Source realtime backend in 1 file

         > Open Source realtime backend in 1 file with database, auth, files, and admin

         - js and dart client sdks

         - pen source Go backend that includes … embedded SQLite, reactivity, rest api

         - local embedded SQLite+WAL. Suggest using Litestream for replication if needed, and suggest fly.io for hosting

      + [Prisma Pulse](https://www.prisma.io/data-platform/pulse) distribute change events to your application at scale, enabling ($ commercial only)

         - From Prisma typescript ORM vendor, who is offering a managed Postgres (EA)

         - Works with BYO databases (postgres-based)

         - Not an open source product

      + [Redwood](https://github.com/redwood/redwood) a highly-configurable, distributed, realtime database and application server 

         - pre-alpha

         - TS client sdk

         - uses [Braid](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#a449217c-cbe3-44c4-8485-279ada569ebb) protocol (http), on an IETF [standards track](https://github.com/braid-org/braid-spec)

         - backend is golang

         - <https://github.com/redwood/redwood>

         > A highly-configurable, distributed, realtime database that manages a state tree shared among many peers.

      + [Replicache](https://replicache.dev) - client-side sync framework for arbitrary backends ✅\
         caps:\
         arch:\
         audience:\
         license: $ hosted option

         

         |  |  | 
         |---|---|
         |  |  | 
         

         > Replicache is a client-side sync framework for building realtime, collaborative web apps with zero-latency user interfaces. It works with most backend stacks.

         - Hosting + open interface and some open components, license required for use

         - Typescript SDK

         - Supports arbitrary backends, as long as they use a database that supports snapshot isolation. Ships a default Typescript backend, and defines an interface for building custom backends.

         - [Replicache sync protocol](https://doc.replicache.dev/byob/intro)

            - Replica data is persisted in the browser/client via XXXXX.

            - “Mutators”, applied optimistically (locally) and then authoritatively (remotely), transaction is completed locally when the remote acks the write.

            - “Subscriptions” at the query level, with reactivity in the client.

            - “Poke” interface for push/pull streaming

         - Requires a license, but is free for non-commercial. Closed source client, ships with default implementations of interfaces they’ve defined which you can use to BYO

         - [Docs](https://doc.replicache.dev)

      + [RxDB](https://rxdb.info) Store data locally to build high performance realtime applications that sync data with the backend and even work when offline ($ licensed functionality)

         - javascript sdk

         - uses a custom replication protocol that can support arbitrary backends, like hosted [postgres from supabase extension](https://github.com/marceljuenemann/rxdb-supabase)

         - has free edition but advanced features require a license

         - good for react native and mobile dev

      + [Scale](https://scale.sh/) A framework for building high-performance plugin systems into any application, powered by WebAssembly.

         - found them through a pycon columbia 2024 session

         - guest plugin runtimes: Go, rust, ts

         - host app runtimes: go, ts

         - Audience: embedding behavior into an application using wasm

         - <https://github.com/loopholelabs/scale>

      + [ShareDB](https://github.com/share/sharedb) Realtime database backend based on Operational Transformation (OT)

         - JS client SDK and backend

         - [can persist](https://share.github.io/sharedb/) to DBs like Mongo/Postgres

         - Protocol is OT-based

         - Designed for [DerbyJS](https://derbyjs.com) MVC framework

      + [SKDB](https://skdb.io) an embedded SQL database that stays in sync ($ hosted option)

         - TS client, reactive, conflict resolution, React client

         - Defines their own language sklang

         - sqlite is in there somewhere, it is opaque

         - Cloud offering in the works (2024-Dec)

         - [Docs](https://www.skdb.io/overview/), [Code](https://github.com/SkipLabs/skip)

      + [SQLSync](https://github.com/orbitinghail/sqlsync) a collaborative offline-first wrapper around SQLite prototype

         > SQLSync is a collaborative offline-first wrapper around SQLite
         >
         > *currently usable for prototypes*

         - js/ts client using sqlsync-worker, “reducer” in rust (wasm); ships React lib

         - Reactive query subscriptions

         - Rust (wasm) backend controlling persistence (demo in Cloudflare)

         - active development, very interesting maybe later.

         - Eventually consistent replicas using optimistic r/w 

         

      + [Supabase](https://supabase.com) an open source Firebase alternative ($ hosted option) ✅

         - Entire stack is open source and self-hostable

         - not local first

         - typescript services

         - docker or npm runtime

         - Postgres db, Postgres HTTP API wrapper, Realtime subscriptions, TS and PG Functions,File store, Vector embeddings, management dashboard, cli

         - <https://github.com/supabase/supabase> Main repo

         - [realtime-py](https://github.com/supabase/realtime-py) A Python Client for Phoenix Channels

         - [realtime](https://github.com/supabase/realtime) Broadcast, Presence, and Postgres Changes via WebSockets

         - [postgres-meta](https://github.com/supabase/postgres-meta) A RESTful API for managing your Postgres

         - [HarryET/supa-manager: Manage self-hosted Supabase instances with an easy to use API & Web Portal (soon)](https://github.com/HarryET/supa-manager)

         hosted postgres; has a [Realtime product](https://github.com/supabase/realtime) that appears to be open source, and it has a [python SDK](https://supabase.com/docs/reference/python/subscribe) as well as js/ts. Realtime is Elixir, just like Electric

      + [TinyBase](https://tinybase.org) reactive data store for local-first apps ✅

         - JS/TS SDK, but has a lot of persistence architecture options.

         + Multiple persistence [architecture options with tinybase](https://tinybase.org/guides/the-basics/architectural-options/):

            > Here, the synchronizer server (which is coordinating messages between clients) *also* acts as a 'client' with an instance of TinyBase itself. This is most usefully then persisted to a server storage solution, such as SQLite, PostgreSQL, the file system, or a Cloudflare Durable Object.

            > Note that changes made to the database (outside of a `[Persister](https://tinybase.org/api/persisters/interfaces/persister/persister/)`) are picked up immediately if they are made via the same connection or library that it is using. If the database is being changed by another client, the `[Persister](https://tinybase.org/api/persisters/interfaces/persister/persister/)` needs to poll for changes. Hence both configuration types also contain an `autoLoadIntervalSeconds` property which indicates how often it should do that. This defaults to 1 second.

            

            | `[Sqlite3Persister](https://tinybase.org/api/persister-sqlite3/interfaces/persister/sqlite3persister/)` | SQLite in Node, via [sqlite3](https://github.com/TryGhost/node-sqlite3) | 
            |---|---|
            | `[SqliteWasmPersister](https://tinybase.org/api/persister-sqlite-wasm/interfaces/persister/sqlitewasmpersister/)` | SQLite in a browser, via [sqlite-wasm](https://github.com/tomayac/sqlite-wasm) | 
            | `[ExpoSqlitePersister](https://tinybase.org/api/persister-expo-sqlite/interfaces/persister/exposqlitepersister/)` | SQLite in React Native, via [expo-sqlite](https://github.com/expo/expo/tree/main/packages/expo-sqlite) | 
            | `[CrSqliteWasmPersister](https://tinybase.org/api/persister-cr-sqlite-wasm/interfaces/persister/crsqlitewasmpersister/)` | SQLite CRDTs, via [cr-sqlite-wasm](https://github.com/vlcn-io/cr-sqlite) | 
            | `[ElectricSqlPersister](https://tinybase.org/api/persister-electric-sql/interfaces/persister/electricsqlpersister/)` | Electric SQL, via [electric](https://github.com/electric-sql/electric) | 
            | `[LibSqlPersister](https://tinybase.org/api/persister-libsql/interfaces/persister/libsqlpersister/)` | LibSQL for Turso, via [libsql-client](https://github.com/tursodatabase/libsql-client-ts) | 
            | `[PowerSyncPersister](https://tinybase.org/api/persister-powersync/interfaces/persister/powersyncpersister/)` | PowerSync, via [powersync-sdk](https://github.com/powersync-ja/powersync-js) | 
            | `[PostgresPersister](https://tinybase.org/api/persister-postgres/interfaces/persister/postgrespersister/)` | PostgreSQL, via [postgres](https://github.com/porsager/postgres) | 
            | `[PglitePersister](https://tinybase.org/api/persister-pglite/interfaces/persister/pglitepersister/)` | PostgreSQL, via [PGlite](https://github.com/electric-sql/pglite) | 
            

            

            | `[YjsPersister](https://tinybase.org/api/persister-yjs/interfaces/persister/yjspersister/)` | Yjs CRDTs, via [yjs](https://github.com/yjs/yjs) | 
            |---|---|
            | `[AutomergePersister](https://tinybase.org/api/persister-automerge/interfaces/persister/automergepersister/)` | Automerge CRDTs, via [automerge-repo](https://github.com/automerge/automerge-repo) | 
            

            

            | `[SessionPersister](https://tinybase.org/api/persister-browser/interfaces/persister/sessionpersister/)` | Browser session storage | 
            |---|---|
            | `[LocalPersister](https://tinybase.org/api/persister-browser/interfaces/persister/localpersister/)` | Browser local storage | 
            | `[FilePersister](https://tinybase.org/api/persister-file/interfaces/persister/filepersister/)` | Local file | 
            | `[IndexedDbPersister](https://tinybase.org/api/persister-indexed-db/interfaces/persister/indexeddbpersister/)` | Browser IndexedDB | 
            | `[RemotePersister](https://tinybase.org/api/persister-remote/interfaces/persister/remotepersister/)` | Remote server | 
            

         - Has a local Store (JSON-serialized, supports transactions), which can be optionally wired to some Persister that’s local (browser storage, WASM dbms etc) or remote (server dbms etc).

         - Remote persistence uses websockets, requires a backend (ships a few defaults, one using Cloudflare Durable Objects)

         - The local Store can be a MergeableStore, that together with a Synchronizer can provide client-client or client-server replication. Ships with a default CRDT Merge/Sync impl, but supports third-party sync using a custom Persister (Yjs, Automerge). 

         - Persisters by default only receive changes made in-process (other clients of the Persister), but can be set to poll in order to receive changes made to the database by other clients.

         - [Docs](https://tinybase.org/api/synchronizer-ws-server-durable-object/)

      + [Triplit](https://www.triplit.dev) an open-source database that syncs data between server and browser in real-time ($ hosted option)

         - js/ts SDK

         - has an [open source server impl](https://www.triplit.dev/docs/self-hosting) with pluggable storage engines like sqlite and [leveldb](https://github.com/google/leveldb). 

         - Protocol is http

         - Ships default JS impl

         - Cloud hosting relationship with Railway

         - offline mode, rather than localfirst; clients do optimistic writes and are considered a cache

      + [Verdant](https://verdant.dev) a framework and philosophy for small, sustainable, human web apps

         - js/ts only for the clients

         - supports syncing to one or more s[erver-side ](https://verdant.dev/docs/sync/storage)sqlite databases. 

         - Source can [be IndexedDB](https://github.com/a-type/verdant?tab=readme-ov-file)

         - [js SDK is reactive](https://github.com/a-type/verdant?tab=readme-ov-file)

         - [Docs](https://verdant.dev/docs/intro), [Code](https://github.com/a-type/verdant)

+ ## Source-to-Source Replication Architectures

   *These components might be (an in some cases are) used as part of a data layer for a Sync Platform.*

   + ### Realtime Cache and Proxy Components

      + [ReadySet](https://readyset.io/docs/get-started) transparent cache (postgres, mysql)

         - a transparent database cache for Postgres & MySQL that gives you the performance and scalability of an in-memory key-value store without requiring that you rewrite your app or manually handle cache invalidation

         - it keeps cached query results in sync with your database automatically by utilizing your database’s replication stream. It is wire-compatible with Postgres and MySQL and can be used along with your current ORM or database client.

         - rust

         - docker, linux binary

         - Integrates with [Supabase](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#96d39a55-56c4-4dc2-aa07-71cd79e6e2a4) via an extension

         - <https://github.com/readysettech/readyset>

   + ### Source-to-Local Replicas

      *To local embedded replicas (PGLite, SQLite/LibSQL)*

      + [Electric SQL](https://electric-sql.com) + [PGLite](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#3e97a6ec-8ff1-49b1-8c91-3250d67351a4)/SQlite syncs [little subsets](https://electric-sql.com/docs/guides/shapes) of your Postgres data into local apps and services ($ hosted option) ✅

         - TS and Elixir client SDKs

         - Backend in Elixir, fronting a Postgres instance

         - Electric Protocol over HTTP, sync and stream

         - Local database is SQLite in wasm; PGLite is supported [but only for local reads](https://pglite.dev/docs/sync) atm (Dec-2024)

         - [Docs](https://electric-sql.com/docs/intro), [Code](https://github.com/electric-sql/electric)

      + [Turso](https://turso.tech) LibSQL ($ hosted option) ✅

         - Wide client runtime support

         - LibSQL client-side

         - LibSQL Server (sqld) source SQLite

         - Edge hosted option

         - [Docs](https://docs.turso.tech/introduction), Code ([libsql](https://github.com/tursodatabase/libsql?tab=readme-ov-file))

      + [Powersync](https://www.powersync.co) a service and set of client SDKs that keeps backend databases in sync with on-device embedded SQLite databases ($ hosted option)

         - Hosted

         - js, react, flutter SDKs

         - postgres db source

         - typescript backend uses WAL, depends on mongodb internally

         - local replica is sqlite (maybe wasm?)

         - [Docs](https://docs.powersync.com/intro/powersync-overview), [Code](https://github.com/powersync-ja)

      + [SQLedge](https://github.com/zknill/sqledge) Replicate postgres to SQLite on the edge (updated fork [pgreplsql](https://github.com/gemini-kenshi/pgreplsql))

         - state: alpha, no commits since 2023

         - written in Go

         > SQLedge uses Postgres logical replication to stream the changes in a source Postgres database to a SQLite database that can run on the edge. SQLedge serves reads from its local SQLite database, and forwards writes to the upstream Postgres server that it's replicating from
         >
         > ![Pasted 2024-12-05-15-29-42.png](./Database%20Sync%20(localfirst)-assets/Pasted%202024-12-05-15-29-42.png)

      ### LibSQL Ecosystem

      - [Turso](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#25adda0f-1212-47c5-a5bd-09e703df3799)

      - [AstroDB](https://docs.astro.build/en/guides/astro-db/) fully-managed SQL database designed for the Astro ecosystem

   + ### Source-to-$X Connectors

      - PipeXXX??

      + [Sequin](https://sequin.io) Postgres CDC to Sink ($ hosted offering, paywall features)

         - Replicates postgres changes to Kafka, SQS, Redis, Webhook, HTTP Pull, and others

         - Elixir

         > Sequin is a tool for change data capture (CDC) in Postgres. Sequin makes it easy to stream Postgres rows and changes to streaming platforms and queues (e.g. Kafka and SQS). You can backfill existing rows and stream new changes in real-time.
         >
         > Sequin even supports native sinks (HTTP GET and webhooks), so you can get started without any other infrastructure.

         - <https://github.com/sequinstream/sequin>

         - <https://sequin.io/docs/sync-process/overview>

         - Hosted offering supports more databases and other connectors

   + ### SQLite Disaster Recovery

      + [Litestream](https://litestream.io) Fully-replicated database with no pain and little cost

         > Litestream is a streaming replication tool for SQLite databases. It runs as a separate background process and continuously copies write-ahead log pages from disk to one or more replicas

         - Consumes WAL, writes to an S3-compatible object storage such as [Tigris](https://www.tigrisdata.com/).

         - [Code](https://github.com/benbjohnson/litestream)

   + ### Clustered SQLite

      #### Multi-master

      + [Vlcn](https://vlcn.io) / [cr-sqlite](https://github.com/vlcn-io/cr-sqlite) Convergent, Replicated SQLite. Multi-writer and CRDT support for SQLite  ✅ [how-to](https://observablehq.com/@tantaman/cr-sqlite-basic-setup)

         - SQLite extension

         node, deno, browser (wasm), python, c, rust; CR at the table layer (“[CRR](https://vlcn.io/docs/appendix/crr)”), future to allow partial sync

         [simonw TIL](https://til.simonwillison.net/sqlite/cr-sqlite-macos)

         > CR-SQLite is a ++[run-time loadable extension(opens in a new tab)](https://www.sqlite.org/loadext.html)++ for ++[SQLite(opens in a new tab)](https://www.sqlite.org/index.html)++ and ++[libSQL(opens in a new tab)](https://github.com/libsql/libsql)++. It allows merging different SQLite databases together that have taken independent writes.

         >  cr-sqlite adds multi-master replication and partition tolerance to SQLite via conflict free replicated data types (++[CRDTs(opens in a new tab)](https://en.wikipedia.org/wiki/Conflict-free_replicated_data_type)++) and/or causally ordered event logs.

         > `cr-sqlite` is distributed as a ++[run time loadable extension(opens in a new tab)](https://www.sqlite.org/loadext.html)++ for SQLite and can be used with any language that has SQLite bindings.

         > ++[cr-sqlite](https://vlcn.io/docs/cr-sqlite/intro)++ is network agnostic. You can use ++[crsql_changes](https://vlcn.io/docs/cr-sqlite/api-methods/crsql_changes)++ to pull changes from one database, ship them over any protocol you like, then use ++[crsql_changes](https://vlcn.io/docs/cr-sqlite/api-methods/crsql_changes)++ on the other end to apply the changes. In any case, a WebSocket implementation is provided for convenience ++[here](https://vlcn.io/docs/cr-sqlite/js/networking)++ and an overview of writing your own network code ++[here](https://vlcn.io/docs/cr-sqlite/networking/whole-crr-sync)++.

      + [IceFireDB](https://www.icefiredb.xyz/)  a DB storage and retrieval protocol built for web3.0

         - SQLite, SQLProxy, NoSQL, RedisProxy, and PubSub impls

         - SQLite impl: IceFireDB SQLite is a decentralized SQLite database designed to facilitate the construction of a global distributed database system. It allows users to write data to IceFireDB using the MySQL protocol. The data is stored in an SQLite database and automatically synchronized among nodes through P2P networking.

         - <https://github.com/IceFireDB/IceFireDB>

      + Expo SQLite

         - <https://github.com/expo/expo/tree/main/packages/expo-sqlite>

      #### Single master

      + [LiteFS](https://github.com/superfly/litefs) FUSE-based file system for replicating SQLite databases across a cluster of machines from fly.io

         - [Docs](https://fly.io/docs/litefs/), [Code](https://github.com/superfly/litefs), fly.io project

         - distributed file system that transparently replicates SQLite databases

         - Fly.io recommends Turso for hosting

         > FUSE file system: intercepts file system calls to record transactions.
         >
         > Leader election: currently implemented by ++[Consul](https://www.consul.io/)++ using sessions
         >
         > HTTP server: provides an API for replica nodes to receive changes.

      + [rqlite](https://github.com/rqlite/rqlite) lightweight, user-friendly, distributed relational database built on SQLite

         Go server impl, wide client lang support. Uses Raft for cluster consensus; each node can recv writes, but they will be forwarded to the leader synchronously to perform the actual write and then wait for consensus (latency considerations); reads might be serviceable by followers, given the consistency settings. sqlite is just a storage engine. protocol is HTTP. [FAQ](https://rqlite.io/docs/faq/), [sqlalchemy engine](https://github.com/rqlite/sqlalchemy-rqlite)

         All writes go to the Raft leader (rqlite makes sure this happens transparently if you don't initially contact the Leader node)

         > rqlite is a distributed relational database that combines the simplicity of SQLite with the robustness of a fault-tolerant, highly available system. It's developer-friendly, its operation is straightforward, and it's designed for reliability with minimal complexity.

      + [dqlite](https://dqlite.io) a fast, embedded, persistent SQL database with Raft consensus that is perfect for fault-tolerant IoT and Edge devices

         - Canonical project ([source](https://github.com/canonical/dqlite)), another raft consensus sqlite cluster

      - ++[ChiselStore](https://github.com/chiselstrike/chiselstore)++ - ChiselStore is an embeddable, distributed SQLite for Rust, powered by Little Raft.

      - ++[ha-sqlite](https://github.com/uglyer/ha-sqlite)++ - High-availability sqlite database service based on raft.

      - ++[raft-sqlite](https://github.com/shettyh/raft-sqlite)++ - Raft backend using SQLite.

      - ++[ReSqlite](https://github.com/jervisfm/resqlite)++ - ReSqlite is an extension of Sqlite that aims to add basic replication functionality to Sqlite database.

      - ++[tqlite](https://github.com/minghsu0107/tqlite)++ - Distributed SQL database with replication, fault-tolerance, tunable consistency and leader election.

   + ### Clustered Postgres

      #### Multi-master Postgres

      + [pgEdge](https://www.pgedge.com) distributed PostgreSQL, optimized for the network edge ($ hosted option)

         - uses spock, not sure how the replica works

         - [Code](https://github.com/pgEdge/pgedge)

      + [EDB BDR](https://www.enterprisedb.com/docs/pgd/4/bdr) PostgreSQL extension providing multi-master replication and data distribution with advanced conflict management ($ commercial?)

         - Commercial product, but seem to be a few docker containers that use it on github

         > BDR is a PostgreSQL extension providing multi-master replication and data distribution with advanced conflict management, data-loss protection, and [throughput up to 5X faster than native logical replication](https://www.enterprisedb.com/blog/performance-improvements-edb-postgres-distributed), and enables distributed Postgres clusters with high availability up to five 9s.

      + [EDB PGD](https://www.enterprisedb.com/docs/pgd/latest/overview/) multi-master replication and data distribution with advanced conflict management ($ commercial?) fork of pglogical

         > PGD provides loosely coupled, multimaster logical replication using a mesh topology. 

      + [pglogical](https://github.com/2ndQuadrant/pglogical) extension provides logical streaming replication for PostgreSQL, using a publish/subscribe model

         - Based on BDR

         > While pglogical is actively maintained, EnterpriseDB (which acquired 2ndQuadrant in 2020) focuses new feature development on a descendant of pglogical: ++[Postgres Distributed](https://www.enterprisedb.com/docs/pgd/latest/overview/)++

      #### Postgres Streaming

      + pgstream

         - replicates schema changes as well as data

            Go binary that monitors WAL, support buffering WAL to kafka for scaling

            dist by [xata.io](xata.io), a serverless postgres vendor

            pgstream and wal2json to emit webhooks on CDC events: <https://xata.io/blog/postgres-webhooks-with-pgstream>

      + debezium

         ### Debezium

         java+kafka stack

      #### HA Postgres

      PostgreSQL HA [documentation](https://wiki.postgresql.org/wiki/Replication,\_Clustering,\_and_Connection_Pooling)

      - [Citus](https://www.citusdata.com) Distributed PostgreSQL as an extension

      + [TiDB](https://pingcap.com/) open-source, cloud-native, distributed SQL database ($ hosted option)

         - horizontal scaled with two-phase commit distributed transactions

         - clustering via raft

         - mysql interface

         - golang

         - docker, k8s operator

         - Apache2.0

         - <https://github.com/pingcap/tidb>

      + [Patroni](https://patroni.readthedocs.io/) A template for PostgreSQL High Availability with Etcd, Consul, ZooKeeper, or Kubernetes

         - <https://github.com/patroni/patroni>

         - Can integrate their HA with a [Citus](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#79eed1e8-3b26-480f-a1a6-17c362f4bf5a) distributed cluster

   + ### Embedded replicated databases (in-browser/wasm)

      + [PGLite](https://pglite.dev) Run a full Postgres database locally in WASM with reactivity and live sync from ElectricSQL

         - Postgres in WASM

         - Can integrate with ElectricSQL for reads, writes being worked on

         - IndexedDB and OPFS storage 

         - [Code](https://github.com/electric-sql/pglite)

   + ### Embedded replicated databases (in-process)

      + [PoloDB](https://www.polodb.org/) is an embedded document database.

         - PoloDB is a library written in Rust that implements a lightweight ++[MongoDB](https://www.mongodb.com/)++.

         - <https://github.com/PoloDB/PoloDB>

         - <https://www.polodb.org/docs>

      - [Pouch](https://pouchdb.com) very cool, couchdb implementation for the browser. awesome sync, but targets web runtime. Supposedly can replicate with other similar impls like Couchbase

      + SurrealDB

         - Build real-time apps faster with the world's most powerful multi-model database

         - Embeds in the browser, and has the tits features

         - EXCEPT for replication

         - capabilities: multi-model

         - <https://github.com/surrealdb/surrealdb.wasm>

         - <https://surrealdb.com/features>

            #### **Versioned temporal tables**

            IN DEVELOPMENT

            \
            Local (wasm) and remote, lots of crazy features

            Has graph and vector and unstructured 

      + Bolt DB ([forks](https://go.etcd.io/bbolt))

         - An embedded key/value database for Go.

         - Ben Johnson’s original project before moving on to fly/LiteFS/etc

         - <https://github.com/etcd-io/bbolt>

      + [chDB](https://clickhouse.com/docs/en/chdb) a fast in-process SQL OLAP Engine powered by [ClickHouse](https://github.com/clickhouse/clickhouse).

         - You can use it when you want to get the power of ClickHouse in a programming language without needing to connect to a ClickHouse server.

         - client runtimes: python, go, rust, nodejs, bun

         - <https://clickhouse.com/docs/en/chdb>

   + ### Standalone replicated databases

      + [Cockroach](https://www.cockroachlabs.com/product/) cloud native, distributed SQL database designed for high availability, effortless scale, and control over data placement ($ hosted option) new mandatory telemetry for free version

         - supports postgres wire protocol

         - multi-master

         - no wasm

         - [Hosted cloud](https://www.cockroachlabs.com/)

         - consensus based

         + LLM summary

            Yes, CockroachDB is a multi-master distributed database. It uses a distributed consensus protocol (an extended version of Raft) to maintain consistency across multiple nodes, where any node can accept both read and write operations.

            Key characteristics:

            1. True multi-master architecture

            - Every node can handle reads and writes

            - No primary/secondary distinction

            - Automatic load balancing

            1. Consistency Model

            - Serializable isolation level

            - Strong consistency (CP in CAP theorem)

            - ACID transactions across nodes

            1. Replication

            - Automatic sharding

            - Range-based data distribution

            - Configurable replication factor

            1. Conflict Resolution

            - Timestamp-based conflict resolution

            - Uses hybrid logical clocks

            - Automatic handling of concurrent writes

            This differs from traditional master-slave or primary-replica architectures where writes must go through a single master node. In CockroachDB, any node can accept writes, and the system handles synchronization and conflict resolution automatically through its consensus protocol.

            The multi-master capability is particularly useful for:

            - Geographic distribution

            - High availability

            - Write scalability

            - Disaster recovery

            - Regional compliance requirements

         - [Docs](https://www.cockroachlabs.com/docs/), [Code](https://github.com/cockroachdb/cockroach?tab=readme-ov-file)

      + [AntidoteDB](https://www.antidotedb.eu) A planet scale, highly available, transactional database built on CRDT technology ($ hosted option)

         - AntidoteDB is a highly available geo-replicated key-value database. AntidoteDB provides features that help programmers to write correct applications while having the same performance and horizontal scalability as AP/NoSQL databases. Furthermore, AntidoteDB operations are based on the principle of synchronization-free execution by using Conflict-free replicated datatypes (*CRDTs*).

         - [Code](https://github.com/AntidoteDB/antidote)

      + [Fauna](https://fauna.com) true serverless database ($ commercial only)

         > true serverless database that combines document flexibility with relational power, automatic scaling, and zero operational overhead.

      - [Firebase (Google)](https://firebase.google.com) ($ commercial only)

      + [Kuzu DB](https://kuzudb.com/) graph database ✅

         - Embeddable property graph database management system built for query speed and scalability. Implements Cypher.

         - Python sdk, in addition to others

         - <https://github.com/kuzudb/kuzu>

      + [CouchDB](https://couchdb.apache.org/) and [Couchbase](https://docs.couchbase.com/home/index.html)

         - Multi-master, and we have PouchDB and Couchbase-lite/sync-gateway

         - Seamless multi-master syncing database with an intuitive HTTP/JSON API

         - <https://github.com/apache/couchdb>

         - <https://docs.couchdb.org/>

      + [Defra DB](https://docs.source.network/) Peer-to-Peer Edge Database

         - ++[MerkleCRDTs](https://arxiv.org/pdf/2004.00107.pdf)++ and the content-addressability of ++[IPLD](https://docs.ipld.io/)++

         - It's the core data storage system for the Source Network Ecosystem, built with IPLD, LibP2P, CRDTs, and Semantic open web properties.

         - <https://github.com/sourcenetwork/defradb>

         - audience: decentralized web

      + Chotki An LSM database turned a CRDT database

         - audience: dec-web, blockchain

         - Internally, it is ++[pebble db](https://github.com/cockroachdb/pebble)++running CRDT natively, using the ++[Replicated Data Interchange](https://github.com/drpcorg/chotki/blob/main/rdx)++ format (RDX). Chotki is sync-centric and causally consistent. That means, Chotki replicas can sync master-master, either incrementally in real-time or stay offline to diff-sync periodically. Chotki API is REST/object based, no SQL yet.

         - <https://github.com/drpcorg/chotki>

         - written in Go

      + Clickhouse TBD

         - [clickhouse-local](https://clickhouse.com/docs/en/chdb/guides/clickhouse-local) cli with embedded db

      + [FoundationDB](https://www.foundationdb.org/) (Apple)

         - caps: multi-model, single master cluster

         - FoundationDB is a distributed database designed to handle large volumes of structured data across clusters of commodity servers. It organizes data as an ordered key-value store and employs ACID transactions for all operations.

         - <https://github.com/apple/foundationdb/>

         - <https://apple.github.io/foundationdb/>

      + [TigerBeetle](https://tigerbeetle.com) The financial transactions database designed for mission critical safety and performance.

         - TigerBeetle is an Online Transaction Processing (OLTP) database built for safety and performance. It is not a general purpose database like PostgreSQL or MySQL. Instead, TigerBeetle works alongside your general purpose database, which we refer to as an Online General Purpose (OLGP) database.

         - Clustered replicas

         - <https://docs.tigerbeetle.com/coding/system-architecture/>

         - [Interesting Architectures.md](./Interesting%20Architectures.md)

      + [greptimedb](https://greptime.com/) An open-source, cloud-native, unified time series database for metrics, logs and events with SQL/PromQL supported ($ hosted option)

         - **GreptimeDB** is an open-source unified & cost-effective time-series database for **Metrics**, **Logs**, and **Events** (also **Traces** in plan).

         - Built for k8s, Single-master

         + LLM

            GrepTimeDB uses a multi-layer architecture but is not fully multi-master in the same way as CockroachDB. It follows a primary-secondary model for its distributed architecture.

            Architecture layers:

            1. Frontend tier

            - Handles query requests

            - Protocol adaptation

            - Query routing

            1. Meta tier (primary-secondary)

            - Metadata management

            - Cluster coordination

            - Uses etcd for consensus

            1. Datanode tier

            - Data storage and computation

            - Can have multiple datanodes

            - Handles read operations

            While multiple nodes can handle reads, write operations are coordinated through primary nodes. It's designed more as a time-series database with distributed capabilities rather than a true multi-master system.

            Key characteristics:

            - Write operations have defined routing paths

            - Read operations can be distributed

            - Uses etcd for metadata consensus

            - Optimized for time-series data

            - Regional distribution possible

            - Eventual consistency model

            This differs from true multi-master systems like CockroachDB where any node can handle any operation with full consistency guarantees.

         - <https://github.com/GreptimeTeam/greptimedb>

      #### For ETL workloads

      + [Artie](https://www.artie.com)

         - arch: (source db) reader, kafka, transfer (local db)

         - lang: golang

         - docker, go binary

         - Various OLTP sources (Postgres, MongoDB etc) and OLAP destinations (Snowflake, s3 etc)

         - No sqlite

         - <https://artie.com/docs/quickstart>

         - <https://github.com/artie-labs>

      + [Mycelial](https://mycelial.com)

         - new, and used for ETL workloads

+ ## Local and Source Databases

   ### Hosted/Edge SQLite

   - Cloudflare [D1](https://blog.cloudflare.com/introducing-d1/)

   - [Turso](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#25adda0f-1212-47c5-a5bd-09e703df3799)

   ### Hosted Postgres

   - [Supabase](https://supabase.com) Postgres (and others) ($ commercial)

   - Neon hosted postgres ($ commercial)

   - Railway ($ commercial)

   - [Fly Postgres](https://fly.io/docs/postgres)  ($ commercial)

   ### In-process and/or in-browser (wasm) Databases

   - SQlite and wa-sqlite

   + [Duckdb](http://www.duckdb.org/) DuckDB is an analytical in-process SQL database management system ($ hosting [option](https://motherduck.com))

      - Supports WASM runtime

      - [Docs](https://duckdb.org/docs/), [Code](https://github.com/duckdb/duckdb)

+ ## Realtime Libraries and Components

   ### Realtime UI Components

   - [DXOS](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#304df122-5e13-4773-a4b4-3ba889667ed1) Editor

   - [Liveblocks](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#9cc728d5-ccbd-4ecd-baf1-8b1309429e50) components

   - [Blocksuite](https://github.com/toeverything/blocksuite) ([AFFiNE](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01e5ab5e-b51f-401e-aabb-8b2c19fc9f1a#1426264c-8933-464b-b60c-1d2619efee7f))

   ### Sync Merge and Conflict Management

   - [Yjs](https://yjs.dev) A CRDT framework with a powerful abstraction of shared data

   - [Automerge](https://automerge.org) a library of data structures for building collaborative applications

   + [Loro](https://loro.dev/docs) Make your JSON data collaborative and version-controlled with CRDTs 

      - Rust, JS (via WASM), and Swift

      - [ rich text CRDT](https://loro.dev/blog/loro-richtext)

      - hierarchy data with movable tree algo

   + [Diamond Types](https://github.com/josephg/diamond-types) The world's fastest CRDT. WIP.

      - rust, js (wasm)

   + Various Dart implementations of CRDTs and SQL databases

      - [sqlite_crdt](https://github.com/cachapa/sqlite_crdt) , [sql_crdt](https://github.com/cachapa/sql_crdt), [postgres_crdt](https://github.com/cachapa/postgres_crdt), others

      - Targets mobile

      + [Watermelon](https://nozbe.github.io/WatermelonDB) reactive database framework

         - iOS, Android, Windows, web, and Node.js; React/React-Native but framework-agnostic; reactive via RxJS

         - Local sqlite

         - Watermelon sync protocol, pull and push

         - BYO backend and source db, existing impls in TS ([firemelon](https://github.com/Ali-Nahhas/firemelon)), [Elixir](https://fahri.id/posts/how-to-build-watermelondb-sync-backend-in-elixir/), Laravel ([ew](https://github.com/nathanheffley/laravel-watermelon))

   ### Messaging protocols or frameworks

   - [NATS](https://nats.io) Connective Technology for Adaptive Edge & Distributed Systems

   - [Socket.io](https://socket.io) Realtime application framework <https://github.com/socketio/socket.io>

   - [Deepstream](https://deepstream.io) a fast and secure data-sync realtime server \
      for mobile, web & iot <https://github.com/deepstreamIO>

   - [Nchan](https://nchan.io) (Nginx) Fast, horizontally scalable, multiprocess pub/sub queuing server and proxy for HTTP, long-polling, Websockets and EventSource (SSE) <https://github.com/slact/nchan>

   ### Peer to Peer/Decentralized Systems 

   + [Gun](https://gun.eco) An open source cybersecurity protocol for syncing decentralized graph data

      - Javascript impl

      - audience: p2p, dec

      - caps: e2ee

      - <https://gun.eco/docs/API>

      - <https://github.com/amark/gun>

      > **GUN** is an ++[ecosystem](https://gun.eco/docs/Ecosystem)++ of **tools** that let you build ++[community run](https://www.nbcnews.com/tech/tech-news/these-technologists-think-internet-broken-so-they-re-building-another-n1030136)++ and ++[encrypted applications](https://gun.eco/docs/Cartoon-Cryptography)++ - like an Open Source Firebase or a Decentralized Dropbox.

      > The ++[Internet Archive](https://news.ycombinator.com/item?id=17685682)++ and ++[100s of other apps](https://github.com/amark/gun/wiki/awesome-gun)++ run GUN in-production.\
      > Multiplayer by default with realtime p2p state synchronization!
      >
      > - Graph data lets you use key/value, tables, documents, videos, & more!
      >
      > - Local-first, offline, and decentralized with end-to-end encryption.

      - e2e encryption

   + [Holepunch](https://holepunch.to) Pears Holepunch is a platform for creating apps that don’t use any servers whatsoever

      - pretty interesting; [massively](https://github.com/holepunchto/pear) [distributed](https://github.com/holepunchto/hyperdb/blob/4316af36cec9b3dced412fb39b3a4ca1df7489d8/lib/engine/rocks.js) [shit](https://docs.pears.com/building-blocks/hyperbee), not for me for now.

      - <https://docs.pears.com>

      - js runtime

      > Pear by Holepunch is a combined Peer-to-Peer (P2P) Runtime, Development & Deployment tool.

   + [OrbitDB](https://orbitdb.org) Peer-to-Peer Databases for the Decentralized Web

      - too much tuna: [Interesting Architectures.md](./Interesting%20Architectures.md) IPFS based

   ### WASM

   + [SQL.js](https://github.com/sql-js/sql.js) A javascript library to run SQLite on the web.

      - uses ++[emscripten](https://emscripten.org/docs/introducing_emscripten/about_emscripten.html)++ to compile ++[SQLite](http://sqlite.org/about.html)++ to webassembly

      - uses a ++[virtual database file stored in memory](https://emscripten.org/docs/porting/files/file_systems_overview.html)++, and thus **doesn't persist the changes** made to the database; However, it allows you to **import** any existing sqlite file, and to **export** the created database as a ++[JavaScript typed array](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Typed_arrays)++.

      - <https://github.com/sql-js/sql.js>

   + SQLite, wasm, WebRTC (PoC)

      - Private peer-to-peer collaborative databases in the browser using SQLite (WebAssembly) and WebRTC.

      - <https://github.com/adhamsalama/sqlite-wasm-webrtc>

   + [jawsm](https://github.com/drogus/jawsm) JavaScript to WASM compiler (experimental)

      - <https://github.com/drogus/jawsm>

   + [WasmEdge](https://wasmedge.org/book/en/) Runtime

      - For compiled languages (e.g., C and Rust), WasmEdge WebAssembly provides a safe, secure, isolated, and containerized runtime as opposed to Native Client (NaCl).

      - For interpreted or managed languages (e.g., JavaScript and Python), WasmEdge WebAssembly provides a secure, fast, lightweight, and containerized runtime instead of Docker + guest OS + native interpreter.

      - WASM VM

      - A cloud native WebAssembly runtime. A secure, lightweight, portable and high-performance alternative to Linux containers.

      - Lightweight database clients in the WasmEdge Runtime:

         - <https://github.com/WasmEdge/wasmedge-db-examples>

   

+ ## Communities, Projects, Resources

   - [lofi.software](https://lofi.software)

   - [localfirst.fm](https://www.localfirst.fm)

   - [Local-first Conf](https://www.localfirstconf.com)

   - [crdt.tech](https://crdt.tech)

   - [SyncFree](https://pages.lip6.fr/syncfree/index.php/crdt-resources.html)

   - [Hydro Project](https://hydro.run/research)

   - [Ink & Switch](https://www.inkandswitch.com)

   - [Dat-ecosystem](https://blog.dat-ecosystem.org/staying-connected)

   - Source Network Own the internet of the future, with decentralized tools and infrastructure that prioritize data interoperability, privacy preservation, and trustless ownership

      - <https://github.com/sourcenetwork>

      - <https://docs.source.network>

      - Various components like DefraDB, some graphql, a blockchain etc

   - [AFFiNE](https://github.com/toeverything) project

   - [Braid](https://braid.org) lots of useful low level stuff

      > Braid: Working Group for Interoperable State Synchronization

   - <https://github.com/tablelandnetwork/awesome-decentralized-database>

## Articles

- <https://jakelazaroff.com/words/building-a-collaborative-pixel-art-editor-with-crdts/>

+ ## Graveyard

   - [RemoteStorage](https://remotestorage.io) old AF

   - RhizomeDB DEAD

   - [Statecraft](https://github.com/josephg/statecraft?tab=readme-ov-file) dead?

   - [synQL](https://github.com/coast-team/synql) not updated in a while

   + [CRStore](https://github.com/Azarattum/CRStore) js only, built on cr-sqlite, not much movement lately. one guy

      > Conflict-free replicated store.

# in

algorithms like egwalker have access to all the information they need to do that. We store character-by-character editing traces from all users. And we store when all changes happened (in causal order, like a git DAG). This is far more information than git has. So it should be very possible to build a CRDT which uses this information to detects & mark conflict ranges when branches are merged. Then we can allow users to manually resolve conflicts.

<https://github.com/PoloDB/PoloDB>

Goat 

<https://goatdb.com/tutorial/>

<https://rxdb.info/alternatives.html>