# LifeOS Lookup Technical Research

## Requirements

1. My data cant be locked away, there needs to be a central database accessible by other workloads

2. The mobile app should work offline

3. The mobile app must be able to read and write from that central server; if it is offline-capable, this implies the mobile app have a read/write replica of some part of the database

4. Updates to the database must sync back to the mobile client, other clients must not be required to integrate with a sync service to make this happen

5. The mobile app must receive updates when they happen, so that it gets new data. The mobile app does **not** need to have “live UI updates”.

### Technical Requirements

- Supports React Native, preferably via Expo Go (mobile app)

- Central database, in target state

- Realtime subscriptions to the mobile app from central db, in the target state

## Analysis

[LifeOS Technical Architecture Options.md](./LifeOS%20Technical%20Architecture%20Options.md)

[LifeOS Lookup App Roadmap.md](./LifeOS%20Lookup%20App%20Roadmap.md)

## Ideas

- Syncing ElectricSQL into a Durable Object

   - <https://github.com/KyleAMathews/electric-demo-cloudflare-sqlite/tree/main>

## Out of Scope

### PowerSync

Has great stuff, I might want to use it later when there are a bunch of systems that all need the same data.

<https://docs.powersync.com/architecture/architecture-overview>

It also needs a MongoDB, queues, its a good “production” system if I want that.

### Livestore

ts only

### CR-Sqlite

<https://vlcn.io/docs/cr-sqlite/quickstart>

Only has js/ts/rust client libraries

Is more of a lower level abstraction, I would need to build all the things

### CRDTs

I do not want to have to build infrastructure, I want to just move fast