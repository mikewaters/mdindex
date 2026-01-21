# Backend Data Frameworks

### Supabase

Supported by ElectricSQL (database)

Alternative to ElectricSQL (realtime)

- [ ] TBD get details from the other doc [Database Sync (localfirst).md](./Database%20Sync%20\(localfirst\).md)

### Trailbase

Alternative to ElectricSQL (realtime sync)

Not supported by ElectricSQL (database - local sqlite)

> An open, blazingly fast, single-executable Firebase alternative with type-safe REST & realtime APIs, built-in JS/ES6/TS runtime, SSR, auth and admin UI built on Rust, SQLite & V8.
>
> <https://trailbase.io/getting-started/install>
>
> <https://github.com/trailbaseio/trailbase>

Properties:

- Open source

- No commercial options

- Rustlang

- Deployment: Single binary or docker container

- Great Admin UI

- Has a data CLI for scripting

- Official supported backend for [TanStack DB](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/35cde992-64ef-479e-a8d1-e7a234b24543#eee30700-dba8-43e3-933e-9ab1253a729d) ([code examples](https://github.com/trailbaseio/trailbase/tree/41427d3da064fcaa90fdab64c702d8b551d70f75/examples/tanstack-db-sync))

Stack elements:

1. Database (SQLite)

2. Data Platform

3. API and Realtime Sync

4. Client SDKs

#### Data layer

Uses a local SQLite database, and can be used on top of an existing database (you will need to create appropriate schema files).

#### Data Platform

Supports basic data accessors.

Allows you to inject your own code, as long as its typescript or javascript.

Allows you to inject behavior via rust as well

#### API with Realtime Sync

HTTP services for get and stream.

[Tutorial on realtime sync](https://trailbase.io/getting-started/first-realtime-app/) using a typescript server-side app

#### Client SDKs

Official JS/TS and Dart clients, also has other languages (python, go, swift)

<https://github.com/trailbaseio/trailbase/tree/41427d3da064fcaa90fdab64c702d8b551d70f75/client>

#### Eval

Would be a good initial option, and I could pivot later

- supports the mobile app stack (tanstack db)

- has realtime

- can work on top of an existing sqlite

- When I change the backend, I can switch the type of Collection used in TanStack DB (to Electric or something)

### PocketBase

> Open Source realtime backend in 1 file
>
> <https://github.com/pocketbase/pocketbase>

Similar to Trailbase

Golang

Js/Ts and Dart SDKs only

SDK supports React Native deployment

Can be extended with JS/TS (like Trailbase), or embedded into a Go app

# 