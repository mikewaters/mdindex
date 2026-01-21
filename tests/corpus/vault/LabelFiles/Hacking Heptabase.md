---
tags:
  - experiment
---
# Hacking Heptabase

I really like this tool, its sits at the right level of abstraction for my broken brain. However, I use other knowledge management tools to fill Heptabase’s functional gaps, but it does not support any integrations with other tools and so my data remains silo’d.

For this reason, I would like to investigate import/export hacks that utilize the app’s internal architecture/data stores.

### Replicating Heptabase sqlite to Cloudflare R2

As of Jan 24, 2025, WAL mode is *not* enabled for the Heptabase sqlite database:

```python
sqlite3 "/Users/mike/Library/Application Support/project-meta/hepta.db" "PRAGMA journal_mode;"
delete
```

Use Litestream to replicate data via WAL.

`/Users/mike/Library/Application Support/project-meta/litestream.yml`:

```python
dbs:
  - path: hepta.db
    replicas:
      - type: s3
        bucket: heptabak
        endpoint: https://10834ace00929bfbe844d77c8ed04a1c.r2.cloudflarestorage.com
        region: us-east-1
```

Requires standard AWS S3 secrets in env (or in config):

```python
export LITESTREAM_ACCESS_KEY_ID=
export LITESTREAM_SECRET_ACCESS_KEY=
```

Running:

```python
$ litestream replicate --config litestream.yml
```

Note this will create internal `_litestream` tables, hasn’t broken anything yet.

### Their DB schema

```python
sqlite3 hepta.db '.fullschema'
```

[heptabase-fullschema-2025-01-24.sql](./Hacking%20Heptabase-assets/heptabase-fullschema-2025-01-24.sql)

### Intercepting Heptabase Traffic

[Reverse Engineering on MacOS.md](./Reverse%20Engineering%20on%20MacOS.md)

+ ## Issues with Heptabase 

   - Cards cannot contain non-static external data (i.e. not an image, file, video etc); an example is an external TODO list

   - Difficult to navigate the iPhone app (but this is improving at a good clip)

   - I cannot import data automatically (only manual markdown); example is Tana PKM data

      - I cannot even manually import most metadata (it gets ignored in Markdown, though that’s mostly Markdown’s fault)

   - I cannot deep link into the Heptabase app, even though it has a URL Handler registered with MacOS. Hence I cannot put hyperlinks to Heptabase resources into other systems unless I use the `http` protocol (which opens Safari instead of the app, and Heptabase has one of the most annoying login systems I’ve ever seen, worse than Notion)

      - I have attempted this in the past: [MacOS URL Handlers.md](./MacOS%20URL%20Handlers.md)

   - Heptabase web login is highly annoying and cannot easily be automated

   - I cannot use deep links to other apps in Heptabase cards (like `drafts://..`)

      - This can be mitigated with a local forwarding proxy

   - The UI is information-dense but meaning-sparse; there are a lot of improvements that could be made to the card UI

+ ## Goals

   1. [Be able to extract data from Heptabase programmatically](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/7cd26903-746d-4ff0-a7cf-2d222a4c64fa#053adfed-78f0-4d14-a154-ed99dd5f373b)

   2. [Be able to manually input data into Heptabase programmatically](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/7cd26903-746d-4ff0-a7cf-2d222a4c64fa#d041fd84-0b10-471e-b49d-d354c8d9c90b)

   3. [Be able to automatically input data into Heptabase programmatically](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/7cd26903-746d-4ff0-a7cf-2d222a4c64fa#4bdbc442-b09b-4eac-8072-66e44f00f20d)

   4. [Be able to create Live Cards in Heptabase, possibly by inputting data directly into the app](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/7cd26903-746d-4ff0-a7cf-2d222a4c64fa#0f52b6ee-90e7-4847-beb9-5556bb184e70)

   5. [Allow Heptabase to open non-http{s} URLs](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/7cd26903-746d-4ff0-a7cf-2d222a4c64fa#9c763cec-3417-43a1-938b-d44ff69660e9)

   6. [Allow other apps to open Heptabase in-app links](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/7cd26903-746d-4ff0-a7cf-2d222a4c64fa#dd9162f7-c75e-4424-b0f5-881e326b4186)

   7. Automatically “decorate” cards based on their metadata or content

+ ## Solutions

   ### Data access

   Extract or update data using [A SQLite client lib that understands Heptabase.md](./A%20SQLite%20client%20lib%20that%20understands%20Heptabase.md)

   Experiment with [Heptabase Markdown Import.md](./Heptabase%20Markdown%20Import.md) from various systems

   ### Integration with other tools

   Implement a [Local MacOS Forwarding HTTP Proxy.md](./Local%20MacOS%20Forwarding%20HTTP%20Proxy.md)

   Waiting for Deep links; still not working as of Aug 4, 2024

   
