# A SQLite client lib that understands Heptabase

Create a library that models the Heptabase data model. Use a copilot, provide it the DDL from sqlite. 

How can I create a cross-platform library, for both Python and Typescript? Or do I create an API that can be run in wasm in the browser and as a server? (pyiodide?)

### Requirements

- python library

- typescript library

## Reads

1. Directly read the sqlite database; this should be fine, though I am not 100% sure the sqlite concurrency model will support safe reads if Heptabase uses transactions

2. Indirectly read the database, by enabling the sqlite db file write-ahead log, and piping that to a replicated solution (rl-sqlite etc). This will require testing.

## Writes