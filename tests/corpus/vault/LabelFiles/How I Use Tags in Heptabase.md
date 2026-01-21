# How I Use Tags in Heptabase

Heptabase is integrated with my ontology via its tag databases. 

> I need this document, because tag databases don’t support notes of their own.

**Tag Databases are namespaced using Sys3 domains where needed:**

- `LifeOS::Inventory`

- etc

### `Heptabase PKM` - Knowledge domain constructs

Certain cards represent - and are titled based on - a `Topic`.  These cards will carry a `#topic` tag, representing the topic itself in some knowledge area taxonomy. Other cards can then be related to it.

### `Sys³ Entities` - Core Ontology

These should be migrated to Obsidian

These tags should be used for classifying Cards which themselves represent some core concept in the LifeOS.

These Cards can then be referenced, either using a backlink, or a defined property on some LifeOS or Knowledge Management tag.

Example: a “Daily Challenges” log:

> - Create a card for a common problem, in the `#thing` tag database with a klass referring to the `#ontology` database entry for `Problem`
>
> - Apply the `#journal` tag to the Daily Challnges card



- `#ontology`

   - This table contains cards that represent the sys3 core definitions and the domains they are a part of. ex: `Vision`, `Behavior`

- `#instance`

   - This contains all other cards that need to reference or derive any of the types.

   - Has property `klass` to define this relationship

- `#type`

   - This contains cards that represent elements of Sys³ that are derived types of the core elements in the ontology. 

   - These elements are part of running a life business

   - Has property `typ` to define this relationship

- `#lifeos element`

#### Professional life (`LifeOS/Professional`)

I would like to move these out, but cannot right now.

- `#dealertrack`

- `#epic`

#### Life Inventory tracking (`LifeOS::Inventory`)

- `#app/service`

   - Cards having this tag represent tools that I use or subscribe to. Tracks whether they are installed, whether I paid for them etc.

### Knowledge Management domain

#### Knowledge metadata (`Contexts`)

- `#inbox/fileme` - catch-all tag for cards I’ve reviewed and decided they need to be moved to the correct location

- `#inbox/ingest` - catch-all tag for cards I’ve reviewed and decided they need to be incorporated into some document

- `#active`

- `#mw` *private*

- `#📌 readme` *need to ingest*

#### Documents (`KM::Document Types`)

What does the entirety of a card represent, within my ontology, where (almost) everything is a document? 

- `# 📑 document` - defined elsewhere

- `# 📓 journal` - timestamped logbook

- `# 📒 notebook` - collection of arbitrary notes about some subject

- `#note` - a single arbitrary note, I probably dont need this in Heptabase

- `#reference`

- `#landscape` - a specific type of document, where I collect information about some subject in order to make inferences or choices.

- `#metadata` - in Heptabase, I can’t attach enough metadata in enough places; in whiteboards, for example, I place a `this.md` file (like I might in a filesystem directory) to describe the purpose of the board.

#### Card Content

What does the content of a card represent, within the KM domain? In my desired world, this would be at a **block** level, not just at a card level.

- `#software`

- `#inspo` *inspirational*

- `# 💡 idea`

- `#concept`

- `#recipe`

- `#boook`

### Other domains

#### Information Domain (`Is a`)

#### Work Domain (`Use/Purpose`)

> *When a card has one of these tags, the actual entity it references can appear in a tag property.*

- `#experiment`

- `#research-thread`

- `#project`

- `#to-learn`

#### ??? (`Areas of Interest|Responsibility`)

TBD

- `#genai`