---
tags:
  - lifeos element
tags:
  - use-case
---
# 📒 LifeOS Use Case Log

| **Theme** |  | 
|---|---|
| [Ontology Service.md](./Ontology%20Service.md) (Substrate) |  | 
| [Deep Research Tool.md](./Deep%20Research%20Tool.md) |  | 
| [Lifedata Dashboard.md](./Lifedata%20Dashboard.md) |  | 

### Notes

Nov 9, 2025

Highlights are more important than I’d considered; basically any note that I didn’t write is a highlight! A highlight can be applied to a bookmark, but doesn’t need one; both of them have an URL, a @bookmark isn’t so much what you have in your browser, it’s a richer structure that’s much more connected to the wider context.

Oct 28, 2025

### Landscape Deep Research

- keep a running list of all the items in the landscape; these will become search/monitoring terms. 

- ex: “Rewind.ai Replacements” starts out with one term to search HN discussions; each time I find a new alternative, that adds a search term.

Oct 13, 2025

Deep Research Behavior:

- every time a command line tool is in the context during a research task, retrieve the instruction man page for the command. Maybe this is a Proper Noun handler? That calls a tool. This is just RAG for docs, and I guess could just be an in-context instruction.

Oct 10, 2025

### A lifedata dashboard activity feed

A dashboard widget that shows the bookmarks that I've made today, in the past couple of days, or even in the last week, with stars or upvotes next to them. Each time I see the widget in the dashboard, I can upvote or star items that I bookmarked and wanted to save for later but didn't do anything about. So then day over day, I can constantly give a signal about things that I saw as well as be reminded of them. And eventually things will float to the top and maybe even be added to my short-term permanent memory so that I'll remember to do them. If I do this consistently, things that sink to the bottom at some point could just drop right off that list because I don't care anymore. And maybe there's an option in that widget on the dashboard to remove them in that case. 

[Lifedata Dashboard.md](./Lifedata%20Dashboard.md)

Dashboard items:

1. Bookmarks saved recently

   1. Raindrop and DoMarks are treated differently 

2. Apps installed on a device

3. Tasks in various apps

The point here is not to see “a list of things”, more so to have a self-reinforcing reminder where a small effort with low anxiety can slowly prioritize and remind over time.

Oct 10, 2025

### Bookmark -> Landscape document (with [Raindrop.io.md](./Raindrop.io.md) automation)

Listen for new bookmarks in given collections and assign them to a document t collection automatically - like a landscape document. This may be a hack shortcut to populating a landscape document.

Oct 10, 2025

### Bookmark or Highlight share → Classification

Upvoting an HN comment - or similarly passively indicating interest in a thing - triggering an agent to try and figure out why I did that and then take action. 

This matches the “[Magnetic Capture.md](./Magnetic%20Capture.md)” paradigm

Example:

- do NER on the comment, and see that it mentions something in my interests like electronics or raspberry pi automation with home assistant 

- use NER to find entity classes that interest me, like apps, github repos, project management tools etc

- find related activities and efforts

Sep 28, 2025

### Automatically researching a Thing, using predefined dimensions

Once the type of thing is identified, an agent can use the [Defining the LifeOS Ontology - Notes.md](./Defining%20the%20LifeOS%20Ontology%20-%20Notes.md) to figure out which dimensions are important to capture.

### Structuring Research with dimensions (to support agents)

[Deep Research Dimensions.md](./Deep%20Research%20Dimensions.md)

When I create a landscape document, I implicitly capture some number of dimensions about the topic. For example what language it’s written in, what the license is, is it hosted etc. How can I use an ontology to structure this? 

- For existing documents: 

   - each document could have a list of the dimensions; when a new item is added to the research, an agent could capture those dimensionns

- For new documents: 

   - An agent could ask me for a list of the dimensions

   - An agent could suggest dimensions based on the topic, and allow me to choose

   - An agent could use an external o to lift to classify the topic, and then walk up the chain of topics looking for those that have a designated structure, and then merge those results.

      - example: Home Assistant will have both Software and Home Automation classification, and those could each have a set of important dimensions (like HomeKit support, zwave support, open source license etc)

- Given the list of dimensions I currently have, I could ask an AI to extract everything into a list, and then I add those to the ontology officially - manually - via a migration

### Adding an item to a landscape document

I have found a GitHub repository that should be added to some research that I am doing. This may be in a Heptabase landscape doc or somewhere else.

I should be able to:

- share the url to the research

- ask an agent to “add this to my research on Something”, and it it figures out what I mean and then adds it

- do the above and have an [LifeOS Agents.md](./LifeOS%20Agents.md) do the actual analysis of the new thing, and update my research by itself

Sep 10, 2025

### Listing things I’ve recently worked on

I documented my new Mac Mini setup **somewhere**, it could be in Heptabase, or Obsidian, or a git repo on one of my computers. I need a Changelog. 

This can be implemented in [LifeOS Agents.md](./LifeOS%20Agents.md)

Sep 8, 2025

### Classifying and linking research topics

I am saving a shit ton of links from HN and elsewhere, either to [Raindrop.io.md](./Raindrop.io.md) , Heptabase, or Obsidian. I don’t know where any of it is. I am not even sure where to put it. A [Capture Tool (Thing Catcher).md](./Capture%20Tool%20\(Thing%20Catcher\).md) would help proactively, but I need [Bookmarks Bot.md](./Bookmarks%20Bot.md) as well.

I need some rigor around this, so I can spend less time on capture, be confident that things are findable when I need them, and to delegate rote tasks to [LifeOS Agents.md](./LifeOS%20Agents.md).

### Recalling command line-fu

Example is `netstat -tunape` and the fact that I cannot for the life of me remember how to do this in a Mac. There are snippet tools.

This is IMO related to keyboard shortcut memory.

Aug 31, 2025

### Organizing research in the browser

When I have a million tabs open, it would be really helpful to see all the places I’ve saved a link, and/or the Life-Items that is associated with. Imagine a github project, when I open the readme it tells me that I saved it to Raindrop, added it to a Runbook, filed it in some research etc.

Aug 27, 2025

### Finding a Life-Item and its related Resources from my iPhone

Writing up my Obsidian migration plan, and want to align it with my work document structure. This requires I find or create a goal, or objective, or solution, tied to that effort. Which is fucking impossible.

Idea: iPhone app that allows me to free form type, and will find anything that matches, and tell me what it is and where to find its artifacts.

Feb 2, 2025

### Hackernews Thread Auto-Researcher

Research bot: grab HN discussions from the past: look for links etc

### Hackernews Thread Saver-Helper

Research bot: reduce the friction I have in creating and adding to Documents in my HB

Jan 21, 2025

### Search the contents of a Known List

“What’s that term I can’t recall” … cognitive bias

“What’s that cognitive bias where X”

“What’s that thing where you think you know lore than you do” … illusion of explanatory depth bias

[Goal! Eliminating Friction.md](./Goal!%20Eliminating%20Friction.md)

[Goal! Augmenting Memory.md](./Goal!%20Augmenting%20Memory.md)

[Goal! Enhancing Focus.md](./Goal!%20Enhancing%20Focus.md)

[Goal! Leverage Visual Learning Abilities.md](./Goal!%20Leverage%20Visual%20Learning%20Abilities.md)