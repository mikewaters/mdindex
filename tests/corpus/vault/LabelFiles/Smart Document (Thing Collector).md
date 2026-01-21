---
tags:
  - workload
---
# Smart Document (Thing Collector)

Notebook that aggregates things and shows information

[[./Magnetic Capture.md]]

[[./Manual data entry tool.md]]

Serves as the basis for:

- Cross-system data related to some entity

- Status

- Dashboards

Composed of blocks that each carry metadata, which are composed of other blocks as well as inline elements/text strings, chunks of which can also carry metadata.

Block (container) metadata:

- what type of entity 

Inline metadata:

- URI (type, identifier)

- URL (source, protocol)

- description

## Technology Choices

### Option: Embedded Web Component

[Editor UI Components.md](./Editor%20UI%20Components.md)

Use a component with a “raw” markdown editor and an inline-preview modality that decorates text without transforming them completely.

- Codemirror

- Plate (Slate)

- Hedgedoc

- Gravity

### Option: Reuse an editor applicaiton

A real editor could be used if it supports an LSP plugin, copilot integration, or similar autocomplete helper. 

- [markdown-oxide ](https://github.com/Feel-ix-343/markdown-oxide)“LSP for PKMs”

### Option: Livecoding tool

- <https://github.com/toplap/awesome-livecoding>

- 

### Also Interesting

#### CopilotKit: <https://github.com/CopilotKit/CopilotKit>

## Technical Requirements

- Supports Markdown and the AST can be customized

- Supports databases, indirectly

- YJS or CRDT support

- Not an WYSIWYG-only

## Ways to view a smart document

- Embedded in a 3rd party website

   - As a streaming video, gif

- As a web component embedded in some UI block/brick

- As a copy/paste-able chunk of markdown links having icons and emojis and gifs

- 

## User Experience

- bring in different types of data at the same time

- easily relate incoming data to another entity if i know it