---
tags:
  - active
---
# MichaelDown

What do I want to change in Markdown?

- Block hierarchies

- Better quoting structures

   - Personal notes/asides (note to self) should be distinct

   - Highlights should also be distinguishable

   - quote colors and headers

- Lists need headers

- Blocks need to be toggle-able

- Timestamps

   - Useful for append-log notes

- Better tag semantics

   - Nested tags

- Better todo lists

   - Ability to slice and dice

- Highlight support

   - Ability to have a block type that contains quotes and references

      - “I found these $\[highlights\] on $webpage, which is related to $\[efforts\] and is import because …”

### Use Detail el?

```python
::::details{open="true"}
:::detailsSummary
details summary 

**bold**
:::

:::detailsContent
details content

**bold**
:::
::::
```

### Relate $x to $y and $z

Research transparency, centralization, organization, mining

```
/thread [search term]
/raindrop
raindrop://[search term]
creates a block that references raindrop,
embeds a link to the snapshot as well as 


```

[MultiMarkdown](https://github.com/fletcher/MultiMarkdown/blob/master/Documentation/MultiMarkdown%20User's%20Guide.md)

Conceptually, there is a:

- Format or grammar, a markdown extension 

- Reader components

- Driven of the idea that one can use text files and text file formats expressively, creating richness beyond semantics, by using technologies that can understand language and ontology

[Markdown AST.md](./Markdown%20AST.md)

The vision looks like this:

```markdown
## current #problems
These are the worst:
- [[I cant sit]] ::priority1 etc
- This important note ::is related to:
```

A good attempt would be:

```markdown
## current problems
- i am never comfortable
- this ^^ text should turn into a problem object

```

### Code samples

#### Obsidian “markdown-attributes” plugin

This [module](https://github.com/javalent/markdown-attributes/blob/main/src/main.ts) apples html is and class attributes to markdown elements, just like I want. It receives some HTML element from Obsidian (an `HTMLElement` and `MarkdownPostProcessorContext`) and descends into it to see if it can be decorated. Its mostly a bunch of regexes and `if` statements.

### Goals

A Markdown variant that:

- supports rich types in metadata

- applies metadata at the block level, not just at the document level

- defines a document block hierarchy supporting arbitrary levels of nesting for more fine grained metadata application (like an outline editor)

An editor component that:

- facilitates metadata association (“labeling”) via interactivity, using character indicators like colon, hash, at.

- supports label hinting via typeahead

- supports hinting in the background in a content viewer pane

- can outline a block, including single and multiple line blocks, including paragraph blocks terminated with an empty line

- can select multiple lines and or blocks and make them into a single block potentially making sub-blocks into children



### Roadmap

1. Ingest life data easily from “GTD list dumps” (whaddyacallit) or from journalling

2. View and edit the life data in a whiteboard format, where related things are linked

3. View and edit the life data in a map of content format; a single document that shows an overview of related stuff - blocks - that are only embeds, they live elsewhere.

4. Support for Obsidian

### Conceptual

Doubles down on blocks as first class citizens.

- Block hierarchies are respected: 

   - An `h{n}` is a child of the nearest preceding `h{n-1}`, if it exists

      - Each inline and block element of the `h{n}` is then a grandchild of `h{n-1}`.

   - A nested list item is a child of the higher level list item

- Block metadata applies at the block level (not the document level)

   - Tags (etc) are block-scoped; a tag anywhere inside a block - as long as its not included in a child block - is scoped to that block

- Block metadata is inherited by children of the block:

   - All descendants of a block inherit the parent block element’s tags and other metadata.

### Multi-line block elements

- heading 1..6

- quote

   > obsidian supports ‘callouts’, which override quotes; could this be extended?

- fences

- paragraph (block of text separated by a blank line)

### Single-line block elements

- list elements

- task elements

   > IDEA: override the task status indicator (`[x]`) behavior to do things?

#### Block metadata

`#` is-a, typing. once there’s a type, can define attributes. sub-tags (`/` sep) denote namespaces

`@` has-a, relatedness, attributes

#### Block separation

Typing return twice creates a new block

### Inline elements

> could bold, italic etc be used in some way?

### Reference elements

> Need to distinguish between entity references and plain old URLs. Example: related to topic; how do we specify the related-to “edge”?

- Wiki link: `[[Three laws of motion]]`

- Markdown: `[Three laws of motion](Three%20laws%20of%``[20motion.md](20motion.md)``)`

- Block reference

> Note protocol handlers might be useful; example `[Note](``<obsidian://open?vault=MainVault&file=Note.md>``)`

### Front matter

Because multi-line blocks - which have their own scope - can have their own frontmatter and thus metadata attributes. Realistically, I only care about doing this for headings.

### Implementation

Given that each block has its own scope, it needs its own database entry.

<https://github.com/Feel-ix-343/markdown-oxide>

INBOX

<https://casual-effects.com/markdeep/>

Markdeep can embed other markdeep documents

<https://casual-effects.com/markdeep/features.md.html#embeddingdocuments/markdeepinmarkdeep>

Markdeep comparative feature set

<https://casual-effects.com/markdeep/features.md.html#differencesfromothermarkdown>