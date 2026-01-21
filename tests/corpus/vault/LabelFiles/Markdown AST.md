# Markdown AST

## Hierarchy in Markdown

> A Markdown AST is flat, with the exception of some node types that can have a hierarchy.

> Header elements are always flat, even though they appear like children (h1 —> h2 etc). This is due to their HTML representation being flat as well.

In Markdown's AST structure, here are the elements that can have children:

1. `root` - The root node that contains all other elements

2. Block-level containers:

- `blockquote` - Can contain paragraphs, lists, and other block elements

- `list` - Contains listItem elements

- `listItem` - Can contain paragraphs and other block elements

- `table` - Contains tableRow elements

- `tableRow` - Contains tableCell elements

1. Inline containers:

- `paragraph` - Can contain inline elements like text, links, emphasis

- `heading` - Can contain inline elements like text, links, emphasis

- `link` - Can contain text and other inline elements

- `emphasis` - Can contain text and other inline elements

- `strong` - Can contain text and other inline elements

+ ## Markdown variants (flavors)

   GitHub Flavored Markdown (GFM)

   - standard for software documentation

   - Adds tables, task lists, auto-linking, code fence syntax highlighting, Emoji shortcodes, username mentions, issue references, Strikethrough and fenced code blocks

   CommonMark

   - Standardized specification of core Markdown

   - Aims to be unambiguous reference implementation

   - Strict parsing rules for consistent rendering

   - Foundation for many other flavors

   Pandoc Markdown

   - Academic and publishing focus

   - Extensive document conversion capabilities

   - Citations, footnotes, metadata, cross-references

   - Math equations, definition lists, attributes

   R Markdown

   - Data science and statistical analysis

   - Embedded R code execution

   - Interactive notebooks

   - Publication-quality documents and presentations

   MDX

   - React components in Markdown

   - JSX integration

   - Dynamic content

   - Used heavily in modern documentation sites

   MultiMarkdown

   - Academic writing extensions

   - Tables, footnotes, citations

   - Math support via LaTeX

   - Metadata and cross-references

   Markdown Extra

   - PHP Markdown extension

   - Tables, footnotes, definition lists

   - Fenced code blocks

   - Special attributes

   GitLab Flavored Markdown

   - Based on GFM with additional features

   - Math equations via KaTeX

   - Mermaid diagrams

   - Advanced task lists

   Discord Markdown

   - Chat-optimized variant

   - Spoiler tags

   - Custom emoji

   - Message formatting

   Stack Overflow Markdown

   - Q&A-focused variant

   - Code highlighting

   - Spoilers

   - HTML sanitization rules

## Obsidian Markdown 

[Obsidian Flavored Markdown - Obsidian Help](https://help.obsidian.md/Editing+and+formatting/Obsidian+Flavored+Markdown)

> Obsidian strives for maximum capability without breaking any existing formats. As a result, we use a combination of flavors of [Markdown](https://help.obsidian.md/Editing+and+formatting/Basic+formatting+syntax).
>
> Obsidian supports [CommonMark](https://commonmark.org/), [GitHub Flavored Markdown](https://github.github.com/gfm/), and [LaTeX](https://www.latex-project.org/). Obsidian does not support using Markdown formatting or blank lines inside of HTML tags.

### Supported Markdown extensions 

| Syntax | Description | 
|---|---|
| `[[Link]]` | [Internal links](https://help.obsidian.md/Linking+notes+and+files/Internal+links) | 
| `![[Link]]` | [Embed files](https://help.obsidian.md/Linking+notes+and+files/Embed+files) | 
| `![[Link#^id]]` | [Block references](https://help.obsidian.md/Linking+notes+and+files/Internal+links#Link%20to%20a%20block%20in%20a%20note) | 
| `^id` | [Defining a block](https://help.obsidian.md/Linking+notes+and+files/Internal+links#Link%20to%20a%20block%20in%20a%20note) | 
| `%%Text%%` | [Comments](https://help.obsidian.md/Editing+and+formatting/Basic+formatting+syntax#Comments) | 
| `~~Text~~` | [Strikethroughs](https://help.obsidian.md/Editing+and+formatting/Basic+formatting+syntax#Bold,%20italics,%20highlights) | 
| `==Text==` | [Highlights](https://help.obsidian.md/Editing+and+formatting/Basic+formatting+syntax#Bold,%20italics,%20highlights) | 
| ```` ``` ```` | [Code blocks](https://help.obsidian.md/Editing+and+formatting/Basic+formatting+syntax#Code%20blocks) | 
| `- [ ]` | [Incomplete task](https://help.obsidian.md/Editing+and+formatting/Basic+formatting+syntax#Task%20lists) | 
| `- [x]` | [Completed task](https://help.obsidian.md/Editing+and+formatting/Basic+formatting+syntax#Task%20lists) | 
| `> [!note]` | [Callouts](https://help.obsidian.md/Editing+and+formatting/Callouts) | 
| (see link) | [Tables](https://help.obsidian.md/Editing+and+formatting/Advanced+formatting+syntax#Tables) | 

### Nested tags 

Nested tags define tag hierarchies that make it easier to find and filter related tags.

Create nested tags by using forward slashes (`/`) in the tag name, for example  `#inbox/to-read` and `#inbox/processing`.