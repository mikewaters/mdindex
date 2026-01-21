# Markdown Language Servers

For [MichaelDown.md](./MichaelDown.md)

1. IWE

2. Markdown Oxide

3. Vale

4. Marksman 

## Analysis

- Obsidian doesn’t support any of these; there was a plugin for Vale, but nobody used it and the author decommed it. No results found in Obsidian Community Plugins for “language server” or “LSP”

- Editors supported are usually Zed, Sublime, VSCode, Nvim, Helix, and then random others.

- Vale looks really cool, it ships mainly as a CLI for writing prose with an LSP that wraps its capabilities

- IWE and Oxide are both up to date as of Sep 5, 2025, and they are both geared towards supporting any - or many - editors, and having a single component that stores the PKM data and exposes it via LSP to those editors. 

- Marksman is not focused on PKM< and it has not been meaningfully updated in almost a year.

- IWE appears to be more AI-forward, storing everything in a graph representation and providing rich analysis capabilities, whereas Oxide seems to be geared towards supporting editing/authoring itself. Then Vale wants to support “writing” (prose), and wants users to use its CLI for the prose and for doing backend validation. And so it is more coming from a “linting” perspective.

### IWE vs Oxide, from IWE author

> At their core, the two projects differ conceptually; IWE is built around the idea of nested documents and graph transformations.

#### IWE

- It normalizes header levels

- Offers code actions that let you extract/inline sections and restructure text.

- IWE is not just an LSP; it's also a command-line utility that supports batch operations.

- It supports document generation through block-reference transclusion.

#### Oxide

- Footnote completion

- Tag support

- Wiki link support

- Heading references (link to a specific header in another document)

- Daily notes



## Libraries

### Marksman

> Marksman is a program that integrates with your editor to assist you in writing and maintaining your Markdown documents. Using ++[LSP protocol](https://microsoft.github.io/language-server-protocol/)++ it provides **completion**, goto **definition**, find **references**, **rename** refactoring, **diagnostics**, and more

<https://github.com/artempyanykh/marksman>

F-Sharp (wtf?)

### IWE

<https://github.com/iwe-org/iwe>

> LSP for Markdown notes taking

> ++[IWE](https://iwe.md/)++ is an open-source, local-first, markdown-based note-taking tool. It serves as a personal knowledge management (PKM) solution **designed for developers**.
>
> IWE integrates seamlessly with popular developer text editors such as **VSCode**, **Neovim**, **Zed**, **Helix**, and others. It connects with your editor through the Language Server Protocol (LSP) to assist you in writing and maintaining your Markdown documents.
>
> IWE offers powerful features such as **search**, **auto-complete**, **go to definition**, **find references**, **rename refactoring**, and more. In addition to standard Markdown, it also supports wiki-style links, tables, and other Markdown extensions.

Discussions:

- <https://www.reddit.com/r/PKMS/comments/1i54zi7/iwe_new_open_source_bring_your_own_text_editor_pkm/>

- 

### Markdown-Oxide

<https://github.com/Feel-ix-343/markdown-oxide>

> PKM Markdown Language Server

> 

### Vale

[Vale CLI - Introduction](https://vale.sh/docs)

> Vale is a command-line tool that brings code-like linting to prose. Vale is cross-platform (Windows, macOS, and Linux), written in Go, and available on GitHub.

Ships [with an LSP](https://vale.sh/docs/guides/lsp) and integration docs for various editors.

![Pasted 2025-09-05-11-13-43.png](./Markdown%20Language%20Servers-assets/Pasted%202025-09-05-11-13-43.png)


