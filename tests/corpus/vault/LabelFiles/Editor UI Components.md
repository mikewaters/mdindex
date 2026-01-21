---
tags:
  - active
  - landscape
---
# Editor UI Components

Is a:: Technical Landscape [Document.md](./Document.md)

`wysiwyg`: Editors that display rendered rich text inline as it’s typed.

`wysiwym`: Separates content from layout, where the content is structured to convey layout semantics (like Markdown)

## Parsing

### Python

<https://github.com/T9Air/MarkEd>

<https://github.com/njvack/markdown-to-json>

#### MyST ecosystem

<https://github.com/executablebooks/markdown-it-py>

<https://github.com/executablebooks/MyST-Parser>

## Inline Preview

### Octo

<https://github.com/davidmyersdev/octo>

Has the inline preview that I like. Vuejs

<https://octo.app>

### Alexandrie

> A website for taking beautiful notes in extended Markdown format.
>
> <https://github.com/Smaug6739/Alexandrie>

> 16 stars, vuejs, French

### Malarkdowny

> A next-generation What You See Is What You Wrote markdown editor
>
> [andrewbaxter.github.io/malarkdowney/](andrewbaxter.github.io/malarkdowney/)

> <https://github.com/andrewbaxter/malarkdowney?tab=readme-ov-file>

## Codemirror-based

### chun-mde

> Markdown editor based on codemirror 6
>
> [madeyoga.github.io/chun-mde/](madeyoga.github.io/chun-mde/)

> <https://github.com/madeyoga/chun-mde>

a good code sample probably

## Monaco-based

### yn

<https://github.com/purocean/yn>

> A highly extensible Markdown editor. Version control, AI Copilot, mind map, documents encryption, code snippet running, integrated terminal, chart embedding, HTML applets, Reveal.js, plug-in, and macro replacement.

> [yank-note.com](yank-note.com)



## Libraries and Components

+ ### Unopinionated base

   #### Prosemirror

   > The ProseMirror WYSIWYM editor
   >
   > [prosemirror.net](prosemirror.net)

   #### Slate

   > A completely customizable framework for building rich text editors. (Currently in beta.)
   >
   > [slatejs.org](slatejs.org)

   [docs](http://docs.slatejs.org/concepts), [code](https://github.com/ianstormtaylor/slate)

   Supports a lot of rendering modalities, and not all are wysiwyg

   Slate demo [examples](https://www.slatejs.org/examples/check-lists)

   #### Monaco

   The VSCode editor component

   > <https://github.com/microsoft/monaco-editor>

   #### Codemirror

   Another browser code editor

   > <https://github.com/codemirror/dev/>

   Very interesting, as obviously one can build a markdown code editor. Works primarily on a “line” concept though, rather than a block one.

   #### Ace editor

   Older thing

   <https://ace.c9.io>

   Django md editor is based on Ace: <https://github.com/agusmakmun/django-markdown-editor?tab=readme-ov-file>

   

+ ### Editors with an inline-preview modality

   [GravityUI](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/beb69366-b105-4388-a80b-f0d7054d910e#1ea5b19b-5e89-46bc-87eb-5d7fcdab178a) editor has both markeup and wysiwyg views, and in the markup view you can bind a toolbar element to a plugin, which is **kinda** what I want (I want to bind a character). This project also does “automatic conversion” of `- ` and `> ` (char+space) to markdown elements in both markup and wysiwyg modes, and so the ability to do thiese two things together is promising.

   [Remirror](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/beb69366-b105-4388-a80b-f0d7054d910e#75db3419-5f55-4451-9d9d-dd5ddbb3d16e) (prosemirror wrapper for React) supports a dual editor which has a decent bidirectional input (type md or wysiwg and the other updates). 

   Slate supports what they call “[an editor with built in markdown preview”](https://www.slatejs.org/examples/markdown-preview), modality which is pretty cool (it keeps markup). 

   #### Hedgedoc

   <https://github.com/hedgedoc/hedgedoc>

   Codemirror-based markdown edit and preview. [Nest.js.md](./Nest.js.md) backend, object store and Postgres, JavaScript frontend.

   #### Gravity (React)

   <https://github.com/gravity-ui/markdown-editor>

   > Markdown wysiwyg and markup editor\
   > Extensibility through the use of ProseMirror and CodeMirror engines.
   >
   > [preview.gravity-ui.com/md-editor/](preview.gravity-ui.com/md-editor/)

   Supports commands

   #### Plate

   <https://github.com/udecode/plate>

   Built on top of Slate, layering in abstractions to componentize 

   Has a very cool Roam-like preview though

   #### SimpleMDE

   <https://github.com/sparksuite/simplemde-markdown-editor>

   This library is tired of being dead, but there are some forks.

+ ### Editors with a dual-pane preview

   #### Remirror

   > A React *toolkit* for building *cross-platform* text editors

   Prosemirror framework for React

   <https://github.com/remirror/remirror>

   [docs](https://www.remirror.io/docs)

   Has a dual editor, but its based only on prosemirror and so is kinda rough.

+ ### Editors that are WYSIWG-only

   #### Quill 

   > Quill is a modern WYSIWYG editor built for compatibility and extensibility
   >
   > [quilljs.com](quilljs.com)

   > Granular access to the editor's content, changes and events through a simple API. Works consistently and deterministically with JSON as both input and output.

   Supports embeds, 43k stars

   #### Lexical

   > Lexical is an extensible text editor framework that provides excellent reliability, accessibility and performance.
   >
   > [lexical.dev](lexical.dev)
   >
   > By facebook

   #### BlockSuite

   The stuff behind AFFine

   <https://block-suite.com>

   #### BlockNote

   Supports nested blocks, but how far can it go? Can it do what Blocksuite does, with arbitrary nesting as well as breaking blocks into fragments?

   > Note: Interestingly, one can [easily host a BlockNote server using a Cloudflare Durable Object via PartyKit](https://github.com/partykit/partykit/tree/main/examples/blocknote)

   Markdown i/o sucks. Its [markdown—>html—>blocknote](https://github.com/TypeCellOS/BlockNote/blob/main/packages/core/src/api/parsers/markdown/parseMarkdown.ts#L51), and [obtuse af](https://github.com/TypeCellOS/BlockNote/blob/main/packages/core/src/api/parsers/markdown/parseMarkdown.ts#L63). Also, markdown does not support some BN features (BN allows more nesting - which I like), and so i/o is “[lossy](https://www.blocknotejs.org/docs/editor-api/converting-blocks)” and they suggest export using JSON. But import doesnt have a workaround like that. The code sucks, bc they are trying to use the same modules for browser- and server-side, and need to do import magics because javascript is maintained by squirrels on adderall.

   I do not like that it doesnt conform to markdown wrt newlines; each newline is a new block, but in markdown it would require an empty line (and [MichaelDown.md](./MichaelDown.md) depends on this). IMO they are not interested in markdown compliance, and their editor is trying to be Notion instead of Obsidian.

   ##### Analysis

   I do not want WYSIWYG. I want to write markdown. That markdown being “decorated” in the editor would be OK, as long as markup is preserved.

   #### Editorjs

   > Free block-style editor with a universal JSON output

   <https://editorjs.io/>

   [code](https://github.com/codex-team/editor.js), [docs](https://editorjs.io/base-concepts/)

   does not support markdown lmao

   each block in the editor maps cleanly to some JSON

   each block is an independent contenteditable

   ui-focused, some plugins need to declare ui helpers like toolbar icons

   shit-ton of plugins

   #### Milkdown

   <https://milkdown.dev/playground>

   <https://github.com/Milkdown/milkdown> 

   Built on prosemirror and remark

   #### Tiptap

   Based on prosemirror, firmly wysiwyg

   + Tiptap based

      + ### [Langchain-AI/OpenCanvas](https://github.com/langchain-ai/open-canvas?tab=readme-ov-file)

         > 📃 A better UX for chat, writing content, and coding with LLMs.
         >
         > [opencanvas.langchain.com](opencanvas.langchain.com/)

         > Open Canvas is an open source web application for collaborating with agents to better write documents.

         > ![image 10.png](./Editor%20UI%20Components-assets/image%2010.png)
         >
         > - Memory: Open Canvas has a built in memory system which will automatically generate reflections and memories on you, and your chat history. These are then included in subsequent chat interactions to give a more personalized experience.
         >
         > - Custom quick actions: Custom quick actions allow you to define your own prompts which are tied to your user, and persist across sessions. These then can be easily invoked through a single click, and apply to the artifact you're currently viewing.
         >
         > - Pre-built quick actions: There are also a series of pre-built quick actions for common writing and coding tasks that are always available.
         >
         > - Artifact versioning: All artifacts have a "version" tied to them, allowing you to travel back in time and see previous versions of your artifact.
         >
         > - Code, Markdown, or both: The artifact view allows for viewing and editing both code, and markdown. You can even have chats which generate code, and markdown artifacts, and switch between them.
         >
         > - Live markdown rendering & editing: Open Canvas's markdown editor allows you to view the rendered markdown while you're editing, without having to toggle back and fourth.

      ### [TypeCellOS/BlockNote](https://github.com/TypeCellOS/BlockNote)

      > A React Rich Text Editor that's block-based (Notion style) and extensible. Built on top of Prosemirror and Tiptap.
      >
      > [www.blocknotejs.org/](www.blocknotejs.org/)

      ### [Prototypr/typr](https://github.com/Prototypr/typr/tree/main)

      > Typr Editor is a writing tool made with Tiptap / Prosemirror with ready-made user state management and publishing workflows.
      >
      > [prototypr.io/typr](prototypr.io/typr)

      > Typr is a Medium-like editor for React that integrates with your user system and CMS. It handles content loading, creation, and auto-saving out of the box. Add your user data and database props to get a fully functional editor with draft and publishing workflows.

      wysiwyg, but mainly meant for a publishing workflow.

      ### [Doist/typist](https://github.com/Doist/typist)

      > The mighty Tiptap-based rich-text editor that powers Doist products.
      >
      > [typist.doist.dev](typist.doist.dev)

      > Typist is the mighty ++[Tiptap](https://tiptap.dev/)++\-based rich-text editor React component that powers Doist products, which can also be used for displaying content in a read-only fashion. Typist also supports a plain-text mode, and comes with HTML/Markdown serializers.

      Supports [single-line mode](https://typist.doist.dev/?path=/docs/documentation-usage-singleline--docs)

      WYSIWYG

   #### Vrite

   > <https://docs.vrite.io/getting-started/introduction/>

   > Has extensions, kinda cutesy WYSIWYG 

   #### MDX Editor (React)

   > A rich text editor React component for markdown

   [mdxeditor.dev](mdxeditor.dev)

   <https://github.com/mdx-editor/editor>

   2k stars

   Built using Codemirror and Lexical, bundles markdown components as JSX to be shat into a React app.