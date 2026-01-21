# Hierarchical Markdown

Part of Michaeldown

Repr a Topic of relevance to that effort

Block sourced from Perplexity:

To transform a flat Markdown header AST into a nested hierarchical structure where `h1` parents `h2`, `h2` parents `h3`, etc., the most direct solution is **md-hierarchical-parser**, a specialized library that extends MDAST with explicit parent-child relationships between headings\[1\]. Here's a breakdown of approaches:

---

## **Recommended Solution: md-hierarchical-parser**

This library directly addresses the problem of flat heading structures in MDAST:  

```javascript
const mdParser = require('md-hierarchical-parser');
const nestedAST = await mdParser.run("input.md", true);
```

- **Output**: Generates JSON with nested headings (e.g., `h2` becomes a child of `h1`).  

- **Example structure**:  

   ```json
   {
     "type": "root",
     "children": [
       {
         "type": "heading",
         "depth": 1,
         "children": [
           { "type": "paragraph", "content": "..." },
           {
             "type": "heading",
             "depth": 2,
             "children": [ ... ]
           }
         ]
       }
     ]
   }
   ```

- **Advantages**: No manual algorithm implementation required; integrates with existing MDAST tooling\[1\].

---

## **Alternative Approaches**

### **1\. Custom Nesting Algorithm**

For libraries like MDAST that output flat headings, implement logic to group sections:  

```javascript
function nestHeadings(flatNodes) {
  const root = { children: [] };
  const stack = [{ depth: 0, node: root }];
  
  for (const node of flatNodes) {
    while (stack.length > 1 && node.depth <= stack.at(-1).depth) {
      stack.pop();
    }
    const parent = stack.at(-1).node;
    const newNode = { ...node, children: [] };
    parent.children.push(newNode);
    stack.push({ depth: node.depth, node: newNode });
  }
  return root.children;
}
```

- **Rule**: Headings nest under the last parent with a smaller depth\[3\]\[4\].  

### **2\. Pandoc with Section Divs**

Use Pandoc’s `--section-divs` to wrap headings in `<section>` tags, then parse the HTML/XML output to build a nested AST\[4\].  

---

## **Library Comparison**

| Tool | Language | Output Structure | Requires Custom Code | 
|---|---|---|---|
| **md-hierarchical-parser** | JavaScript | Native nested JSON | No | 
| **MDAST + Custom Logic** | JavaScript | Modified MDAST | Yes | 
| **Pandoc** | Haskell/Any | XML/HTML with `<section>` | Yes (filter needed) | 

For most cases, **md-hierarchical-parser** provides the simplest solution\[1\]. For advanced use cases (e.g., sorting sections alphabetically), combine Pandoc’s section-divs with custom filters\[4\].

Sources
\[1\] Markdown parser with the support for hierarchical heading structure <https://github.com/hj-n/md-hierarchical-parser>
\[2\] Getting Started - CRAN <https://cran.r-project.org/web/packages/parsermd/vignettes/parsermd.html>
\[3\] How to fix this algorithm to for create nested sections in an AST <https://stackoverflow.com/questions/66622399/how-to-fix-this-algorithm-to-for-create-nested-sections-in-an-ast>
\[4\] How to sort sections of markdown by heading? - Stack Overflow <https://stackoverflow.com/questions/68068196/how-to-sort-sections-of-markdown-by-heading>
\[5\] syntax-tree/mdast: Markdown Abstract Syntax Tree format - GitHub <https://github.com/syntax-tree/mdast>
\[6\] markdown_ast - Rust - [Docs.rs](http://Docs.rs) <https://docs.rs/markdown-ast/latest/markdown_ast/>
\[7\] How to create an AST with ANTLR from a hierarchical key-value ... <https://stackoverflow.com/questions/13511252/how-to-create-an-ast-with-antlr-from-a-hierarchical-key-value-syntax>
\[8\] Basic writing and formatting syntax - GitHub Docs <https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax>
\[9\] How can I parse Markdown into an AST, manipulate it, and write it ... <https://stackoverflow.com/questions/67797326/how-can-i-parse-markdown-into-an-ast-manipulate-it-and-write-it-back-to-markdo>
\[10\] An attempt to build cursor's @codebase feature - part 1/2 <https://blog.lancedb.com/rag-codebase-1/>
\[11\] syntax-tree/mdast: Markdown Abstract Syntax Tree format - GitHub <https://github.com/syntax-tree/mdast>
\[12\] [Rules.md](http://Rules.md) - DavidAnson/markdownlint - GitHub <https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md>
\[13\] Paru—Pandoc wrapped around in Ruby - Heer de [Beer.org](http://Beer.org) <https://heerdebeer.org/Software/markdown/paru/>
\[14\] Transform a flat list to nested List - python - Stack Overflow <https://stackoverflow.com/questions/6720574/transform-a-flat-list-to-nested-list>
\[15\] How to nest code within a list using Markdown - Meta Stack Exchange <https://meta.stackexchange.com/questions/3792/how-to-nest-code-within-a-list-using-markdown>
\[16\] Customization Overview - CommonMark for PHP <https://commonmark.thephpleague.com/2.6/customization/overview/>
\[17\] How to write nested numbered lists - Meta Stack Exchange <https://meta.stackexchange.com/questions/85474/how-to-write-nested-numbered-lists>
\[18\] Markdown -> AST round-trip always reproduce the original ... <https://talk.commonmark.org/t/can-ast-markdown-ast-round-trip-always-reproduce-the-original/3959>
\[19\] 4.20 Manipulate Markdown via Pandoc Lua filters (\*) - Bookdown <https://bookdown.org/yihui/rmarkdown-cookbook/lua-filters.html>
\[20\] Pandoc Template to Convert Markdown Blockquotes nested in Lists <https://tex.stackexchange.com/questions/729834/pandoc-template-to-convert-markdown-blockquotes-nested-in-lists>
\[21\] Scripting with pandoc <https://pandoc.org/scripting-1.11.html>
\[22\] Term for this style of blocks (Similar to Markdown headers) - Reddit <https://www.reddit.com/r/ProgrammingLanguages/comments/xim275/term_for_this_style_of_blocks_similar_to_markdown/>
\[23\] Converting and customizing Markdown files to HTML with Unified ... <https://pedromarquez.dev/blog/2022/9/markdown-to-html>
\[24\] Converting flat array of object to array of nested objects \[duplicate\] <https://stackoverflow.com/questions/67917103/converting-flat-array-of-object-to-array-of-nested-objects>
\[25\] Basic writing and formatting syntax - GitHub Docs <https://docs.github.com/articles/basic-writing-and-formatting-syntax>
\[26\] syntax-tree/mdast-util-to-hast - GitHub <https://github.com/syntax-tree/mdast-util-to-hast>
\[27\] Nested lists require 4 spaces of indent · Issue #3 · Python-Markdown ... <https://github.com/Python-Markdown/markdown/issues/3>
\[28\] Authoring Content in Markdown - Astro Starlight <https://starlight.astro.build/guides/authoring-content/>
\[29\] Markdown Nested List: Essential Formatting Guide - Mental Pivot <https://mentalpivot.com/formatting-nested-lists-with-the-ghost-editor/>
\[30\] ASTs, Markdown and MDX - [Telerik.com](http://Telerik.com) <https://www.telerik.com/blogs/asts-markdown-and-mdx>
\[31\] Algorithm for parsing inline markdown elements, avoiding sef-nesting <https://stackoverflow.com/questions/77403283/algorithm-for-parsing-inline-markdown-elements-avoiding-sef-nesting>
\[32\] Getting Started • parsermd <https://rundel.github.io/parsermd/articles/parsermd.html>
\[33\] Markdown nested headings in function list <https://forums.ultraedit.com/markdown-nested-headings-in-function-list-t18103.html>
\[34\] Typography - MyST Markdown <https://mystmd.org/guide/typography>