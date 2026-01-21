---
tags:
  - document 📑
---
# Web Text Editor Implementation Approaches

> In tldraw the text is always in the DOM, so the browser can use your native font rendering (for example ClearType on Windows). In Excalidraw the text is only in the DOM while you're editing it.
>
> <https://news.ycombinator.com/item?id=42469074>

+ ## 1\. ContentEditable Approach

   ### Overview

   ContentEditable is a native HTML attribute that makes any element editable by the user.

   ```html
   <div contenteditable="true">
     This text can be edited by the user directly
   </div>
   ```

   ### Advantages

   - Native browser editing behaviors (copy/paste, cursor movement)

   - Built-in mobile support

   - Native spellcheck

   - Native IME (Input Method Editor) support for international text

   - Automatic handling of keyboard shortcuts

   - No need to reimplement basic editing operations

   ### Disadvantages

   - Inconsistent behavior across browsers

   - Limited control over DOM mutations

   - Unpredictable HTML output

   - Difficult to implement complex features

   - Hard to maintain a consistent document state

   - Challenging to implement collaborative editing

   ### Examples

   - Draft.js (Facebook's original editor)

   - Early versions of [medium.com](http://medium.com) editor

   - Basic CKEditor implementations

   - TinyMCE's traditional mode

   ## 2\. Custom DOM Rendering

   ### Overview

   Complete control over rendering by managing a virtual DOM and custom selection/cursor model.

   ```javascript
   class Editor {
     constructor() {
       this.content = [];
       this.selection = { start: 0, end: 0 };
     }
     
     render() {
       return this.content.map(node => {
         const element = document.createElement(node.type);
         element.textContent = node.text;
         return element;
       });
     }
     
     handleKeyPress(event) {
       // Custom handling of all keyboard input
       const char = event.key;
       this.insert(char, this.selection.start);
       this.selection.start++;
       this.selection.end = this.selection.start;
       this.render();
     }
   }
   ```

   ### Advantages

   - Complete control over editing behavior

   - Predictable document model

   - Easier to implement collaborative editing

   - Consistent cross-browser behavior

   - Better suited for complex features

   ### Disadvantages

   - Must implement all editing operations

   - Need to handle selection, cursor, copy/paste

   - Complex IME integration

   - Mobile support requires extra work

   - Performance challenges with large documents

   - Higher implementation complexity

   ### Examples

   - Prosemirror's core implementation

   - Slate's newer versions

   - CodeMirror 6

   - Monaco Editor (VS Code's editor)

   ## 3\. Hybrid Approach (Modern Editors)

   ### Overview

   Uses contenteditable as an input mechanism but maintains a separate document model and carefully manages DOM updates.

   ```javascript
   class HybridEditor {
     constructor() {
       this.model = createEditorState();
       this.contentEditableElement = document.querySelector('#editor');
       
       // Listen for DOM mutations
       this.observer = new MutationObserver(mutations => {
         // Convert DOM changes to model updates
         const modelUpdates = this.convertToModelUpdates(mutations);
         this.updateModel(modelUpdates);
       });
       
       // Render model changes
       this.model.subscribe(state => {
         this.renderToDOM(state);
       });
     }
     
     convertToModelUpdates(mutations) {
       // Carefully translate DOM changes to model operations
       return mutations.map(mutation => {
         return {
           type: 'insert',
           position: this.mapDOMPositionToModel(mutation.target),
           content: mutation.addedNodes
         };
       });
     }
   }
   ```

   ### Advantages

   - Benefits of native editing behavior

   - Controlled document model

   - Better handling of complex features

   - Good mobile support

   - Reasonable performance

   - Easier implementation than fully custom

   ### Disadvantages

   - Still needs careful browser compatibility work

   - Complex synchronization between model and DOM

   - Requires careful handling of edge cases

   - Selection management can be tricky

   ### Examples

   - Lexical (Meta's modern approach)

   - Tiptap 2.0

   - Slate's modern implementation

   - CKEditor 5

   - Quill's Parchment system

   ## 4\. Canvas-based Approach

   ### Overview

   Renders text and UI elements to a canvas, handling all input and rendering manually.

   ```javascript
   class CanvasEditor {
     constructor(canvas) {
       this.canvas = canvas;
       this.ctx = canvas.getContext('2d');
       this.content = '';
       this.cursorPos = 0;
       
       canvas.addEventListener('keydown', this.handleKeyDown.bind(this));
     }
     
     render() {
       this.ctx.clearRect(0, 0, canvas.width, canvas.height);
       this.ctx.font = '16px monospace';
       this.ctx.fillText(this.content, 10, 20);
       
       // Render cursor
       if (this.shouldShowCursor) {
         const textMetrics = this.ctx.measureText(
           this.content.substring(0, this.cursorPos)
         );
         this.ctx.fillRect(
           10 + textMetrics.width,
           4,
           2,
           16
         );
       }
     }
   }
   ```

   ### Advantages

   - Complete control over rendering

   - Consistent cross-platform behavior

   - Potential for better performance with large documents

   - Good for specialized editing (code, math)

   ### Disadvantages

   - Must implement all text rendering

   - Complex text measurement

   - No native copy/paste

   - Difficult accessibility

   - Limited mobile support

   - Heavy memory usage

   ### Examples

   - Monaco Editor (in some modes)

   - Custom code editors

   - Mathematical formula editors

   - Specialized diagram editors

   ## Selection Guide

   ### Choose ContentEditable When:

   - Building simple text editing

   - Need quick implementation

   - Basic formatting requirements

   - Mobile support is crucial

   ### Choose Custom DOM When:

   - Need complete control

   - Building specialized editors

   - Implementing real-time collaboration

   - Complex document model required

   ### Choose Hybrid Approach When:

   - Building modern rich text editors

   - Need balance of control and native behavior

   - Implementing complex features

   - Need good cross-browser support

   ### Choose Canvas-based When:

   - Building specialized editors (code, math)

   - Need precise rendering control

   - Performance with large documents is crucial

   - Specialized input handling required