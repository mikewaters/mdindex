---
tags:
  - document 📑
---
# DOM Manipulation 

There are many new additions to the HTML and related specs, related to wasm and web components.



<https://github.com/solidjs/solid>

Solid is a declarative JavaScript library for creating user interfaces. Instead of using a Virtual DOM, it compiles its templates to real DOM nodes and updates them with fine-grained reactions. Declare your state and use it throughout your app, and when a piece of state changes, only the code that depends on it will rerun.

Solid uses signals

> [Dioxus](https://github.com/dioxuslabs/dioxus) is a UI library for Rust that makes it easy to target almost any platform with the same React-like codebase. You can build apps for WASM, desktop, mobile, TUI, static-sites, SSR, LiveView, and more

<https://dioxuslabs.com/blog/templates-diffing>

# JSX

JSX is just a different DSL to the createElement function call pattern (see Preact.h for example)

# State

since Dioxus relies on a VirtualDom, it can be used as the primary state system for any renderer. And we have a ton of options for renderers:

- Desktop (webview)

- Mobile (webview)

- Web

- TUI

- Skia

- LiveView

- Blitz (WGPU)

- SSR + Hydration

- Static site generation

- VR/AR (coming soon!)

# Rendering

# LiveView Pattern

![image 6.png](./DOM%20Manipulation%20-assets/image%206.png)