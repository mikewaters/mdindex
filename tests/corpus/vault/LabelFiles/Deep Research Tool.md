# Deep Research Tool

## Notes

### Sandboxing existing tools to expose them via MCP

Sandboxing something like Perplexity within a Chrome profile, which could be called by an agent via MCP like Browser-Use. I am doing this already, to have “apps” that are just wrappers around the website.

```bash
# Create isolated instance for app.something.com
chrome --user-data-dir="/path/to/isolated/profile" --app="https://app.something.com"
```

```bash
Command-line flags:
--site-per-process - Forces strict site isolation
--isolate-origins=https://app.something.com - Isolates specific origins
--disable-site-isolation-trials - For custom isolation setups
```

- source: [Browser App Isolation! Open Source Solutions for Chrome and Safari.md](./Browser%20App%20Isolation!%20Open%20Source%20Solutions%20for%20Chrome%20and%20Safari.md)

- Chrome profile params: [Chrome Enterprise and Education Help - Protect your data with site isolation](https://support.google.com/chrome/a/answer/7581529?hl=en)

- Browser-use connecting to an existing Chrome profile: [Real Browser - Browser Use](https://docs.browser-use.com/customize/browser/real-browser)

### Replicating existing tools

<https://jan.ai/post/deepresearch>

<https://github.com/MODSetter/SurfSense>

<https://github.com/LearningCircuit/local-deep-research>

<https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/>

<https://news.ycombinator.com/item?id=45789602>