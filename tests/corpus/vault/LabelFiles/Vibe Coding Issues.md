# Vibe Coding Issues

## UI Dev

### Each browser invocation consumes a port, and so you cannot know port number in advance with multi-agent

Option: Makefile chooses an available port, and echos it to stdout. Ex:

> • Pre-scan for the first free port ≥3000 using a tiny inline Node script (safer than relying on Stencil’s silent
>
> fallback).
>
> • Pass the chosen port explicitly to Stencil via --port.
>
> • Echo a clear line with the port before starting (Stencil’s own logs are noisier to parse).
>
> • Avoid extra files by embedding the logic directly in the Makefile.