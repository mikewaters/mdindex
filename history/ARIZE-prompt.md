You are a senior Python engineer. Your task is to add OPTIONAL, configuration-driven OpenTelemetry instrumentation for local `mlx-lm` and `mlx-embedding` interactions, exporting traces compatible with Arize Phoenix.

Hard requirements
- The instrumentation must be OFF by default.
- A user must be able to enable it via a CLI option (e.g., `--phoenix-tracing` and optionally `--phoenix-endpoint`).
- Behavior must be configuration-driven using the project’s existing config patterns:
  - Integrate with the existing config system (do not invent a new config framework).
  - If the project uses env vars / pydantic settings / YAML / TOML / argparse, follow that pattern.
  - CLI options should override config values, and environment variables should override anything (consistent with existing precedence rules).
- Any custom code for this adaptation must be stored in a NEW python module file within the project’s `core` module/package (e.g., `core/instrumentation.py`).
- Do NOT require a long-running server to run the app; traces should be exported to a Phoenix endpoint if provided. If no endpoint is configured, allow a safe local default if that matches Phoenix best practices, but do not auto-start any server process. 
- Make minimal invasive changes: wrap/instrument at the narrowest layer where `mlx_lm` generation and embedding calls occur.

Goals
1) Capture each `mlx-lm` generation call as a trace/span with:
   - span name: `mlx_lm.generate` (or similarly consistent)
   - attributes: model identifier, device, dtype/quantization if known, prompt length, max_tokens, temperature, top_p, seed, stop strings count, streaming vs non-streaming
   - timings: latency in milliseconds
   - outcome: success/failure and exception info when failed
   - output metadata: output length and (if you can compute) token counts; if token counts aren’t available, store char counts and explicitly mark token_count_unknown=true
2) Capture each `mlx-embedding` call as a trace/span with:
   - span name: `mlx_embedding.embed`
   - attributes: embedding model id, batch size, input lengths (summary), pooling strategy if any
   - timings + success/failure
3) If there are higher-level “request/session” constructs in the app, nest `mlx_*` spans under them where possible.

Implementation guidance (use these choices unless project constraints force otherwise)
- Use OpenTelemetry Python SDK instrumentation manually (no auto-instrumentation expected for MLX).
- Use a robust “enable/disable” gate:
  - If disabled, calls must behave identically with negligible overhead.
  - If enabled but Phoenix endpoint is unreachable, do not crash; degrade gracefully (log a warning once).
- Prefer OpenInference semantic conventions if easy, but do not block on it. Ensure Phoenix can display traces from OTel exporters.
- Structure:
  - Create new module: `core/instrumentation.py`
    - `def configure_phoenix_tracing(config) -> Tracer | None: ...`
    - `def instrument_mlx_lm(tracer, *, module_or_functions) -> contextmanager/wrapper: ...`
    - `def instrument_mlx_embedding(tracer, *, module_or_functions) -> contextmanager/wrapper: ...`
    - Provide wrapper functions or decorators that wrap the actual generation + embedding call sites.
  - Update CLI parsing to add a flag (name aligned with project style) and plumb it into the config.
  - Update the entrypoint/bootstrap to call `configure_phoenix_tracing()` early if enabled.

Concrete tasks
A) Repo discovery
- Inspect the repository structure to locate:
  - where `mlx_lm` is imported and called (generation)
  - where embeddings are computed (e.g., `mlx_embeddings`, `mlx_embedding_models`, etc.)
  - existing config system and precedence rules
  - CLI framework (argparse/typer/click) and how options are passed down
- Summarize findings briefly in your final output (files changed, key call sites).

B) New module (core)
- Implement `core/instrumentation.py` with:
  - config dataclass/Settings integration aligned with existing patterns
  - OpenTelemetry setup:
    - `TracerProvider`
    - Resource attributes: service.name (project name), service.version if known, environment if available
    - exporter selection:
      - If Phoenix supports OTLP HTTP/gRPC endpoint, implement OTLP exporter config (endpoint + headers if configured).
      - Provide sane defaults, but do not assume a running server.
    - `BatchSpanProcessor` (or `SimpleSpanProcessor` if the app is short-lived; choose based on app nature and justify)
  - Span creation wrappers:
    - Use `with tracer.start_as_current_span(...):`
    - Attach attributes listed above
    - Record exceptions with `record_exception` and set status appropriately
    - Ensure wrapper supports both sync and generator/streaming outputs:
      - If `mlx-lm` yields tokens/strings, wrap the iterator to measure time and final lengths without consuming early.
- Include unit-testable design:
  - functions should be testable without requiring a live Phoenix server (mock exporter or in-memory span exporter).

C) CLI/config integration
- Add CLI option(s) to enable tracing.
- Update config loading so:
  - CLI flag enables tracing even if config disabled
  - Endpoint can be set via config and/or CLI option
  - Optional: `--phoenix-tracing-sample-rate` or `--phoenix-tracing-sampler` if project already supports sampling; otherwise default to AlwaysOnSampler when enabled.

D) Tests & quality
- Add tests for:
  - Disabled path: wrappers don’t create spans and have near-zero overhead (assert tracer is None / no spans exported)
  - Enabled path: a mocked `mlx_lm` call creates spans with expected names/attributes
  - Streaming path: wrapper doesn’t break iteration and still captures final metadata
- Add minimal documentation:
  - Update README or CLI help text showing how to enable tracing and point to Phoenix endpoint.
  - Mention that Phoenix server is optional and can be run separately if desired.

Deliverable
- Provide a concise PR-style summary:
  - new file(s)
  - modified file(s)
  - how to enable via CLI
  - what spans/attributes are emitted
  - how to point to a Phoenix endpoint
- Keep code style consistent with the project and run formatting/linting if configured.
