# Best Practices for Monitoring Claude Code with OpenTelemetry on macOS (import)

Claude Code includes **native OpenTelemetry support** that enables comprehensive monitoring of AI coding sessions, token usage, costs, and performance metrics. This guide covers the best practices for setting up OpenTelemetry tools on macOS to monitor Claude Code effectively.

## Overview of Claude Code's OpenTelemetry Capabilities

Claude Code provides built-in telemetry data including:[^1](https://docs.claude.com/en/docs/claude-code/monitoring-usage)

- **Cost Analysis**: Total token usage and USD cost tracking

- **Session Analytics**: Daily/Weekly/Monthly Active Users (DAU/WAU/MAU)

- **Performance Metrics**: API latency, success rates, and command duration

- **Tool Usage**: Which Claude Code tools are used most frequently

- **Model Distribution**: Usage patterns across different Claude models (Sonnet, Opus, etc.)

- **User Activity**: Request counts, terminal types (VS Code, Apple Terminal), and decision patterns

## Quick Start Configuration

### Basic Environment Variables Setup

The simplest way to enable Claude Code monitoring is through environment variables:[^1](https://docs.claude.com/en/docs/claude-code/monitoring-usage)

```bash
# Enable telemetry
export CLAUDE_CODE_ENABLE_TELEMETRY=1

# Configure exporters
export OTEL_METRICS_EXPORTER=otlp
export OTEL_LOGS_EXPORTER=otlp

# Set OTLP endpoint and protocol
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Optional: Reduce intervals for debugging
export OTEL_METRIC_EXPORT_INTERVAL=10000  # 10 seconds
export OTEL_LOGS_EXPORT_INTERVAL=5000     # 5 seconds

# Run Claude Code
claude
```

### Administrator Configuration (Organization-wide)

For team deployments, use the managed settings file at `/Library/Application Support/ClaudeCode/managed-settings.json`:[^1](https://docs.claude.com/en/docs/claude-code/monitoring-usage)

```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector.company.com:4317",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer company-token"
  }
}
```

## Recommended OpenTelemetry Tools for macOS

### 1\. SigNoz (Comprehensive Solution)

**SigNoz** is a full-stack open-source observability platform with excellent Claude Code integration:[^2](https://signoz.io/docs/claude-code-monitoring/)

**Setup with SigNoz Cloud**:

```bash
CLAUDE_CODE_ENABLE_TELEMETRY=1 \
OTEL_METRICS_EXPORTER=otlp \
OTEL_LOGS_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
OTEL_EXPORTER_OTLP_ENDPOINT="https://ingest.<region>.signoz.cloud:443" \
OTEL_EXPORTER_OTLP_HEADERS="signoz-ingestion-key=<your-ingestion-key>" \
OTEL_METRIC_EXPORT_INTERVAL=10000 \
OTEL_LOGS_EXPORT_INTERVAL=5000 \
claude
```

**Key Features**:

- Out-of-the-box Claude Code dashboards

- Real-time metrics and logs correlation

- Built on ClickHouse for high performance

- Comprehensive visualization with Flamegraphs and Gantt charts

### 2\. Grafana Cloud (Simple Setup)

**Grafana Cloud** offers an easy-to-use solution with free tier support:[^4](https://quesma.com/blog/track-claude-code-usage-and-limits-with-grafana-cloud/)

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=otlp
export OTEL_LOGS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp-gateway-prod-eu-north-0.grafana.net/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic YOUR_TOKEN_HERE"
export OTEL_METRIC_EXPORT_INTERVAL=10000
export OTEL_LOGS_EXPORT_INTERVAL=5000
```

**Advantages**:

- No local collector required

- LogQL support for extracting metrics from logs

- Built-in alerting and dashboards

- Free tier available

### 3\. Jaeger + OpenTelemetry Collector (Local Setup)

For local development, **Jaeger** with the OpenTelemetry Collector provides a lightweight solution:[^5](https://www.jaegertracing.io/docs/1.21/deployment/opentelemetry/)

**Install Jaeger on macOS**:

```bash
# Download Jaeger binary
wget https://github.com/jaegertracing/jaeger/releases/download/v1.56.0/jaeger-1.56.0-darwin-amd64.tar.gz
tar -xzf jaeger-1.56.0-darwin-amd64.tar.gz
cd jaeger-1.56.0-darwin-amd64

# Run Jaeger all-in-one
./jaeger-all-in-one
```

**OpenTelemetry Collector Configuration**:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  jaeger:
    endpoint: localhost:14250
    tls:
      insecure: true
  
processors:
  batch:

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger]
```

**Access Jaeger UI**: `http://localhost:16686`

### 4\. DataDog (Enterprise Solution)

**DataDog** offers comprehensive Claude Code monitoring with three setup options:[^7](https://ma.rtin.so/posts/monitoring-claude-code-with-datadog/)

```bash
# Configure DataDog endpoint
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=otlp
export OTEL_LOGS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.datadoghq.com"
export OTEL_EXPORTER_OTLP_HEADERS="DD-API-KEY=your-api-key"
```

## Best Practices and Tips

### 1\. Start Simple with Console Logging

For initial testing, use the console exporter:[^1](https://docs.claude.com/en/docs/claude-code/monitoring-usage)

```bash
export OTEL_METRICS_EXPORTER=console
export OTEL_LOGS_EXPORTER=console
```

### 2\. Monitor Key Metrics

Focus on these essential Claude Code metrics:

- `claude_code.session.count`: CLI session frequency

- `claude_code.cost.usage`: Session costs in USD

- `claude_code.lines_of_code.count`: Code modifications

- `claude_code.pull_request.count`: PR creation tracking

- Token consumption and quota usage

### 3\. Enable User Prompt Logging (Optional)

For detailed analysis, enable prompt logging:[^8](https://docs.langchain.com/langsmith/trace-claude-code)

```bash
export OTEL_LOG_USER_PROMPTS=1
```

**Note**: Only enable this in secure environments due to privacy considerations.

### 4\. Use Docker for Consistent Setup

For teams, containerized setups provide consistency:[^9](https://betterstack.com/community/guides/observability/opentelemetry-collector/)

```bash
# Pull OpenTelemetry Collector
docker pull otel/opentelemetry-collector-contrib:latest

# Run with custom config
docker run -v $(pwd)/config.yaml:/etc/otelcol-contrib/config.yaml \
  otel/opentelemetry-collector-contrib:latest
```

## Community Resources and Examples

### GitHub Repositories

1. **ColeMurray/claude-code-otel**: Comprehensive observability stack with detailed dashboards[^10](https://github.com/ColeMurray/claude-code-otel)

2. **disler/claude-code-hooks-multi-agent-observability**: Real-time monitoring system with WebSocket updates[^11](https://github.com/disler/claude-code-hooks-multi-agent-observability)

3. **Maciek-roboblog/Claude-Code-Usage-Monitor**: Terminal-based monitoring tool with ML predictions[^12](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

### Documentation and Guides

- **Official Claude Code monitoring docs**: Complete configuration reference[^1](https://docs.claude.com/en/docs/claude-code/monitoring-usage)

- **SigNoz Claude Code guide**: Step-by-step setup with dashboard examples[^2](https://signoz.io/blog/claude-code-monitoring-with-opentelemetry/)

- **Grafana Cloud integration**: Simple cloud-based monitoring setup[^4](https://quesma.com/blog/track-claude-code-usage-and-limits-with-grafana-cloud/)

### Community Discussions

- **Reddit discussions**: Real-world experiences and troubleshooting[^13](https://www.reddit.com/r/Anthropic/comments/1nnxv4a/how_we_instrumented_claude_code_with/)

- **GitHub issues**: Known issues and solutions (note: some versions have telemetry bugs)[^15](https://github.com/anthropics/claude-code/issues/7493)

## Troubleshooting Common Issues

### Version Compatibility

Some Claude Code versions have telemetry issues. If monitoring isn't working:[^15](https://github.com/anthropics/claude-code/issues/7493)

- Verify Claude Code version compatibility

- Check console logs for telemetry initialization

- Test with `console` exporter first

### Network Configuration

Ensure proper endpoint configuration:

- Use correct protocol (gRPC vs HTTP)

- Verify authentication headers

- Check firewall settings for OTLP ports (4317/4318)

### Performance Considerations

- Start with longer export intervals for production

- Use batch processors to reduce overhead

- Monitor collector resource usage

## Conclusion

Claude Code's native OpenTelemetry support makes it straightforward to implement comprehensive monitoring on macOS. **SigNoz** and **Grafana Cloud** offer the best balance of simplicity and features for most users, while **Jaeger** provides an excellent local development option. The key is starting with basic telemetry configuration and gradually expanding based on your monitoring needs.

For teams serious about AI coding observability, the community has built excellent tools and dashboards that provide deep insights into Claude Code usage patterns, costs, and performance metrics.
