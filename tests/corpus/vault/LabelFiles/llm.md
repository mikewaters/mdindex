# llm

<https://llm.datasette.io/en/stable/usage.html>

<https://github.com/simonw/llm>

` llm -m claude-3.5-sonnet "prompt"`

### Find of list models

```python
> llm models -q "3.5-sonn"
Anthropic Messages: claude-3-5-sonnet-latest (aliases: claude-3.5-sonnet, claude-3.5-sonnet-latest)
```

### Install

```python
#brew install llm
uv tool install llm
llm keys set openai
llm keys set claude

llm install llm-gpt4all
llm install llm-claude-3
llm install llm-ollama

# plugin sentence-transformers
# EDIT: Torch is b0rked on beaker
#llm install llm-python
#llm python -m pip install \
#  --pre torch torchvision \
#  --index-url https://download.pytorch.org/whl/nightly/cpu
#llm install llm-sentence-transformers

llm models  # lists installed models
```