# Msty.app

MacOS app

Removed Jan-2025 - closed source

> The simplest way to use local and online LLMs

https://msty.app

MacOS app, uses [Ollama 1.md](./Ollama%201.md) internally but runs its own service. You can access Ollama models from an existing instance though.

## Setting up Msty + existing Ollama

[Find your Ollama models path](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/942ec86e-7b34-4374-b370-dc329519192e#db2f4afd-c4a6-41a7-a3a6-0871b55736e9)

Navigate to Text Models and select the “Edit Model Path” option in the service menu (top-right).

Enter that path to the models directory (ex: `/Users/mike/.ollama/models/`), once you click “Confirm” Msty will update/restart the service and your Ollama models should be available for selection.

Old model path: `/Users/mike/Library/Application Support/Msty/models`




