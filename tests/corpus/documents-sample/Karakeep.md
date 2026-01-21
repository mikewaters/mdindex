# Karakeep

Preferred run method is [Docker with Colima.md](./Docker%20with%20Colima.md) Compose; I am using [Docker Desktop Alternative ! H... → podman desktop.md](../Highlight/Docker%20Desktop%20Alternative%20!%20H...%20→%20podman%20desktop.md) on work machine to run the containers

ref: [Ollama 1.md](./Ollama%201.md)

### Installation

ref: [Ollama model list](https://ollama.com/library)

#### Setup

```shell
mkdir ~/Develop/karakeep
# download their docker-compose.yml
# create an .env file and populate for local models
openssl rand -base64 36  # for the local secrets
ollama pull gemma3:4b
```

#### .env file

```shell

```

#### Run

~~`docker compose up -d`~~

`podman compose --file docker-compose.yml up --detach`

### Debugging

`cat ~/.ollama/logs/server.log`

`OLLAMA_DEBUG=1; OLLAMA_HOST=0.0.0.0 ollama serve`

`curl http://0.0.0.0:11434/api/generate -d '{"model": "gemma3:4b","prompt":"Why is the sky blue?"}'`

### Updating

> Updating karakeep will depend on what you used for the `KARAKEEP_VERSION` env variable.
>
> - If you pinned the app to a specific version, bump the version and re-run `docker compose up -d`. This should pull the new version for you.
>
> - If you used `KARAKEEP_VERSION=release`, you'll need to force docker to pull the latest version by running `docker compose up --pull always -d`.


