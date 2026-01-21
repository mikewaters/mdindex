---
tags:
  - document 📑
---
# Data Serialization

## Wire: Protobuf, rpc etc

- [DocArray](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/01b53e30-3850-4d6a-8922-a0093f0ac5b8#bfb5c567-69df-400d-9df2-9c742e1ba343)

   - data transmission as JSON over **HTTP** or as **++[Protobuf](https://protobuf.dev/)++** over **++[gRPC](https://grpc.io/)++**.

## Model: Pydantic and friends

### DocArray

<https://github.com/docarray/docarray>

Used by Jina.ai, part of Linux Foundation; uses Pydantic, to run on FastAPI or JinaServe.

<https://docs.docarray.org/user_guide/sending/api/jina/>

> DocArray empowers you to **represent your data** in a manner that is inherently attuned to machine learning.

> This is particularly beneficial for various scenarios:
>
> - 🏃 You are **training a model**: You're dealing with tensors of varying shapes and sizes, each signifying different elements. You desire a method to logically organize them.
>
> - ☁️ You are **serving a model**: Let's say through FastAPI, and you wish to define your API endpoints precisely.
>
> - 🗂️ You are **parsing data**: Perhaps for future deployment in your machine learning or data science projects.

> for the ++[representation](https://github.com/docarray/docarray#represent)++, ++[transmission](https://github.com/docarray/docarray#send)++, ++[storage](https://github.com/docarray/docarray#store)++, and ++[retrieval](https://github.com/docarray/docarray#retrieve)++ of multimodal data
>
> Tailored for the development of multimodal AI applications, its design guarantees seamless integration with the extensive Python and machine learning ecosystems

- data transmission as JSON over **HTTP** or as **++[Protobuf](https://protobuf.dev/)++** over **++[gRPC](https://grpc.io/)++**.

- support for vector databases such as \*\*Weaviate, Qdrant, ElasticSearch, Redis, Mongo Atlas, and HNSWLib.

- Based on **++[Pydantic](https://github.com/pydantic/pydantic)++**, and instantly compatible with web and microservice frameworks like **++[FastAPI](https://github.com/tiangolo/fastapi/)++** and **++[Jina](https://github.com/jina-ai/jina/)++**.