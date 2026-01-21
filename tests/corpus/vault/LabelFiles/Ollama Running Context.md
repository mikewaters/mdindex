# Ollama Running Context

<https://news.ycombinator.com/item?id=39501356>

> Ollama provides a running binary context that represents all the processing your GPU has done in the conversation so far, but the last time I looked into LangChain they wanted you to build up a plaintext context on the client and ask Ollama to re-process it on every request. This works well as part of the wider suite of LangChain tools, but isn't great if you really just need Ollama.
>
> For JS/TS and Python, Ollama now provides an official client library which presumably does take advantage of all Ollama features \[0\]. For other languages, the Ollama API is actually very simple \[1\] and easy to wrap yourself, and I'd recommend doing so unless you specifically need some of the abstractions LangChain provides.
>
> \[0\] <https://ollama.com/blog/python-javascript-libraries>
>
> \[1\] <https://github.com/ollama/ollama/blob/main/docs/api.md#gener>...

>  That binary data is basically a big vector representation of a textual context's contents.  

> Multimodal models use embeddings as well, the difference there is that they've been trained to associate the same position in latent space to text and to the image that text describes, that way it can turn a textual response into an image and vice versa.  A lot of models use CLIP, an embedding method from openAI.