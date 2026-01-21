# File-based RAG Idea

Can this be done with [Anything LLM.md](./Anything%20LLM.md)

<https://news.ycombinator.com/item?id=42214486>

>  is how I am implementing something close to what you mentioned. In my setup, I make sure to create a [readme.md](readme.md) at the root of every folder which is a document for me as well as LLM that tells me what is inside the folder and how it is relevant to my life or project. kind of a drunken brain dump for the folder .
>
>  I have a cron job that executes every night and iterates through my filesystem looking for changes since the last time it ran. If it finds new files or changes, it creates embeddings and stores them in Milvus.
>
> The chat with LLM using Embeddings if not that great yet. To be fair,I have not yet tried to implement the GraphRAG or Claude's contexual RAG approaches. I have a lot of code in different programming languages, text documents, bills pdf, images. Not sure if one RAG can handle it all.
>
> I am using AWS Bedrock APIs for LLama and Claude and locally hosted MilvusŴe