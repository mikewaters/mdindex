# GPT4o Image Generation

[CLIP Embedding Model.md](./CLIP%20Embedding%20Model.md)

### Architecture

> It's a hybrid model. The AR component generates control embeddings that then get decoded by a diffusion model. But the control embeddings are accurate enough to edit and reconstruct the images surprisingly well.\
> Given that there are some details on how images are embedded, using multi-scale CLIP-like embeddings (LLAVA 1.6 or was it LLAVA-Next did this too), it's likely that's how they're generating the images too. Essentially, if you can encode the images into a latent space (the CLIP embeddings), then you can get the LLM to output these embeddings as well (other MM-LLMs have done this). Wurstchen showed that these compressed latent spaces have strong correlation to the final decoded image, which is how the image preview shows up before the final image, and why it's not a perfect representation of the final.\
> The TL;DR is that it's probably very similar to Wurstschen where 4o replaces the Stage C model (autoregressive generation of CLIP embeddings), followed by an auxiliary (and likely very large - maybe bigger than Flux) latent diffusion decoder.
>
> <https://www.reddit.com/r/MachineLearning/comments/1jkt42w/comment/mjyvwx5/>