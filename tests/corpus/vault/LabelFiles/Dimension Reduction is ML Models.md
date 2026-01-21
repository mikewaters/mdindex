# Dimension Reduction is ML Models

It would be ideal if we can somehow visualize the embeddings and verify the clusters of similar images. Even though ML models can comfortably work with 100s of dimensions, to visualize them we may have to further reduce the dimensions ,using techniques like T-SNE or UMAP , so that we can plot embeddings in two or three dimensional space.

Here is a handy T-SNE method to do just that

from sklearn.manifold import TSNE
tsne = TSNE(random_state = 0, metric = 'cosine',perplexity=2,n_components = 3)
embeddings_3d = tsne.fit_transform(array_of_embeddings)

<https://martinfowler.com/articles/gen-ai-patterns/>