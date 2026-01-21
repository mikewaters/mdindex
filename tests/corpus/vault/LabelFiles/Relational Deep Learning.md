# Relational Deep Learning

### Relational Graph Transformers

<https://kumo.ai/research/relational-graph-transformers/>

> By turning the relational database into a richly structured graph, we enable **deep learning models to reason directly over the underlying relationships**—without flattening, joins, or feature engineering. This transformation preserves the semantics of the original data, respects its structure, and supports heterogeneous and temporal modeling out of the box.

### Aggregating Relational Data

[https://www.youtube.com/embed/DS_0-7XIc5s](https://www.youtube.com/embed/DS_0-7XIc5s)

[Feature Engineering.md](./Feature%20Engineering.md) to generate tabular data from relational data (somehow aggregating joined data) and then analyze tabular data using [XGBoost.md](./XGBoost.md) or [LightGBM.md](./LightGBM.md)

Foundation model for relation database analysis. Defines a new query language (PQL - sql for predictions), and internally translates relational data into a temporal graph- which is the same technique I’ve seen in research around relational DL.

<https://kumo.ai/company/news/kumo-relational-foundation-model/>

A tabular foundation model

<https://github.com/PriorLabs/TabPFN>

See NotebookLM

[https://www.youtube.com/embed/aw-5gKMRsI8](https://www.youtube.com/embed/aw-5gKMRsI8)