# LightGBM

Gradient boosting decision tree implementation

Related: [XGBoost.md](./XGBoost.md)

LightGBM is a high-performance, open-source gradient boosting framework developed by Microsoft. It is designed for efficiency, scalability, and accuracy, particularly with large datasets. Key features include:

- **Leaf-wise Tree Growth:** LightGBM grows trees by selecting the leaf with the most significant loss reduction, leading to faster training and deeper trees, but may require regularization to prevent overfitting\[1\]\[4\].

- **Histogram-based Algorithm:** It uses histograms to speed up decision tree construction by grouping data into bins, reducing computational complexity\[1\]\[3\].

- **Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB):** GOSS optimizes data instance selection based on gradients, while EFB bundles mutually exclusive features to reduce dimensionality and enhance efficiency\[1\]\[4\].

These innovations make LightGBM significantly faster and more efficient than traditional gradient boosting methods like XGBoost, especially on large datasets.

Sources
\[1\] Mastering LightGBM: Unravelling the Magic Behind Gradient Boosting <https://data-ai.theodo.com/en/technical-blog/mastering-lightgbm-unravelling-the-magic-behind-gradient-boosting>
\[2\] LightGBM (Light Gradient Boosting Machine) - GeeksforGeeks <https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/>
\[3\] LightGBM - What and how to use ? - Best Simple Guide <https://inside-machinelearning.com/en/lightgbm-guide/>
\[4\] LightGBM Demystified: Understanding the Math Behind the Algorithm <https://www.intelligentmachines.blog/post/lightgbm-demystified-understanding-the-math-behind-the-algorithm>
\[5\] What is Light GBM? Advantages & Disadvantages? Light ... - Kaggle <https://www.kaggle.com/general/264327>
\[6\] Features — LightGBM 4.6.0.99 documentation <https://lightgbm.readthedocs.io/en/latest/Features.html>
\[7\] An Overview of LightGBM - avanwyk <https://www.avanwyk.com/an-overview-of-lightgbm/>