# k-Nearest Neighbors (k-NN) 

**k-Nearest Neighbors (k-NN)** is a non-parametric, supervised machine learning algorithm used for both classification and regression tasks. It operates on the principle that similar data points are likely to be found close to each other in feature space, making proximity a key factor in predicting the output for new data\[1\]\[2\]\[4\]\[5\].

---

**How k-NN Works**

- The algorithm stores the entire labeled dataset (the training data).

- When a new (unlabeled) data point needs to be classified or predicted, k-NN calculates the distance between this point and all points in the training set. Common distance metrics include Euclidean, Manhattan, or Minkowski distance.

- It then identifies the $$ k $$ closest data points (neighbors) to the new point.

- For classification:

   - The new data point is assigned the class most common among its $$ k $$ nearest neighbors (majority vote)\[2\]\[4\].

- For regression:

   - The predicted value is the average (or sometimes a weighted average) of the values of the $$ k $$ nearest neighbors\[1\]\[2\]\[4\].

---

**Key Properties**

- **Non-parametric:** k-NN does not make assumptions about the underlying data distribution and does not build a model during training. Instead, it "memorizes" the training data and performs computation at prediction time, which is why it is called a "lazy" learning algorithm\[1\]\[4\].

- **Supervised:** The algorithm requires labeled data for training\[1\]\[4\].

- **Versatile:** Can be used for both classification and regression problems\[1\]\[2\]\[4\].

- **Distance-based:** The choice of distance metric and feature scaling can significantly impact performance, especially if features are on different scales\[2\].

---

**Choosing the Value of $$ k $$**

- The parameter $$ k $$ determines how many neighbors are considered when making a prediction.

- A small $$ k $$ (e.g., $$ k=1 $$) can make the model sensitive to noise, while a large $$ k $$ can smooth out predictions but may ignore local patterns.

- The optimal $$ k $$ is often chosen by testing different values and selecting the one that minimizes prediction error on validation data (using techniques like the "elbow method" or grid search)\[1\].

---

**Strengths and Weaknesses**

| Strengths | Weaknesses | 
|---|---|
| Simple to understand and implement | Computationally expensive for large datasets | 
| No explicit training phase | Sensitive to irrelevant or redundant features | 
| Flexible for classification/regression | Performance degrades with high-dimensional data (curse of dimensionality)\[2\] | 

---

**Applications**

- Recommender systems

- Image and text recognition

- Medical diagnosis

- Financial risk prediction

- Climate forecasting\[1\]

---

**Example**

Suppose you have a dataset of fruits labeled by sweetness and crunchiness. To classify a new fruit, k-NN will:

- Calculate the distance from the new fruit to all labeled fruits.

- Select the $$ k $$ closest fruits.

- Assign the most common fruit type among these neighbors to the new fruit\[6\].

---

**Summary**

k-NN is an intuitive, instance-based algorithm that predicts the class or value of new data points by considering the proximity of existing labeled data. Its effectiveness depends on the choice of $$ k $$, distance metric, and feature scaling, and it is best suited for problems where the notion of similarity (distance) is meaningful\[1\]\[2\]\[4\].

Sources
\[1\] K-Nearest Neighbor (KNN) Explained - Pinecone <https://www.pinecone.io/learn/k-nearest-neighbor/>
\[2\] k-nearest neighbors algorithm - Wikipedia <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>
\[3\] StatQuest: K-nearest neighbors, Clearly Explained - YouTube <https://www.youtube.com/watch?v=HVXime0nQeI>
\[4\] What is k-Nearest Neighbor (kNN)? - Elastic <https://www.elastic.co/what-is/knn>
\[5\] What Is a K-Nearest Neighbor Algorithm? | Built In <https://builtin.com/machine-learning/nearest-neighbor-algorithm>
\[6\] What is the K-Nearest Neighbor (KNN) Algorithm? - YouTube <https://www.youtube.com/watch?v=b6uHw7QW_n4>