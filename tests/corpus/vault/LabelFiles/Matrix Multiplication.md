# Matrix Multiplication

Here's an example of matrix multiplication with two matrices (A) and (B), and their product (C):

$$
$$

- Matrix (A):
   \[
   \\begin{pmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6
   \\end{pmatrix}
   \]

- Matrix (B):
   \[
   \\begin{pmatrix}
   7 & 8 \\
   9 & 10 \\
   11 & 12
   \\end{pmatrix}
   \]

- Product (C = A \\times B):
   \[
   \\begin{pmatrix}
   58 & 64 \\
   139 & 154
   \\end{pmatrix}
   \]

The product matrix (C) is calculated by taking the dot product of rows in (A) with columns in (B). For example, the top-left element of (C) (58) is calculated by multiplying the elements of the first row of (A) with the corresponding elements of the first column of (B) and adding them together: ((1 \\times 7) + (2 \\times 9) + (3 \\times 11) = 58).

```python
import numpy as np

# Define two matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

# Multiply the matrices
C = np.dot(A, B)

A, B, C
```

```python
(array([[1, 2, 3],
        [4, 5, 6]]),
 array([[ 7,  8],
        [ 9, 10],
        [11, 12]]),
 array([[ 58,  64],
        [139, 154]]))
```