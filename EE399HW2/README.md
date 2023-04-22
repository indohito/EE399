# EE399HW2

## Akash Shetty, HW2
This is the second homework of Intro to Machine Learning in this we go over Correlation Matrices and utilizing Singular Value Decomposition

## Introduction and Overview
For this homework are given a data file of images and are asked to create a Correlation Matrix for the images and also Utilize SVD to extract the important features that are available in the data. The following is what instructions and data we are given to start our implementation. 

<pre>
This file has a total of 39 different faces with about 65 lighting scenes for each face (2414 faces in all).
The individual images are columns of the matrix X, where each image has been downsampled to 32 × 32
pixels and converted into gray scale with values between 0 and 1. So the matrix is size 1024 × 2414. To
import the file, use the following:
</pre>
```python
import numpy as np
from scipy.io import loadmat
results=loadmat(’yalefaces.mat’)
X=results[’X’]
```

## Theoretical Backround
#### Correlation Matrices:
Correlation Matrices are the mathmatical representation of the relationship between a set of variables
for this problem we are using a Gram matrix which can be represented by the following relationship: $X^\top X$

#### Singular Value Decompostion:
Singular Value Decomposition is known as the factorization of a matrix into its left singular matrix, its singular values and its right singular matrix it can be represented as the following:  
$X = USV^\top$  
Where $X$ is an mxn real matrix. The $U$ represents an mxm matrix. $S$ is an mxn matrix with diagonal singular values. $V^\top$ is an nxn matrix. SVD has a wide range of uses in machine learning, it can reduce the dimensionality of the data while retaining the important information. It extracts the important features that are in in images.


## Algorithm Implementation and Development

a) Compute a 100 × 100 correlation matrix $C$ where you will compute the dot product (correlation)
between the first 100 images in the matrix $X$. Thus each element is given by $c_{jk} = x^\top_{j} x_{k}$
where $x_{j}$ is the jth column of the matrix. Plot the correlation matrix using pcolor.

```python
C = np.matmul(X[:, :100].T, X[:, :100])
# We create a C_max and C_min and fill the diagonals so that the correlation does not return the same pictures
C_max = C.copy()
C_min = C.copy()
np.fill_diagonal(C_max, 0)
# for C_min it could return the same image if the images is mostly blank or completely black
np.fill_diagonal(C_min, 250)
```

(b) From the correlation matrix for part (a), which two images are most highly correlated? Which are
most uncorrelated? Plot these faces.
```python
# This function gives the index of the correlation matrix which corresponds to the images in the X matrix 
max_ind = np.unravel_index(np.argmax(C_max), C.shape)
min_ind = np.unravel_index(np.argmin(C_min), C.shape)
```
(c) Repeat part (a) but now compute the 10 × 10 correlation matrix between images and plot the
correlation matrix between them.
```python
[1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
```
(Just for clarification, the first image is labeled as one, not zero like python might do)
```python
X_10 = np.array([1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005])
# -1 because python starts at 0
X_10 = X_10 - 1
# Same as in A
C = np.matmul(X[:, X_10].T, X[:, X_10])
```
(d) Create the matrix $Y = X X^\top$ and find the first six eigenvectors with the largest magnitude eigenvalue.
```python

Y = np.matmul(X, X.T)
eigvalues, eigenvectors = np.linalg.eig(Y)
# sort the eig values by descending order
sort_eig = np.argsort(eigvalues)[::-1]
# get the first six largest eigvalues eigenvectors 
largest_eigvec = eigenvectors[:, sort_eig[:6]]

```
(e) SVD the matrix $X$ and find the first six principal component directions.
```python
u,s,v = np.linalg.svd(X)
# v_t gives the principal directions of the matrix while u are the principal component of the matrix matching up to the eigenvectors
six_principal_directions = v[:, :6]

```
(f) Compare the first eigenvector $v_{1}$ from (d) with the first SVD mode $u_{1}$ from (e) and compute the
norm of difference of their absolute values.
```python
norm_svd_and_eig = np.linalg.norm(np.abs(largest_eigvec[:,0])- np.abs(u[:,0]))
print(norm_svd_and_eig)
#4.465152617342574e-16
```
a smaller norm means that the vectors are closer to each other so they are very similar vectors

(g) Compute the percentage of variance captured by each of the first 6 SVD modes. Plot the first 6
SVD modes
```python
sum_squared = np.sum(s**2)

variances = [s[i]**2/sum_squared for i in range(6)]
print(variances)
#[72.92756746909562, 15.281762655694365, 2.56674494298527, 1.877524851471473, 0.6393058444446512, 0.5924314415034935]
```
This is computed using this equation for variance:
$R^2 = \frac{s_{i}^{2}}{\sum_{j}s_{j}^{2}}$
## Computational Results

<img width="400" alt="image" src="https://user-images.githubusercontent.com/107958888/232594667-f853264c-3800-422c-9dac-d1ac5d648cf6.png">
Here is the correlation matrix of the first 100 images that are given in our data. 

<img width="400" alt="image" src="https://user-images.githubusercontent.com/107958888/232594724-12c84601-f942-4556-84fc-0b48d8207714.png">
If we look closer at our Correlation Matrix we can find the most and least correlated images that are given to us  
Highest Correlated: Image 88 and Image 86  
Lowest Correlated: Image 54 and Image 64
<img width="400" alt="image" src="https://user-images.githubusercontent.com/107958888/232609035-aa626a29-e468-4479-899f-5e2c6904a722.png">
Here is a the correlation matrix of the following images that we want to compare

```python
[1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/107958888/232678599-610869b6-2f9f-4e03-ad12-983fcec9ba75.png">
These are the first 6 SVD modes and below is the variances of these 6 SVD modes:  

```python
[72.92756746909562, 15.281762655694365, 2.56674494298527, 1.877524851471473, 0.6393058444446512, 0.5924314415034935]
```
These variances tell us about the variances of the original data that is captured in the modes that are displayed, A higher variance that is displayed from the original data means that we were able to capture more of the variance from the original data making it more useful for data analysis. 

## Summary and Conclusions
Correlation Matrices are very useful when finding the relationship between the variables and how close they are based on the different pixels that are in this image classificaiton. Singular Value Decomposition is also very useful when classifying images to reduce the dimensionality and extract certain features from the original data. 
