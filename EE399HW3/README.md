# EE399HW3

##  Akash Shetty, HW3

## Introduction and Overview
In this homework we are analyzing the MNIST dataset of images of digits and are working on different ways of classifying these images, we will be using Numpy, Sklearn, Linear Discriminant Analysis, Descision Tree Classifers, Support Vector Machines and dicuss the differences of each one.

## Theoreretical Backround
### Singular Value Decomposition:  
Singular Value Decomposition is known as the factorization of a matrix into its left singular matrix, its singular values and its right singular matrix it can be represented as the following:  

$X = U \Sigma V^T$  

Where $X$ is an mxn real matrix. The $U$ represents an mxm matrix. $\Sigma$ is an mxn matrix with diagonal singular values. $V^\top$ is an nxn matrix. SVD has a wide range of uses in machine learning, it can reduce the dimensionality of the data while retaining the important information. It extracts the important features that are in in images.

### Linear Discriminant Analysis
Linear Discriminant Analysis is a technique that is used in classifcation which works by reducing the dimensionality of the data. It is reduced by finding the eigenvectors of the product of  

$S_w^{-1}S_b w = \lambda w$  

where the eigenvalues and eigenvectors give the quantity of interest and the projection bias.
### Support Vector Machines:  
Support Vector Machines are algorithms that we are using in this homework for classification. They work by creating a hyperplane in a higher dimension that is able to seperate the data which can be shown by the figure below. 
<img width="439" alt="image" src="https://user-images.githubusercontent.com/107958888/234407139-ebbbd7c9-f585-4c88-a4b7-6db9e60620b7.png">

### Descision Tree Classifiers:  
A decision tree classifier is a machine learning tool that optimally splits data using a hierachal basis in order to classify data sets. An example of a decision tree classifier can be show below.
<img width="713" alt="image" src="https://user-images.githubusercontent.com/107958888/234408796-c359e6fe-61f9-458d-a74f-cba24dc03723.png">

## Algorithm Implementation and Computaional Results
#### Do an SVD analysis of the digit images. You will need to reshape each image into a column vector and each column of your data matrix is a different image:
```python 
mnist = fetch_openml('mnist_784')
X = mnist.data / 255.0  # Scale the data to [0, 1]
y = mnist.target
U, S, Vt = np.linalg.svd(X.T, full_matrices = False)
```
#### What does the singular value spectrum look like and how many modes are necessary for good image reconstruction? (i.e. what is the rank r of the digit space?)
```python 
plt.plot(S)
plt.title("Singular Value Spectrum")
plt.xlabel("rank")
plt.ylabel("singular value")
rank = 50
```
<img width="499" alt="image" src="https://user-images.githubusercontent.com/107958888/234401127-01f77d97-ead8-4bdc-ad87-dff507ef965d.png">
We can see from the above figure that most of the non-zero points are which is around rank 50 which means the rank of the data would be around 50.  
&nbsp;  

&nbsp;

#### What is the interpretation of the U, Σ, and V matrices?  
$U$ : The left singular vectors that represent the projections on to the image for the principal components of the data  
$\Sigma$ : The singular values that can be used to compute the variance of the principal components  
$V$ : The right singular vectors that represent the principal components    
&nbsp;

#### On a 3D plot, project onto three selected V-modes (columns) colored by their digit label. For example, columns 2,3, and 5.
For this we are projecting the original data $X$ onto the vector columns of $V$ specfically of the principal components 2 3 and 5
```python,
pca = PCA(n_components = rank)
X_pca = pca.fit_transform(X)
print(X_pca.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 1], X_pca[:, 2], X_pca[:, 4], c=mnist.target.astype(int), s=20)
```
<img width="346" alt="image" src="https://user-images.githubusercontent.com/107958888/234401203-8af6d648-2398-4c8e-b09c-9ea7a1d0f359.png">

#### Pick two digits. See if you can build a linear classifier (LDA) that can reasonable identify/classify them.
```python
pca = PCA(n_components = rank)
X_pca = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

# 2 digit classifier 
d1 = '1'
d2 = '8'

X_train_2d = X_train[(y_train == d1) | (y_train == d2)]
y_train_2d = y_train[(y_train == d1) | (y_train == d2)]

X_test_2d = X_test[(y_test == d1) | (y_test == d2)]
y_test_2d = y_test[(y_test == d1) | (y_test == d2)]

lda = LDA()
lda.fit(X_train_2d, y_train_2d)

y_pred_2d_tr = lda.predict(X_train_2d)
score_2d_tr = accuracy_score(y_train_2d, y_pred_2d_tr)
print("Accuracy of TRAIN classification between digits 1 and 8: {:.2f}".format(score_2d_tr))

y_pred = lda.predict(X_test_2d)
score = accuracy_score(y_test_2d, y_pred)

print("Accuracy of TEST classification between digits 1 and 8: {:.2f}".format(score))
```
<img width="374" alt="image" src="https://user-images.githubusercontent.com/107958888/234401277-126b6446-23c2-4fd3-8d7a-a336d9472cf4.png">

#### Pick three digits. Try to build a linear classifier to identify these three now.

```python
# 3 digit linear classifier
d1 = '1'
d2 = '4'
d3 = '5'
X_train_3d = X_train[(y_train == d1) | (y_train == d2) | (y_train == d3)]
y_train_3d = y_train[(y_train == d1) | (y_train == d2) | (y_train == d3)]

X_test_3d = X_test[(y_test == d1) | (y_test == d2) | (y_test == d3)]
y_test_3d = y_test[(y_test == d1) | (y_test == d2) | (y_test == d3)]

lda = LDA()
lda.fit(X_train_3d, y_train_3d)

y_pred_3d_tr = lda.predict(X_train_3d)
score_3d_tr = accuracy_score(y_train_3d, y_pred_3d_tr)
print("Accuracy of TRAIN classification between digits 1 4 and 5: {:.2f}".format(score))

y_pred = lda.predict(X_test_3d)
score = accuracy_score(y_test_3d, y_pred)
print("Accuracy of TEST classification between digits 1 4 and 5: {:.2f}".format(score))
```
<img width="387" alt="image" src="https://user-images.githubusercontent.com/107958888/234401316-20ba832d-3521-4467-976f-4747aa595a16.png">

#### Which two digits in the data set appear to be the most difficult to separate? Quantify the accuracy of the separation with LDA on the test data. Which two digits in the data set are most easy to separate? Quantify the accuracy of the separation with LDA on the test data.

```python
pairs = []
for i in range(10):
    for j in range(10):
        if (i,j) not in pairs and (j,i) not in pairs and i!=j:
            pairs.append((i,j))
            
def two_digit_seperator(x, y, d1, d2):
    X = x[( y == d1) | (y == d2)]
    y_2 = y[(y == d1) | (y == d2)]
    return X, y_2
```
```python
accuracy_2d = []
for pair in pairs:
    X_train_2d, y_train_2d = two_digit_seperator(X_train, y_train, str(pair[0]), str(pair[1]))
 
    X_test_2d, y_test_2d = two_digit_seperator(X_test, y_test, str(pair[0]), str(pair[1]))
    lda = LDA()
    lda.fit(X_train_2d, y_train_2d)
    y_pred = lda.predict(X_test_2d)
    score = accuracy_score(y_test_2d, y_pred)
    accuracy_2d.append(score)
print(f"The highest is {pairs[np.argmax(accuracy_2d)]} with and a test accuracy of {np.max(accuracy_2d)}")
print(f"The lowest is {pairs[np.argmin(accuracy_2d)]} with and test accuracy of {np.min(accuracy_2d)}")
```
<img width="388" alt="image" src="https://user-images.githubusercontent.com/107958888/234401379-2fe8a693-aa44-4b58-a98f-75063a82703d.png">

#### SVM (support vector machines) and decision tree classifiers were the state-of-the-art until about 2014. How well do these separate between all ten digits?

```python
# using SVM to seperate all ten digits
clf = SVC()
clf.fit(X_train, y_train)

y_pred_SVC_t = clf.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_SVC_t)

print(f"Training Accuracy for SVM: {accuracy:.2f}")

y_pred_SVC = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_SVC)

print(f"Testing Accuracy for SVM: {accuracy:.2f}")
```

<img width="224" alt="image" src="https://user-images.githubusercontent.com/107958888/234401444-3f4f728b-b9dd-4d17-8620-5d02c45ef2b8.png">

```python
# using DescionTreeClassifier to seperarate all ten digits
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state = 42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred_DTC_t = clf.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_DTC_t)

print(f"Training Accuracy for DecisionTreeClassifier: {accuracy:.2f}")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Testing Accuracy for DecisionTreeClassifier: {accuracy:.2f}")
```

<img width="338" alt="image" src="https://user-images.githubusercontent.com/107958888/234401523-0c9e93f1-7430-44c3-b3d6-058472426485.png">

#### Compare the performance between LDA, SVM and decision trees on the hardest and easiest pair of digits to separate (from above).

```python 
clf = SVC()
clf.fit(X_train_SVC, y_train_SVC)

y_pred = clf.predict(X_train_SVC)
accuracy_easiest = accuracy_score(y_train_SVC, y_pred)
print(f"Training Accuracy for easiest pair SVC {accuracy_easiest}")
y_pred = clf.predict(X_test_SVC)
accuracy_easiest = accuracy_score(y_test_SVC, y_pred)
print(f"Test Accuracy for easiest pair SVC {accuracy_easiest}")
#hardest
X_train_SVC, y_train_SVC = two_digit_seperator(X_train, y_train, ld1, ld2)
X_test_SVC, y_test_SVC = two_digit_seperator(X_test, y_test, ld1, ld2)
clf = SVC()
clf.fit(X_train_SVC, y_train_SVC)

y_pred = clf.predict(X_train_SVC)
accuracy_hardest = accuracy_score(y_train_SVC, y_pred)
print(f"Training Accuracy for hardest pair SVC {accuracy_hardest}")

y_pred = clf.predict(X_test_SVC)
accuracy_hardest = accuracy_score(y_test_SVC, y_pred)
print(f"Test Accuracy for hardest pair SVC {accuracy_hardest}")
```

<img width="379" alt="image" src="https://user-images.githubusercontent.com/107958888/234401559-daec613e-8584-410f-8e86-16bb51d99d45.png">

```python 
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train_DTC, y_train_DTC)

y_pred = dtc.predict(X_train_DTC)
accuracy_hardest = accuracy_score(y_train_DTC, y_pred)
print(f"Train Accuracy for easiest pair Descision Tree Classifier {accuracy_easiest}")

y_pred = dtc.predict(X_test_DTC)
accuracy_hardest = accuracy_score(y_test_DTC, y_pred)
print(f"Test Accuracy for easiest pair Descision Tree Classifier {accuracy_easiest}")

# hardest
X_train_DTC, y_train_DTC = two_digit_seperator(X_train, y_train, ld1, ld2)
X_test_DTC, y_test_DTC = two_digit_seperator(X_test, y_test, ld1, ld2)

dtc.fit(X_train_DTC, y_train_DTC)

y_pred = dtc.predict(X_train_DTC)
accuracy_easiest = accuracy_score(y_train_DTC, y_pred)
print(f"Train Accuracy for hardest pair Descision Tree Classifier {accuracy_hardest}")

y_pred = dtc.predict(X_test_DTC)
accuracy_easiest = accuracy_score(y_test_DTC, y_pred)
print(f"Test Accuracy for hardest pair Descision Tree Classifier {accuracy_hardest}")
```

<img width="501" alt="image" src="https://user-images.githubusercontent.com/107958888/234401592-5d9b30fe-b2a5-4259-af00-a97911806ea4.png">
&nbsp;

From the data above we can see that descicion tree classifiers are the worst at classifying all 10 digits but however when seperating two digits both the easiest and the hardest it does better and equal to the LDA and the SVM classifiers. We can also see the problem of overfitting on some of the data which we would need to fix in the future. In terms of classifying all 10 SVC seems to be the best perhaps because of the way the pixels are because there are clear points in the pixels that correspond to a certain number like how 1 is always a straight line or 0 is always a circle with nothing in the middle. 
## Summary 

In conclusion I like the way that we are able to explore these classification techniques and open up to a new genre geared towards dimensionality fo data and dealing with that. 
