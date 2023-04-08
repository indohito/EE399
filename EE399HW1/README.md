# EE399HW1

## Akash Shetty, HW1, EE399

This is a paper to go over the concepts of curve fitting and error modeling
in it we will be utilizing Python and Python libraries such as 

## Introduction and Overview
In this hw we are asked to answer the following questions over error evalution
and data fitting. The following questions are asked of us based on given data points

These are the following data points and equations

<pre>
X=np.arange(0,31)
Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
</pre>


$f(x) = A \cos(Bx) + Cx + D$

## Theoretical Backround

To minimize the error that our data fitting returns we will be using the Least Squares error function displayed below 

$E = \sqrt{\frac{1}{n}\sum_{j=1}^{n} (f(x_j) - y_j)^2}$

using this equation we can find the minimun error and find the parameters to the function that is given above

## Algorithm Implementation and Development
In  this HW we are asked to answer questions based on data fitting and minimizing error below I will go over the code development below 

(i) Write a code to find the minimum error and determine the parameters A, B, C, D

code for LSE equation
<pre>
def velfit(c, x, y):
    e2 = np.sqrt(np.sum((c[0]*np.cos(c[1]*x)+c[2]*x+c[3]-y)**2))
    return e2
</pre>

code to minimize the error and get optimal parameters
<pre>
res= opt.minimize(velfit, v0 , args=(X, Y), method='Nelder-Mead')
</pre>




(ii) With the results of (i), fix two of the parameters and sweep through values of the
other two parameters to generate a 2D loss (error) landscape. Do all combinations of
two fixed parameters and two swept parameters. You can use something like pcolor to
visualize the results in a grid. How many minima can you find as you sweep through
parameters?

fixing two of the parameters and sweeping through the other ones (in this case we sweep through C and D)

<pre>

A = c[0]
B = c[1]
C_range = np.linspace(c[2]-10, c[2]+10, 100)
D_range = np.linspace(c[3]-10, c[3]+10, 100)


error_values = np.zeros((len(C_range), len(D_range)))
for i in range(len(C_range)):
    for j in range(len(D_range)):
        C = C_range[i]
        D = D_range[j]
        error_values[i, j] = velfit([A, B, C, D], X, Y)
       
</pre>

We then are able to visualize the results of the sweeping using p color

(iii) Using the first 20 data points as training data, fit a line, parabola and 19th degree
polynomial to the data. Compute the least-square error for each of these over the training
points. Then compute the least square error of these models on the test data which are
the remaining 10 data points.

<pre>
X_train = X[:20]
Y_train = Y[:20]

X_test = X[20:]
Y_test = Y[20:]

a, b = np.polyfit(X_train, Y_train, 1)
linear_fit_train = a*X_train + b
linear_fit_test = a*X_test + b

X_train_error_1 = np.sqrt(np.sum((linear_fit_train-Y_train)**2))
X_test_error_1 = np.sqrt(np.sum((linear_fit_test-Y_test)**2))

a_2, b_2, c_2 = np.polyfit(X_train, Y_train, 2)
parabolic_fit_train = a_2*X_train**2 +b_2*X_train + c_2
parabolic_fit_test = a_2*X_test**2 + b_2*X_test + c_2

X_train_error_2 = np.sqrt(np.sum((linear_fit_train-Y_train)**2))
X_test_error_2 = np.sqrt(np.sum((linear_fit_test-Y_test)**2))


poly_19 = np.polyfit(X_train, Y_train, 19)
poly_19_train = np.polyval(poly_19, X_train)
poly_19_test = np.polyval(poly_19, X_test)

X_train_error_19 = np.sqrt(np.sum((poly_19_train-Y_train)**2))
X_test_error_19 = np.sqrt(np.sum((poly_19_test-Y_test)**2))
</pre>

(iv) Repeat (iii) but use the first 10 and last 10 data points as training data. Then fit the
model to the test data (which are the 10 held out middle data points). Compare these
results to (iii).

<pre>
X_train = np.concatenate([X[:10], X[-10:]])
Y_train = np.concatenate([Y[:10], Y[-10:]])

X_test = X[10:20]
Y_test = Y[10:20]

a, b = np.polyfit(X_train, Y_train, 1)
linear_fit_train = a*X_train + b
linear_fit_test = a*X_test + b

X_train_error_1 = np.sqrt(np.sum((linear_fit_train-Y_train)**2))
X_test_error_1 = np.sqrt(np.sum((linear_fit_test-Y_test)**2))

a_2, b_2, c_2 = np.polyfit(X_train, Y_train, 2)
parabolic_fit_train = a_2*X_train**2 +b_2*X_train + c_2
parabolic_fit_test = a_2*X_test**2 + b_2*X_test + c_2

X_train_error_2 = np.sqrt(np.sum((linear_fit_train-Y_train)**2))
X_test_error_2 = np.sqrt(np.sum((linear_fit_test-Y_test)**2))

poly_19 = np.polyfit(X_train, Y_train, 19)
poly_19_train = np.polyval(poly_19, X_train)
poly_19_test = np.polyval(poly_19, X_test)

X_train_error_19 = np.sqrt(np.sum((poly_19_train-Y_train)**2))
X_test_error_19 = np.sqrt(np.sum((poly_19_test-Y_test)**2))

</pre>

## Computational Results
from the code and algorithms above we are able to get plots from p color and from fitting the data in parts iii and iv these are shown below

(ii)

<img width="460" alt="image" src="https://user-images.githubusercontent.com/107958888/230698362-e98ea72a-33d7-4c7d-8334-9927f02329b6.png">

This image shows the relationship between error values of C and D the part of the graph which is the darkest is the most optimal parameters for these functions which show the many minima

(iii)

<img width="460" alt="image" src="https://user-images.githubusercontent.com/107958888/230698415-d3c39948-8354-4154-a6df-828c1395c8a5.png">
<img width="460" alt="image" src="https://user-images.githubusercontent.com/107958888/230698423-82ad1255-23f9-4811-b5dc-fc8c1d59e27f.png">

These images show when the first 20 points are used as the training data the first graph being linear and quadratic and the second graph being the function being fit with the 19th degree polynomial, you can see that there is huge error for the 19th degree polynomial and the quadratic function becuase the 19th degree polynomial only will properly fit the 20 points as training data and the quadratic function falls off after the training data. 

(iv)

<img width="460" alt="image" src="https://user-images.githubusercontent.com/107958888/230698660-07dd7bd8-6077-4198-8c02-f6980c04aebe.png">
<img width="464" alt="image" src="https://user-images.githubusercontent.com/107958888/230698681-63e9816b-0fde-44f4-85ff-f258e6e9c4a8.png">

These images show when the first and last 10 data points are instead used as training data with the first graph being linear and quadratic fits and the second graph being the function fit with the 19th degree polynomial. As you can see from the graphs the linear and quadratic fits are a lot better then the previous training data because it is able to fit on both ends of the data so it takes into account more of the data points in the middle if the function was linear. The second graph however shows again that a 19th degree polynomial cant be fit to the test data properly because of the huge error that it has when not being fit to the training data.

## Summary and Conclusions

In conlusion I believe that this was an effective excercise in learning about error values and fitting data

