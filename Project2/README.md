# Applied Data Analysis and Machine Learning, Project 2
## Classification and Regression, from linear and logistic regression to neural networks

Kamilla Ida Julie Sulebakk, Marcus Berget and Sebastian Amundsen

In this project we have studied both classification and regression problems by developing our own feed-forward neural network (FFNN) and logistic regression code. For the regression problem we have analyzed terrain data produced by Franke's function. The data set utilized for the classification problem is the Wisconsin Breast Cancer data of images representing various features and tumors.

## The project is created with:
* Python version: 3
  * sklearn
  * numpy
  * matplotlib
  * autograd
  * mpl_toolkits

* LaTeX

## How to run the code:
Open terminal window, these commands compile and execute the programs:
```
# To compile and execute exercise a):
python3 main_GD.py

# To compile and execute exercise b) and c):
python3 main_NN.py

# To compile and execute exercise d):
python3 main_cancer_NN.py

# To compile and execute exercise e):
python3 main_cancer_LR.py

```
Here;
* python3 main_GD.py fits the terrain data using gradient descent and calculates:
  * Provide plots given the Ridge and OLS regression methods with both ordinary and momentum SGD.
  * Provide plot over MSE given by lambda parameter in Ridge.
    * with batchsize M=10, gamma = 0.9, lambda = 4.28*10^(-2), 1000 number of epochs and learning schedule function.

* python3 main_NN.py fits terrain data using FFNN and calculates:

* python3 main_cancer_NN.py fits the cancer data using FFNN and:  
  * calculates and prints the accuracy and cost of the fit
  * provide plots of the accuracy as function of epochs
    * with 2 hidden layers, 20 hidden neurons, batch size of 50 samples, lambda = 0.001 and the learning schedule function for the value of the step length.

* python3 main_cancer_NN.py fits the cancer data using Logistic Regression and:  
  * calculates and prints the accuracy and cost of the fit
  * provide plots of the accuracy as function of epochs
    * with 2 hidden layers, 20 hidden neurons, batch size of 50 samples, lambda = 0.05 and the learning schedule function for the value of the step length.

If one wantes to change the initial contitions, this is done in the different main files.

Due to long run times we recommend to comment out the functions you don't want to use :)

