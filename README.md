# Applied Data Analysis and Machine Learning, Project 1
## Regression analysis and resampling methods

Kamilla Ida Julie Sulebakk, Marcus Berget and Sebastian Amundsen

In this project we have utilized the OLS, Lasso and Ridge regression for fitting polynomials to a specific two-dimensional function called Franke’s function with  
stochastic noise of normal distribution. Furthermore, we have employed the resamling techniques cross-validation and bootstrap in order to perform a proper assessment of our models. 


## The project is created with:
* Python version: 3
  * sklearn
  * numpy	
  * matplotlib
  * imageio
  * mpl_toolkits
  
* LaTeX

## How to run the code:
Open terminal window, these commands compile and execute the programs: 
```
# To compile and execute exercise 1-5 at once:
python3 main.py

# To compile and execute exercise 6: 
python3 terrain.py

```
Here; 
* python3 main.py fits the Franke’s function and calculates:
  * confidence intervals of the parameters beta for OLS without resampling. 
  * the MSE and R2-score when using OLS, Ridge and Lasso regression without resampling.
  * the MSE, bias and variance for all the regression methods with Bootstrap as resampling technique. 
  * the MSE obtained with our own algorithm as well as that from Scikit-Learn for all the regression methods with cross-validation. 
    * with N = 20, N_bootstraps = 20, k = 10, N_lambdas = 20, polynomials up to order 10. 

* python3 terrain.py fits real data of Norway and calculates:
  * the MSE and R2-score when using OLS, Ridge and Lasso regression without resampling.
  * the MSE, bias and variance for all the regression methods with Bootstrap as resampling technique. 
  * the MSE obtained with our own algorithm as well as that from Scikit-Learn for all the regression methods with cross-validation. 
    * with N = 500, N_bootstraps = 100, k = 10, N_lambdas = 20, polynomials up to order 10. 
    
    
### Plotting for excercise 1 - 5)
#### Produces all results provided in pdf 
```
python3 main.py
```

### Plotting for excercise 6)
#### Produces all results provided in pdf 

```
python3 terrain.py
```
