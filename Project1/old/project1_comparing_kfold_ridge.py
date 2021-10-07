import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# For nice plots
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(2018)

# Make data set.
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(-0.1, 0.1, (N,N))


# Generate the data.
N = 200
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

xy = np.zeros((N,2))
xy[:,0] = x
xy[:,1] = y

x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

## Cross-validation on Ridge regression using KFold only

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 10)

# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)

# Initialize a KFold instance, number of splitted parts
k = 5 # try 5 - 10 folders
#Provides train/test indices to split data in train/test sets.
#Split dataset into k consecutive folds (without shuffling by default).
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((nlambdas, k))

i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)
    j = 0
    for train_inds, test_inds in kfold.split(xy):
        xytrain = xy[train_inds]
        ztrain = z[train_inds]

        xytest = xy[test_inds]
        ztest = z[test_inds]

        # Makes the design matrix for train data
        Xtrain = poly.fit_transform(xytrain)
        ridge.fit(Xtrain, ztrain)

        # Makes the design matrix for test data
        Xtest = poly.fit_transform(xytest)
        ypred = ridge.predict(Xtest)

        # Scaling the data
        scaler = StandardScaler()
        scaler.fit(Xtrain)
        scaler.fit(Xtest)

        Xtrain = scaler.transform(Xtrain) #subtracts mean and divide over standard diviation
        Xtest = scaler.transform(Xtest)

        # MSE
        scores_KFold[i,j] = np.sum((ypred - ztest)**2)/np.size(ypred)

        j += 1
    i += 1


estimated_mse_KFold = np.mean(scores_KFold, axis = 1)

## Cross-validation using cross_val_score from sklearn along with KFold

estimated_mse_sklearn = np.zeros(nlambdas)
i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb) #, fit_intercept = False)

    X = poly.fit_transform(xy)
    estimated_mse_folds = cross_val_score(ridge, X, z, scoring='neg_mean_squared_error', cv=kfold)
    # cross_val_score return an array containing the estimated negative mse for every fold.
    # we have to the the mean of every array in order to get an estimate of the mse of the model
    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

    i += 1

## Plot and compare the slightly different ways to perform cross-validation

plt.figure()
plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'k-folding, sklearn')
plt.plot(np.log10(lambdas), estimated_mse_KFold, 'r--', label = 'k-folding, own code')
plt.xlabel('log10(lambda)')
plt.ylabel('mse')
plt.legend()
# plt.savefig('plots/exercise3_comparing_k-fold_Ridge.pdf')
plt.show()
