import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
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
N = 100
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

xy = np.zeros((N,2))
xy[:,0] = x
xy[:,1] = y

x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

poly_degrees = np.arange(1, 7)


# Initialize a KFold instance, number of splitted parts
k = 5 # try 5 - 10 folders
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((2, len(poly_degrees), k))

for idx, degree in enumerate(poly_degrees):
    poly = PolynomialFeatures(degree)
    OLS = LinearRegression() #fit_intercept = False
    j = 0
    for train_inds, test_inds in kfold.split(xy):
        xytrain = xy[train_inds]
        ztrain = z[train_inds]

        xytest = xy[test_inds]
        ztest = z[test_inds]

        # Makes the design matrix for train data
        Xtrain = poly.fit_transform(xytrain)
        # OLS.fit(Xtrain, ztrain)
        beta = np.linalg.inv(Xtrain.T@Xtrain)@Xtrain.T@ztrain

        # Makes the design matrix for test data
        Xtest = poly.fit_transform(xytest)
        ytilde = Xtrain@beta #OLS.predict(Xtrain)
        ypred = Xtest@beta #OLS.predict(Xtest)

        # # Scaling the data
        # scaler = StandardScaler()
        # scaler.fit(Xtrain)
        # scaler.fit(Xtest)
        #
        # Xtrain = scaler.transform(Xtrain) #subtracts mean and divide over standard diviation
        # Xtest = scaler.transform(Xtest)

        # MSE
        scores_KFold[0, idx,j] = np.sum((ytilde - ztrain)**2)/np.size(ytilde)
        scores_KFold[1, idx,j] = np.sum((ypred - ztest)**2)/np.size(ypred)

        j += 1


estimated_mse_KFold = np.mean(scores_KFold, axis = 2)


# Cross-validation using cross_val_score from sklearn along with KFold
estimated_mse_sklearn = np.zeros(len(poly_degrees))
for idx, degree in enumerate(poly_degrees):
    poly = PolynomialFeatures(degree)
    OLS = LinearRegression() #fit_intercept = False

    X = poly.fit_transform(xy)
    estimated_mse_folds = cross_val_score(OLS, X, z, scoring='neg_mean_squared_error', cv=kfold)
    # cross_val_score return an array containing the estimated negative mse for every fold.
    # we have to the the mean of every array in order to get an estimate of the mse of the model
    estimated_mse_sklearn[idx] = np.mean(-estimated_mse_folds)


## Plot and compare the slightly different ways to perform cross-validation

plt.figure()
plt.plot(poly_degrees, estimated_mse_sklearn, label = 'k-folding, sklearn')
plt.plot(poly_degrees, estimated_mse_KFold[1], 'r--', label = 'k-folding, own code')
plt.xlabel('polydegrees')
plt.ylabel('mse')
plt.legend()
plt.show()


plt.figure()
plt.plot(poly_degrees, estimated_mse_KFold[0], label = 'train')
plt.plot(poly_degrees, estimated_mse_KFold[1], 'r--', label = 'test')
plt.xlabel('polydegrees')
plt.ylabel('mse')
plt.legend()
plt.show()