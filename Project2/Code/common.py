import autograd.numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def MSE(y_data, y_model):
    """Calculate MSE"""
    #sum = np.where(sum > 1, 1, sum)
    return np.mean((y_data - y_model)**2)

def FrankeFunction(x, y):
    N = x.shape[0]
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0,0.1,(N,N))

def create_X(x, y, n):
    """Create design Matrix"""
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X

def prepare_cancer_data():
    cancer = load_breast_cancer()
    # Feature matrix of 569 rows (samples) and 30 columns (parameters)
    X = cancer.data
    # Label array of 569 rows (0 for benign and 1 for malignant)
    Y = cancer.target.reshape(-1, 1)

    # Generate training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=1)

    # scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def scale(X):
    """Scale all columns except the first (which is all ones)"""
    scaler = StandardScaler()
    scaler.fit(X[:, 1:])
    X[:, 1:] = scaler.transform(X[:, 1:])
    return X

def learning_schedule(t, t0, t1):
    """Learning rate used in SGD"""
    return t0/(t+t1)

def accuracy(pred, target):
    return np.mean(abs(pred - target) < 0.5)

