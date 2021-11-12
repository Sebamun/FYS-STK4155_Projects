import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from sklearn.model_selection import train_test_split
import time
from methods_K import Sigmoid, Tang_hyp, RELU, ELU, Leaky, Heaviside
from common_K import FrankeFunction, initialize
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

start = time.time()
np.random.seed(1235)

#define the initial conditions for generating data
cancer = load_breast_cancer()


inputs = cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
outputs = cancer.target                  #Label array of 569 rows (0 for benign and 1 for malignant)
labels = cancer.feature_names[0:30]

X = inputs
Y = outputs

print('The content of the breast cancer dataset is:')      #Print information about the datasets
print(labels)
print('-------------------------')
print("inputs =  " + str(inputs.shape))
print("outputs =  " + str(outputs.shape))
print("labels =  "+ str(labels.shape))

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

print("Shape of design matrix = " + str(np.shape(X)))

Y = outputs.reshape(-1, 1)

#Generate training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

epochs = 100000
eta = 0.001
lmbd = 0.00001
gamma = 0.9
batch_size = 50
n_layers = 2
n_hidden_neurons = 20
n_features = X.shape[1]

sigmoid_model = Sigmoid(eta, lmbd, gamma, n_layers, n_hidden_neurons, n_features, 'classification')
sigmoid_model.train(X_train_scaled, y_train, epochs, batch_size)

y_h, a_h, y_o_sigmoid, a_L = sigmoid_model.feed_forward(X_test_scaled)
time_sigmoid = time.time()

# compare z_o_sigmoid and z_test
print(y_o_sigmoid.shape)
print(y_test.shape)

accuracy = np.mean(abs(a_L - y_test) < 0.5)
print(f'Accuracy = {accuracy}, Sigmoid, own code')

# Now with Sklearn
clf = MLPClassifier(hidden_layer_sizes=(1),random_state=1,\
 activation='logistic', batch_size=10, learning_rate = 'constant',\
  learning_rate_init = 0.001, max_iter = epochs,shuffle=True,\
    tol=0.000000001, momentum=0.9).fit(X_train, y_train.ravel())

accuracy_sklearn = clf.score(X_test, y_test)
print(f'Accuracy = {accuracy_sklearn}, sigmoid, sklearn')

end_time = time.time() - start
print(f'It took {(end_time):.1f} seconds.')


#Visualisation of dataset (for correlation analysis)
# plt.figure()
# plt.scatter(X[:,0], X[:,2], s=40, c=Y, cmap=plt.cm.Spectral)
# plt.xlabel('Mean radius', fontweight='bold')
# plt.ylabel('Mean perimeter', fontweight='bold')
# plt.show()
#
# plt.figure()
# plt.scatter(X[:,5], X[:,6], s=40, c=Y, cmap=plt.cm.Spectral)
# plt.xlabel('Mean compactness', fontweight='bold')
# plt.ylabel('Mean concavity', fontweight='bold')
# plt.show()
#
#
# plt.figure()
# plt.scatter(X[:,0], X[:,1], s=40, c=Y, cmap=plt.cm.Spectral)
# plt.xlabel('Mean radius', fontweight='bold')
# plt.ylabel('Mean texture', fontweight='bold')
# plt.show()
#
# plt.figure()
# plt.scatter(X[:,2], X[:,1], s=40, c=Y, cmap=plt.cm.Spectral)
# plt.xlabel('Mean perimeter', fontweight='bold')
# plt.ylabel('Mean compactness', fontweight='bold')
# plt.show()

#Select features relevant to classification (texture,perimeter,compactness and symmetery)
#and add to input matrix
# temp1 = np.reshape(X[:,1], (len(X[:,1]),1))
# temp2 = np.reshape(X[:,2], (len(X[:,2]),1))
# temp3 = np.reshape(X[:,3], (len(X[:,3]),1))
# temp4 = np.reshape(X[:,4], (len(X[:,4]),1))
# temp5 = np.reshape(X[:,5], (len(X[:,5]),1))
# temp6 = np.reshape(X[:,6], (len(X[:,6]),1))
# temp7 = np.reshape(X[:,7], (len(X[:,7]),1))
# temp8 = np.reshape(X[:,8], (len(X[:,8]),1))
# X = np.hstack((temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8))
