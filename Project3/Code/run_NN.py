import numpy as np
import LFPy
from lfpykit.eegmegcalc import NYHeadModel
from lfpykit import CellGeometry, CurrentDipoleMoment
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from NN_methods import NeuralNetwork
from plot import bias_accuracy
from common import prepare_data

# Prepare some data
# N_samples = 5000
# eeg, pos_list = prepare_data(N_samples)
eeg = np.load(f'data/eeg_100.npy')
pos_list = np.load(f'data/pos_list_100.npy')
pos_list = pos_list.T

# Define initial parameters
inputsize = eeg.shape[1]
N_layers = 5
N_neurons = 100
N_epochs = 30 #100
batch_size = 30
eta = 0.0001
lmbd = 1e-6
N_folds = 5 #10

# Scale the data and append noice
scaler = StandardScaler()
scaler.fit(eeg)
eeg = scaler.transform(eeg)
eeg += np.random.normal(0,0.1, eeg.shape)
X_train, X_test, y_train, y_test = train_test_split(eeg, pos_list, test_size=0.2)


model = NeuralNetwork(X_train, X_test, y_train, y_test)
# act_func = 'relu'
# pred, loss, val_loss, accuracy, val_accuracy = model.simple(
#                                                 inputsize, N_layers, N_neurons,
#                                                 N_epocs, batch_size, eta, lmbd,
#                                                 act_func
#                                                 )
#
# print("Predictions without k-folding")
# print(pred[0:5])
# print(y_test[0:5])


# Merge inputs and targets
act_funcs = ['relu', 'sigmoid', 'tanh']
for i in range(len(act_funcs)):
    pred, target_data, loss, val_loss, accuracy, val_accuracy = model.kfold(
                                                    inputsize, N_layers, N_neurons,
                                                    N_epochs, N_folds, batch_size,
                                                    eta, lmbd, act_funcs[i]
                                                    )

    loss = np.mean(loss, axis=0)
    val_loss = np.mean(val_loss, axis=0)
    accuracy = np.mean(accuracy, axis=0)
    accuracy = np.mean(val_accuracy, axis=0)

    bias_accuracy(loss, val_loss, accuracy, val_accuracy, N_epochs, act_funcs[i])

    print(f"Predictions with k-folding using {act_funcs[i]}:")
    print("Pediction:")
    print(pred[-1][0:5][:])
    print("Target:")
    print(target_data[-1][0:5][:])




