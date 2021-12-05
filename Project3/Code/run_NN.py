import numpy as np
import LFPy
from lfpykit.eegmegcalc import NYHeadModel
from lfpykit import CellGeometry, CurrentDipoleMoment
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from NN_methods import NeuralNetwork
from plot import plot_bias_accuracy
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

#Run simple DNN
def run_simple_DNN(act_func):
    pred, loss, val_loss, accuracy, val_accuracy = model.simple(
                                                    inputsize, N_layers, N_neurons,
                                                    N_epocs, batch_size, eta, lmbd,
                                                    act_func
                                                    )

    print("Predictions without k-folding")
    print(pred[0:5])
    print(y_test[0:5])

# Run DNN with k-folding with different activation functions
def run_kfold_DNN(act_funcs):
    # act_funcs = ['relu', 'sigmoid', 'tanh']
    loss = np.zeros((3, N_epochs))
    val_loss = np.zeros_like(loss)
    val_loss = np.zeros_like(loss)
    accuracy = np.zeros_like(loss)

    for i in range(len(act_funcs)):
        pred, target_data, loss, val_loss, accuracy, val_accuracy = model.kfold(
                                                        inputsize, N_layers, N_neurons,
                                                        N_epochs, N_folds, batch_size,
                                                        eta, lmbd, act_funcs[i]
                                                        )

        loss[i,:] = np.mean(loss, axis=0)
        val_loss[i, :] = np.mean(val_loss, axis=0)
        accuracy[i, :] = np.mean(accuracy, axis=0)
        val_accuracy[i, :] = np.mean(val_accuracy, axis=0)

        print(f"Predictions with k-folding using {act_funcs[i]}:")
        print("Pediction:")
        print(pred[-1][0:5][:])
        print("Target:")
        print(target_data[-1][0:5][:])

    plot_bias_accuracy(loss, val_loss, accuracy, val_accuracy, N_epochs, act_funcs)

# Run PCA with k-folding
def run_kfold_PCA(act_func):
    pca = PCA(n_components=20)
    pca.fit(eeg)
    xx = pca.transform(eeg)
    X_train, X_test, y_train, y_test = train_test_split(xx, pos_list, test_size=0.2)
    model = NeuralNetwork(X_train, X_test, y_train, y_test)
    pred, target_data, loss, val_loss, accuracy, val_accuracy = model.kfold(
                                                xx.shape[1], N_layers, N_neurons,
                                                N_epochs, N_folds, batch_size,
                                                eta, lmbd, act_func
                                                )

    loss = np.mean(loss, axis=0)
    val_loss = np.mean(val_loss, axis=0)
    accuracy = np.mean(accuracy, axis=0)
    accuracy = np.mean(val_accuracy, axis=0)

    print(f"Predictions for PCA with k-folding using {act_func}:")
    print("Pediction:")
    print(pred[-1][0:5][:])
    print("Target:")
    print(target_data[-1][0:5][:])



# run_simple_DNN('relu')
run_kfold_DNN(['relu', 'sigmoid', 'tanh'])
# run_kfold_PCA('relu')