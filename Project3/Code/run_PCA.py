import numpy as np
import LFPy
from lfpykit.eegmegcalc import NYHeadModel
from lfpykit import CellGeometry, CurrentDipoleMoment
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from PCA_methods import NN_PCA
from plot import plot_bias_accuracy
from common import prepare_data
import matplotlib.pyplot as plt
from plot import plot_test_train_loss, plot_expl_var, plot_test_loss, plot_test_train_reduced
import time

# Generate data
N_samples = 1000
eeg, pos_list = prepare_data(N_samples)
# eeg = np.load(f'data/eeg_100.npy')
# pos_list = np.load(f'data/pos_list_100.npy')
pos_list = pos_list.T

# Define initial parameters
inputsize = eeg.shape[1]
N_layers = 5
N_neurons = 200
N_epochs = 100
batch_size = 40
eta = 0.0001
lmbd = 1e-5
N_folds = 5 #10

# Scale the data
scaler = StandardScaler()
scaler.fit(eeg)
eeg = scaler.transform(eeg)


#Do a PCA on data with varying noise and plot
model1 = NN_PCA(eeg, pos_list, N_layers, N_neurons, batch_size, eta, lmbd)
error_std_list = [0.1, 1, 5]
expl_var_list = np.array([0,0,0], dtype=object)

for i, err in enumerate(error_std_list):
    expl_var_list[i] = model1.append_expl_var(err)

plot_expl_var(expl_var_list, error_std_list)

#Add noise
error_std = 0.1
eeg += np.random.normal(0, error_std, eeg.shape)
model2 = NN_PCA(eeg, pos_list, N_layers, N_neurons, batch_size, eta, lmbd)
loss, val_loss, stop_time = model2.run_kfold_PCA(['tanh'], 10, N_epochs, N_folds)

#Train the NN on n_comps principle components and plot results
n_comps_list = [7, 8, 9, 10]
empty = np.zeros(len(n_comps_list))
val_loss = np.array(empty, dtype=object)
loss = np.array(empty, dtype=object)
stop_time_list = np.array(empty)
for i, n_comps in enumerate(n_comps_list):
    loss[i], val_loss[i], stop_time_list[i] = model2.run_kfold_PCA(['tanh'], n_comps, N_epochs, N_folds)
plot_test_loss(val_loss, N_epochs, n_comps_list, error_std)
plot_test_train_loss(loss, val_loss, N_epochs, ['tanh'], lmbd, n_comps_list, N_samples, error_std)


#Train the model where 10 features are extracted from data, picked out Uniformly and then the first 10
eeg_uni = eeg[:,0:-1:23]
eeg_first_ten = eeg[:,0:10]
model3 = NN_PCA(eeg_uni, pos_list, N_layers, N_neurons, batch_size, eta, lmbd)
model4 = NN_PCA(eeg_first_ten, pos_list, N_layers, N_neurons, batch_size, eta, lmbd)
inputs = np.array([0,0], dtype=object)
val_loss = np.array(empty, dtype=object)
loss = np.array(empty, dtype=object)
stop_time_list = np.array(empty)

loss[0], val_loss[0], stop_time_list[0] = model3.run_kfold_reduced(N_epochs, ['tanh'])
loss[1], val_loss[1], stop_time_list[1] = model4.run_kfold_reduced(N_epochs, ['tanh'])
plot_test_train_reduced(loss, val_loss, N_epochs, ['tanh'], lmbd, N_samples, error_std)
