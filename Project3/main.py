import numpy as np
from tensorflow.keras.models import Sequential      # This allows appending layers to existing models
from tensorflow.keras.layers import Dense           # This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             # This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           # This allows using whichever regularizer we want (l1,l2,l1_l2)
from sklearn.model_selection import train_test_split as splitter
from sklearn.preprocessing import StandardScaler
import LFPy
from lfpykit.eegmegcalc import NYHeadModel
from lfpykit import CellGeometry, CurrentDipoleMoment

np.random.seed(1234)

def NN_model(inputsize, n_layers, n_neuron, eta, lamda):
    model = Sequential()
    model.add(
        Dense(n_neuron,
            activation='relu',
            kernel_regularizer=regularizers.l2(lamda),
            input_dim=inputsize
        )
    )
    for _ in range(n_layers - 1):       # add hidden layers to the network
        model.add(Dense(n_neuron, activation='relu', kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(3, activation=None))

    sgd = optimizers.SGD(learning_rate=eta, momentum=0.9)
    model.compile(optimizer=sgd, loss='mse')
    # model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

eeg = np.load(f'data/eeg_100.npy')              # (1000, 231)
pos_list = np.load(f'data/pos_list_100.npy')    # (3, 1000)

num_sensors = 231
num_samples = 5000

def prepare_data(num_samples):

    nyhead = NYHeadModel()
    dipole_locations = nyhead.cortex  # Der svulsten kan plasseres
    num_positions = dipole_locations.shape[1]

    pos_list = np.zeros((3, num_samples))   # Posisjonene til svulstene
    for i in range(num_samples): # 0 til 1000
        idx = np.random.randint(0, num_signals) # Tilfeldig valgt posisjon blant X
        pos_list[:, i] = dipole_locations[:, idx] # Legger til posisjonen til svulsten

    eeg = np.zeros((num_samples, 231))

    for i in range(num_samples):
        nyhead.set_dipole_pos(pos_list[:,i])
        M = nyhead.get_transformation_matrix()
        p = np.array(([0.0], [0.0], [1.0]))
        p = nyhead.rotate_dipole_to_surface_normal(p)
        eeg_i = M @ p
        # eeg_i += np.random.normal(0,0.1, eeg_i.shape)
        eeg[i, :] = eeg_i.T
    return eeg, pos_list

eeg, pos_list = prepare_data(num_samples)
print(eeg.shape)
print(pos_list.shape)

pos_list = pos_list.T

inputsize = eeg.shape[1]
n_layers = 5
n_neuron = 100
eta = 0.0001
lamda = 1e-6

scaler = StandardScaler()
scaler.fit(eeg)
eeg = scaler.transform(eeg)
eeg += np.random.normal(0,0.1, eeg.shape) # adding noice
X_train, X_test, y_train, y_test = splitter(eeg, pos_list, test_size=0.2)


DNN_model = NN_model(inputsize, n_layers, n_neuron, eta, lamda)
DNN_model.fit(X_train, y_train, epochs=50, batch_size=30, verbose=2)
print('training complete')
scores = DNN_model.evaluate(X_test, y_test)
print(scores)

pred = DNN_model.predict(X_test[0:5])
print(pred)
print(y_test[0:5])


from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(eeg)
xx = pca.transform(eeg)
X_train, X_test, y_train, y_test = splitter(xx, pos_list, test_size=0.2)
print(X_train.shape[1])
print(X_test.shape[1])
PCA_model = NN_model(xx.shape[1], n_layers, n_neuron, eta, lamda)
PCA_model.fit(X_train, y_train, epochs=50, batch_size=30, verbose=0)
print('training complete using PCA')

pred = PCA_model.predict(X_test[0:5])
print(pred)
print(y_test[0:5])
