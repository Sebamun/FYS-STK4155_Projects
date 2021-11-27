import numpy as np
from tensorflow.keras.models import Sequential      # This allows appending layers to existing models
from tensorflow.keras.layers import Dense           # This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             # This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           # This allows using whichever regularizer we want (l1,l2,l1_l2)
from sklearn.model_selection import train_test_split as splitter
from sklearn.preprocessing import StandardScaler

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
    return model

eeg = np.load(f'data/eeg_100.npy')              # (1000, 231)
pos_list = np.load(f'data/pos_list_100.npy')    # (3, 1000)
pos_list = pos_list.T

inputsize = eeg.shape[1]
n_layers = 5
n_neuron = 100
eta = 0.0001
lamda = 1e-6

X_train, X_test, y_train, y_test = splitter(eeg, pos_list, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

DNN_model = NN_model(inputsize, n_layers, n_neuron, eta, lamda)
DNN_model.fit(X_train, y_train, epochs=30, batch_size=30, verbose=2)
print('training complete')
scores = DNN_model.evaluate(X_test, y_test)
print(scores)

pred = DNN_model.predict(X_test[0:5])
print(pred)
print(y_test[0:5])
