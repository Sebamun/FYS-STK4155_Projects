import numpy as np
from sklearn.model_selection import train_test_split as splitter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

np.random.seed(1234)

def convolutional_NN_model(inputsize, eta, lamda):
    model = Sequential()
    model.add(Conv1D(filters=100, kernel_size=2, activation='relu', input_shape=inputsize))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dropout(0.5))
    # model.add(Dense(n_outputs, activation='relu', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_outputs, activation='relu'))
    model.add(Dense(n_outputs, activation=None))
    model.summary()
    sgd = optimizers.SGD(learning_rate=eta, momentum=0.9)
    model.compile(optimizer='adam', loss='mse')
    return model

# load data
eeg = np.load(f'data/eeg_100.npy')
pos_list = np.load(f'data/pos_list_100.npy')
pos_list = pos_list.T

scaler = StandardScaler()
scaler.fit(eeg)
scaler.transform(eeg)

eeg = eeg.reshape(eeg.shape[0], eeg.shape[1], 1)

# split into train and test data
X_train, X_test, y_train, y_test = splitter(eeg, pos_list, test_size=0.2)

# n_layers = 5
eta = 0.00001
lmbd = 1e-6

inputsize = (X_train.shape[1],X_train.shape[2])
n_outputs = (y_train.shape[1])

CNN_model = convolutional_NN_model(inputsize, eta, lmbd)

# fit network
verbose, epochs, batch_size = 2, 30, 30
CNN_model.fit(X_train, y_train, epochs=30, batch_size=30, verbose=2)
print('training complete')

scores = CNN_model.evaluate(X_train, y_train)
print(f'scores = {scores}')

pred = CNN_model.predict(X_test)
MSE = mean_squared_error(y_test, pred)
print(f'MSE = {MSE}')

# Printing some of the predictions to check if they matches
pred_ = CNN_model.predict(X_test[0:5])
print(pred_)
print(y_test[0:5])
