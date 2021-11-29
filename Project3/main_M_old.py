# import numpy as np
# import LFPy
# from lfpykit.eegmegcalc import NYHeadModel
# from lfpykit import CellGeometry, CurrentDipoleMoment
# from sklearn.preprocessing import StandardScaler
# from common import FrankeFunction
#
# np.random.seed(1234)
#
# # def prepare_data():
# #     nyhead = NYHeadModel()
# #     dipole_locations = nyhead.cortex[:, 0:74381:1000]
# #     samples = dipole_locations.shape[1]
# #     eeg = np.zeros((samples, 231))
# #
# #     for i in range(samples):
# #         nyhead.set_dipole_pos(dipole_locations[:,i])
# #         M = nyhead.get_transformation_matrix()
# #         p = np.array(([0.0], [0.0], [1.0]))
# #         p = nyhead.rotate_dipole_to_surface_normal(p)
# #         eeg_i = M @ p
# #         eeg[i, :] = eeg_i.T
# #     return eeg, dipole_locations
#
# def prepare_data():
#     nyhead = NYHeadModel()
#     dipole_locations = nyhead.cortex[:, 0:74381:30000] # 8 positions
#     samples = 10
#     pos_list = np.zeros((3, samples))
#     for i in range(samples):
#         idx = np.random.randint(0,dipole_locations.shape[1])
#         pos_list[:, i] = dipole_locations[:, idx]
#         #pos_list
#
#     eeg = np.zeros((samples, 231))
#
#     for i in range(samples):
#         nyhead.set_dipole_pos(pos_list[:,i])
#         M = nyhead.get_transformation_matrix()
#         p = np.array(([0.0], [0.0], [1.0]))
#         p = nyhead.rotate_dipole_to_surface_normal(p)
#         eeg_i = M @ p
#         eeg[i, :] = eeg_i.T
#     return eeg, pos_list
#
#
# eeg, pos_list = prepare_data()
# # Find which position has the best match
# # eeg_last = eeg[-1,:]
# # eeg_last = np.reshape(eeg_last, (1,231))
# # pred = [-4.916407, 23.108973, -0.55642927]
# # dipole_locations = dipole_locations.T
# # diff = dipole_locations - pred
# # diff = np.abs(diff)
# # diff = np.mean(diff, axis = 1)
# # argmin = np.argmin(diff)
# # print(argmin)
# # print(dipole_locations[argmin,:])
# # print(pred)
# # print(dipole_locations[-4,:])
# # quit()
#
#
#
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
# from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
# from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
# from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
# from sklearn.model_selection import train_test_split as splitter
#
# def NN_model(inputsize, n_layers, n_neuron, eta, lamda):
#     model=Sequential()
#     for i in range(n_layers):       #Run loop to add hidden layers to the model
#         if (i==0):                  #First layer requires input dimensions
#             model.add(Dense(n_neuron,activation='tanh',kernel_regularizer=regularizers.l2(lamda),input_dim=inputsize))
#         else:                       #Subsequent layers are capable of automatic shape inferencing
#             model.add(Dense(n_neuron,activation='tanh',kernel_regularizer=regularizers.l2(lamda)))
#     model.add(Dense(3))
#     sgd=optimizers.SGD(learning_rate=eta)#, momentum=0.9)
#     model.compile(optimizer='sgd', loss='mse')
#     #Tror feilen ligger i kostfunksjonen, det kan virke som at det optimaliserer i forhold til hele gjennomsnittet
#     #Eller ikke, aner ikke hva som er feilen.
#     return model
#
# pos_list = pos_list.T
#
# # raw_data = eeg.T @ pos_list
# print(eeg[0:4, 0:4])
# print(pos_list[0:4, :])
# # split_data = eeg.T @ pos_list
# # print(raw_data - split_data)
# input()
# X_train,X_test,y_train,y_test = splitter(eeg, pos_list , test_size=0.1)
# # split_data = X_train.T @ y_train
# print(X_train[0:4, 0:4])
# print(y_train[0:4, :])
# input()
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# # print(eeg.shape[1])
# # quit()
# # print(np.std(X_train, axis = 0))
# quit()
#
# DNN_model = NN_model(eeg.shape[1], 3, 20, 0.0001, 15)
# DNN_model.fit(X_train,y_train,epochs=100,batch_size=30,verbose=2)
# print('training complete')
# scores = DNN_model.evaluate(X_test, y_test)
# print(scores)
#
# eeg_last = eeg[-4,:]
# eeg_last = np.reshape(eeg_last, (1,231))
# pred = DNN_model.predict(eeg_last)
# print(pred)
# # diff = dipole_locations - pred
# # argmin = np.argmin(diff, axis = 1)
# print(pos_list[-4,:])


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from sklearn.model_selection import train_test_split as splitter

eeg = np.load(f'data/eeg_100.npy')              # (1000, 231)
pos_list = np.load(f'data/pos_list_100.npy')    # (3, 1000)
pos_list = pos_list.T

def NN_model(inputsize, n_layers, n_neuron, eta, lamda):
    model=Sequential()
    for i in range(n_layers):       #Run loop to add hidden layers to the model
        if (i==0):                  #First layer requires input dimensions
            model.add(Dense(n_neuron,activation='tanh',kernel_regularizer=regularizers.l2(lamda),input_dim=inputsize))
        else:                       #Subsequent layers are capable of automatic shape inferencing
            model.add(Dense(n_neuron,activation='tanh',kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(3))
    sgd=optimizers.SGD(learning_rate=eta)#, momentum=0.9)
    model.compile(optimizer='sgd', loss='mse')
    #Tror feilen ligger i kostfunksjonen, det kan virke som at det optimaliserer i forhold til hele gjennomsnittet
    #Eller ikke, aner ikke hva som er feilen.
    return model

pos_list = pos_list.T

# raw_data = eeg.T @ pos_list
print(eeg[0:4, 0:4])
print(pos_list[0:4, :])
# split_data = eeg.T @ pos_list
# print(raw_data - split_data)
X_train,X_test,y_train,y_test = splitter(eeg, pos_list , test_size=0.1)
# split_data = X_train.T @ y_train
print(X_train[0:4, 0:4])
print(y_train[0:4, :])
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# print(eeg.shape[1])
# quit()
# print(np.std(X_train, axis = 0))

DNN_model = NN_model(eeg.shape[1], 3, 20, 0.0001, 15)
DNN_model.fit(X_train,y_train,epochs=100,batch_size=30,verbose=2)
print('training complete')
scores = DNN_model.evaluate(X_test, y_test)
print(scores)

eeg_last = eeg[-4,:]
eeg_last = np.reshape(eeg_last, (1,231))
pred = DNN_model.predict(eeg_last)
print(pred)
# diff = dipole_locations - pred
# argmin = np.argmin(diff, axis = 1)
print(pos_list[-4,:])
