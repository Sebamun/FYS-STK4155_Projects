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
import pandas as pd
import matplotlib.pyplot as plt

def NN_model(inputsize, n_layers, n_neuron, eta, lamda, act_func):
    model = Sequential()
    #Input Layer
    model.add(
        Dense(n_neuron,
            activation='relu',
            kernel_regularizer=regularizers.l2(lamda),
            input_dim=inputsize
        )
    #Hidden layers
    )
    for _ in range(n_layers - 1):       # add hidden layers to the network
        model.add(Dense(n_neuron, activation=act_func, kernel_regularizer=regularizers.l2(lamda)))
    #Output layer
    model.add(Dense(3, activation=None))

    sgd = optimizers.SGD(learning_rate=eta, momentum=0.9)
    model.compile(optimizer=sgd, loss='mse', metrics = ['accuracy'])
    return model

# eeg = np.load(f'data/eeg_100.npy')              # (1000, 231)
# pos_list = np.load(f'data/pos_list_100.npy')    # (3, 1000)

def prepare_data(num_positions):

    nyhead = NYHeadModel()
    step_length = nyhead.cortex.shape[1]//num_positions #+ 1   # step_length som vi skal hente ut posisjoner fra
    dipole_locations = nyhead.cortex[:, ::step_length]      # Der svulsten kan plasseres
    num_positions = dipole_locations.shape[1]

    pos_list = np.zeros((3, num_samples))   # Posisjonene til svulstene
    for i in range(num_samples): # 0 til 1000
        idx = np.random.randint(0, num_positions) # Tilfeldig valgt posisjon blant X
        pos_list[:, i] = dipole_locations[:, idx] # Legger til posisjonen til svulsten

    eeg = np.zeros((num_samples, 231))
    #print(pd.DataFrame(eeg))
    #quit()

    for i in range(num_samples):
        nyhead.set_dipole_pos(pos_list[:,i]) #Lager en instans tilhørende posisjon pos_list[:,i]
        M = nyhead.get_transformation_matrix() #Henter ut transformasjonsmatrisen tilhørende posisjon pos_list[:,i]
        p = np.array(([0.0], [0.0], [1.0]))
        #Roterer retningen til dipolmomentet slik at det står normalt på hjernebarken,
        #i forhold til posisjonen pos_list[:, i]
        p = nyhead.rotate_dipole_to_surface_normal(p)
        eeg_i = M @ p #Genererer EEG-signalet tilhørende et dipolmoment i posisjon pos_list[:,i]
        eeg[i, :] = eeg_i.T
    return eeg, pos_list

def bias_accuracy(act_funcs):

    eeg, pos_list = prepare_data(num_positions)
    #print(eeg.shape)
    #print(pos_list.shape)
    pos_list = pos_list.T

    inputsize = eeg.shape[1]
    n_layers = 5
    n_neuron = 100
    eta = 0.0001 # 0.0001
    lamda = 1e-6

    scaler = StandardScaler()
    scaler.fit(eeg)
    eeg = scaler.transform(eeg)
    eeg += np.random.normal(0,0.1, eeg.shape)
    X_train, X_test, y_train, y_test = splitter(eeg, pos_list, test_size=0.2)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title('Training loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('bias')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_title('Validation loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('bias')
    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.set_title('Training accuracy')
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('accuracy')
    fig4 = plt.figure()
    ax4 = fig4.add_subplot()
    ax4.set_title('Validation accuracy')
    ax4.set_xlabel('epoch')
    ax4.set_ylabel('accuracy')
    for i in range(len(act_funcs)):
        DNN_model = NN_model(inputsize, n_layers, n_neuron, eta, lamda, act_funcs[i])
        history = DNN_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=30, verbose=0) # epoch = 50, verbose=2

        _, train_acc = DNN_model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = DNN_model.evaluate(X_test, y_test, verbose=0)

        # plot loss during training
        ax1.plot(history.history['loss'], label=f'Train for {act_funcs[i]}')
        ax1.legend()
        # Plot validation loss
        ax2.plot(history.history['val_loss'], label=f'Test for {act_funcs[i]}')
        ax2.legend()
        # plot accuracy during training
        ax3.plot(history.history['accuracy'], label=f'Train for {act_funcs[i]}')
        ax3.legend()
        # plot validation accuracy
        ax4.plot(history.history['val_accuracy'], label=f'Test for {act_funcs[i]}')
        ax4.legend()
    plt.show()

num_sensors = 231
num_samples = 5000
num_positions = 74382

act_funcs = ['relu', 'sigmoid', 'tanh']

bias_accuracy(act_funcs)
