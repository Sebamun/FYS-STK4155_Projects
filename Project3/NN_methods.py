import numpy as np
from tensorflow.keras.models import Sequential      # This allows appending layers to existing models
from tensorflow.keras.layers import Dense           # This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             # This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           # This allows using whichever regularizer we want (l1,l2,l1_l2)
from sklearn.model_selection import KFold

class NeuralNetwork:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def simple(self, inputsize, N_layers, N_neurons, N_epochs, batch_size, eta, lmbd, act_func):
        model = Sequential()
        #Input Layer
        model.add(
            Dense(N_neurons,
                activation='relu',
                kernel_regularizer=regularizers.l2(lmbd),
                input_dim=inputsize
            )
        )
        #Hidden layers
        for _ in range(N_layers - 1):       # add hidden layers to the network
            model.add(Dense(N_neurons, activation=act_func, kernel_regularizer=regularizers.l2(lmbd)))
        #Output layer
        model.add(Dense(3, activation=None))

        sgd = optimizers.SGD(learning_rate=eta, momentum=0.9)
        model.compile(optimizer=sgd, loss='mse', metrics = ['accuracy'])

        # Fit data to model
        fit_ = model.fit(
                self.X_train, self.y_train, N_epochs,
                batch_size, verbose=0,
                validation_data=(self.X_test, self.y_test)
                )
        print('Training completed')

        pred = model.predict(self.X_test)
        loss = fit_.history['loss'] # train loss
        val_loss = fit_.history['val_loss'] # test loss
        accuracy = fit_.history['accuracy'] # train accuracy
        val_accuracy = fit_.history['val_accuracy'] # test accuracy

        return pred, loss, val_loss, accuracy, val_accuracy


    def kfold(self, inputsize, N_layers, N_neurons, N_epochs, N_folds, batch_size, eta, lmbd, act_func):
        inputs = np.concatenate((self.X_train, self.X_test), axis=0)
        targets = np.concatenate((self.y_train, self.y_test), axis=0)

        kfold = KFold(n_splits=N_folds, shuffle=True)
        pred = []
        target_data = []
        loss = []
        val_loss = []
        accuracy = []
        val_accuracy = []

        fold_no = 1
        for train, test in kfold.split(inputs, targets):
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            model = Sequential()
            #Input Layer
            model.add(
                Dense(N_neurons,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(lmbd),
                    input_dim=inputsize
                )
            )
            #Hidden layers
            for _ in range(N_layers - 1):       # add hidden layers to the network
                model.add(Dense(N_neurons, activation=act_func, kernel_regularizer=regularizers.l2(lmbd)))
            #Output layer
            model.add(Dense(3, activation=None))

            sgd = optimizers.SGD(learning_rate=eta, momentum=0.9)
            model.compile(optimizer=sgd, loss='mse', metrics = ['accuracy'])

            # Fit data to model
            fit_ = model.fit(
                    inputs[train], targets[train], N_epochs,
                    batch_size, verbose=0,
                    validation_data=(inputs[test], targets[test])
                    )

            pred.append(model.predict(inputs[test]))
            target_data.append(targets[test])

            loss.append(fit_.history['loss']) # train loss
            val_loss.append(fit_.history['val_loss']) # test loss
            accuracy.append(fit_.history['accuracy']) # train accuracy
            val_accuracy.append(fit_.history['val_accuracy']) # test accuracy

            # Increase fold number
            fold_no = fold_no + 1
        print('Training completed')
        return pred, target_data, loss, val_loss, accuracy, val_accuracy

