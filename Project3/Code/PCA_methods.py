import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import time
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential      # This allows appending layers to existing models
from tensorflow.keras.layers import Dense           # This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             # This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers
from plot import plot_test_train_loss

class NN_PCA:
    def __init__(self, eeg, pos_list, N_layers, N_neurons, batch_size, eta, lmbd):
         self.eeg = eeg
         self.pos_list = pos_list
         self.N_layers = N_layers
         self.N_neurons = N_neurons
         self.batch_size = batch_size
         self.eta = eta
         self.lmbd = lmbd

    def append_expl_var(self, error_std):
        input = self.eeg
        input += np.random.normal(0, error_std, self.eeg.shape)
        pca = PCA(n_components=10)
        pca.fit(input)
        expl_var = pca.explained_variance_ratio_
        return expl_var

    def run_kfold_PCA(self, act_func, n_comps, N_epochs, N_folds):
        pca = PCA(n_components=n_comps)
        pca.fit(self.eeg)
        T = pca.transform(self.eeg)
        X_train, X_test, y_train, y_test = train_test_split(T, self.pos_list, test_size=0.2)
        start_time = time.time()
        pred, target_data, loss, val_loss, accuracy, val_accuracy = self.kfold(
                                                    T.shape[1],N_epochs, N_folds, act_func[0],
                                                    X_train, X_test, y_train, y_test
                                                    )

        stop_time = time.time() - start_time
        loss = np.mean(loss, axis=0)
        val_loss = np.mean(val_loss, axis=0)
        accuracy = np.mean(accuracy, axis=0)
        val_accuracy = np.mean(val_accuracy, axis=0)

        print(f'Loss = {loss[-1]}')
        print(f'val_loss = {val_loss[-1]}')
        print(f'accuracy = {accuracy[-1]}')
        print(f'val_accuracy = {val_accuracy[-1]}')

        # plot_test_train_loss(loss, val_loss, accuracy, val_accuracy, N_epochs, act_func, lmbd, stop_time, n_comps, N_samples)

        print(f"Predictions for PCA with k-folding using {act_func}:")
        print("Pediction:")
        print(pred[-1][0:5][:])
        print("Target:")
        print(target_data[-1][0:5][:])
        return loss, val_loss, stop_time

    def run_kfold_reduced(self, N_epochs, act_func):
        start_time = time.time()
        N_folds = 5
        X_train, X_test, y_train, y_test = train_test_split(self.eeg, self.pos_list, test_size=0.2)
        pred, target_data, loss, val_loss, accuracy, val_accuracy = self.kfold(
                                                    self.eeg.shape[1], N_epochs, N_folds, act_func[0],
                                                    X_train, X_test, y_train, y_test
                                                    )
        stop_time = time.time() - start_time
        loss = np.mean(loss, axis=0)
        val_loss = np.mean(val_loss, axis=0)
        accuracy = np.mean(accuracy, axis=0)
        val_accuracy = np.mean(val_accuracy, axis=0)

        print(f'Loss = {loss[-1]}')
        print(f'val_loss = {val_loss[-1]}')
        print(f'accuracy = {accuracy[-1]}')
        print(f'val_accuracy = {val_accuracy[-1]}')

        # plot_test_train_loss(loss, val_loss, accuracy, val_accuracy, N_epochs, act_func, lmbd, stop_time, n_comps, N_samples)

        print(f"Predictions for PCA with k-folding using {act_func}:")
        print("Pediction:")
        print(pred[-1][0:5][:])
        print("Target:")
        print(target_data[-1][0:5][:])
        return loss, val_loss, stop_time

    def kfold(self, inputsize, N_epochs, N_folds, act_func, X_train, X_test, y_train, y_test):
        inputs = np.concatenate((X_train, X_test), axis=0)
        targets = np.concatenate((y_train, y_test), axis=0)

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
                Dense(self.N_neurons,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(self.lmbd),
                    input_dim=inputsize
                )
            )
            #Hidden layers
            for _ in range(self.N_layers - 1):       # add hidden layers to the network
                model.add(Dense(self.N_neurons, activation=act_func, kernel_regularizer=regularizers.l2(self.lmbd)))
            #Output layer
            model.add(Dense(3, activation=None))

            sgd = optimizers.SGD(learning_rate=self.eta, momentum=0.9)
            model.compile(optimizer=sgd, loss='mse', metrics = ['accuracy'])

            # Fit data to model
            fit_ = model.fit(
                    inputs[train], targets[train], self.batch_size, N_epochs,
                    verbose=0, validation_data=(inputs[test], targets[test])
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
        return pred, target_data, np.array(loss), np.array(val_loss), np.array(accuracy), np.array(val_accuracy)
