import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common import R2
from NN_methods import NeuralNetwork
from plot import plot_bias_accuracy, plot_bias_accuracy_simple, plot_R2, plot_bias_variance_tradeoff
import load_eeg_data

N_samples = 1000
eeg, pos_list = load_eeg_data.load_data(N_samples)

# Define initial parameters
inputsize = eeg.shape[1]
N_layers = 5
N_neurons = 100
N_epochs = 100 #100
batch_size = 30
eta = 0.0001
lmbd = 1e-6
N_folds = 10

# Scale the data and append noice
scaler = StandardScaler()
scaler.fit(eeg)
eeg = scaler.transform(eeg)
eeg += np.random.normal(0, 0.1, eeg.shape) # marcus la til 0.5 for å få best mulig resultater.
X_train, X_test, y_train, y_test = train_test_split(eeg, pos_list, test_size=0.2)

model = NeuralNetwork(X_train, X_test, y_train, y_test)

#Run simple DNN
def run_simple_DNN(act_funcs):
    loss_array = np.zeros((len(act_funcs), N_epochs), dtype=object)
    val_loss_array = np.zeros_like(loss_array)
    accuracy_array = np.zeros_like(loss_array)
    val_accuracy_array = np.zeros_like(loss_array)

    for i in range(len(act_funcs)):
        pred, loss, val_loss, accuracy, val_accuracy = model.simple(
                                                    inputsize, N_layers, N_neurons,
                                                    N_epochs, batch_size,
                                                    eta, lmbd, act_funcs[i]
                                                    )
        loss_array[i, :] = loss
        val_loss_array[i, :] = val_loss
        accuracy_array[i, :] = accuracy
        val_accuracy_array[i, :] = val_accuracy

        print("Predictions without k-folding using {act_funcs[i]}:")
        print(pred[0:5])
        print(y_test[0:5])

    plot_bias_accuracy_simple(loss_array, val_loss_array, accuracy_array, val_accuracy_array, N_epochs, act_funcs)

# Run DNN with k-folding with different activation functions
def run_kfold_DNN(act_funcs):
    loss_array = np.zeros((len(act_funcs), N_epochs))
    val_loss_array = np.zeros_like(loss_array)
    accuracy_array = np.zeros_like(loss_array)
    val_accuracy_array = np.zeros_like(loss_array)
    R2_score_array = np.zeros_like(loss_array)

    for i in range(len(act_funcs)):
        pred, target_data, loss, val_loss, accuracy, val_accuracy, R2_score = model.kfold(
                                                        inputsize, N_layers, N_neurons,
                                                        N_epochs, N_folds, batch_size,
                                                        eta, lmbd, act_funcs[i]
                                                        )
        loss_array[i,:] = np.mean(loss, axis=0)
        val_loss_array[i, :] = np.mean(val_loss, axis=0)
        accuracy_array[i, :] = np.mean(accuracy, axis=0)
        val_accuracy_array[i, :] = np.mean(val_accuracy, axis=0)
        R2_score_array[i, :] = np.mean(R2_score, axis=0)
        # print(f"Predictions with k-folding using {act_funcs[i]}:")
        # print("Prediction:")
        # print(pred[-1][0:5][:])
        # print("Target:")
        # print(target_data[-1][0:5][:])

    plot_bias_accuracy(loss_array, val_loss_array, accuracy_array, val_accuracy_array, act_funcs)
    plot_R2(R2_score_array, N_epochs, act_funcs)
    plot_bias_variance_tradeoff(loss_array, val_loss_array, act_funcs)

# Run PCA with k-folding
def run_kfold_PCA(act_func):
    pca = PCA(n_components = 20)
    pca.fit(eeg)
    xx = pca.transform(eeg)
    X_train, X_test, y_train, y_test = train_test_split(xx, pos_list, test_size=0.2)
    model = NeuralNetwork(X_train, X_test, y_train, y_test)
    pred, target_data, loss, val_loss, accuracy, val_accuracy, loss_2 = model.kfold(
                                                xx.shape[1], N_layers, N_neurons,
                                                N_epochs, N_folds, batch_size,
                                                eta, lmbd, act_func
                                                )

    loss = np.mean(loss, axis=0)
    val_loss = np.mean(val_loss, axis=0)
    accuracy = np.mean(accuracy, axis=0)
    val_accuracy = np.mean(val_accuracy, axis=0)

    print(f"Predictions for PCA with k-folding using {act_func}:")
    print("Prediction:")
    print(pred[-1][0:5][:])
    print("Target:")
    print(target_data[-1][0:5][:])

run_simple_DNN(['relu', 'sigmoid', 'tanh'])
run_kfold_DNN(['relu', 'sigmoid', 'tanh'])
run_kfold_PCA('relu')

def simple_DNN_noise(act_funcs):
    loss_array = np.zeros((len(act_funcs), N_epochs), dtype=object)
    val_loss_array = np.zeros_like(loss_array)
    accuracy_array = np.zeros_like(loss_array)
    val_accuracy_array = np.zeros_like(loss_array)

    pred, loss, val_loss, accuracy, val_accuracy = model.simple(
                                                    inputsize, N_layers, N_neurons,
                                                    N_epochs, batch_size,
                                                    eta, lmbd, act_funcs[0]
                                                    )
    loss_array[0, :] = loss
    val_loss_array[0, :] = val_loss
    accuracy_array[0, :] = accuracy
    val_accuracy_array[0, :] = val_accuracy

    #print("Predictions without k-folding using {act_funcs[i]}:")
    #print(pred[0:5])
    #print(y_test[0:5])

    return val_loss_array, val_accuracy_array

#run_simple_DNN(['relu', 'sigmoid', 'tanh'])
run_kfold_DNN(['relu', 'sigmoid', 'tanh']) #, 'sigmoid', 'tanh'])
#run_kfold_PCA('relu')

N_samples = 1000 # 10000
eeg, pos_list = prepare_data(N_samples)
#eeg = np.load(f'data/eeg_100.npy')
#pos_list = np.load(f'data/pos_list_100.npy')
pos_list = pos_list.T

# Define initial parameters
inputsize = eeg.shape[1]
N_layers = 5
N_neurons = 100
N_epochs = 30 # 30
batch_size = 30
eta = 0.0001
lmbd = 1e-6
N_folds = 10 #10

# Scale the data and append noice
scaler = StandardScaler()
scaler.fit(eeg)
eeg = scaler.transform(eeg)
noise_normal = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10]
#eeg_with_noise = np.tile(eeg,(len(noise),1))
eeg_with_noise = np.array(np.zeros(len(noise_normal)), dtype=object)
#eeg_with_noise = np.concatenate((eeg, np.tile(eeg,len(noise))))
np.random.seed(1235)
f = open("../Textfiles/loss_and_accuracy_given_noise.txt", "w")
f.write('Noise deviation | Accuracy | Loss | \n')
f.write('------------------------- \n')
for i in range(len(noise_normal)):
    #eeg_with_noise[i] = eeg + np.random.normal(0, noise[i], eeg.shape)
    eeg_with_noise = eeg + np.random.normal(0, noise_normal[i], eeg.shape)
    X_train, X_test, y_train, y_test = train_test_split(eeg_with_noise, pos_list, test_size=0.2)
    #loss_median = MSE(y_test, pos_list)

    model = NeuralNetwork(X_train, X_test, y_train, y_test)
    #model_2 = NeuralNetwork(X_train, X_test, y_train, y_test)
    data_1 = simple_DNN_noise(['tanh'])[0]
    data_2 = simple_DNN_noise(['tanh'])[1]
    print(data_1[-1][-1])

    #data_2 = simple_DNN_noise(['tanh'])[1]
    f.write(f' {noise_normal[i]} | {data_1[-1][-1]} | {data_2[-1][-1]} | \n')
    #print(data_1[0][0]) # Henter ut lossen fra test settet.
    #print(data_2[0][1])
f.close()
