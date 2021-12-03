import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def bias_accuracy(act_funcs):
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

