import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def bias_accuracy(loss, val_loss, accuracy, val_accuracy, N_epochs, act_func):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    # plot loss during training
    ax1.plot(loss, label=f'Train data for {act_func}')
    ax1.legend(fontsize=18)
    ax1.set_title('Training loss', fontsize=20)
    ax1.set_xlabel('Number of epochs', fontsize=18)
    ax1.set_ylabel('bias', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    fig1.savefig(f'../Plots/loss_{act_func}')
    plt.close(fig1)


    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    # Plot validation loss
    ax2.plot(val_loss, label=f'Test data for {act_func}')
    ax2.legend(fontsize=18)
    ax2.set_title('Validation loss', fontsize=20)
    ax2.set_xlabel('Number of epochs', fontsize=18)
    ax2.set_ylabel('bias', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    fig2.savefig(f'../Plots/val_loss_{act_func}')
    plt.close(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    # plot accuracy during training
    ax3.plot(accuracy, label=f'Train data for {act_func}')
    ax3.legend(fontsize=18)
    ax3.set_title('Training accuracy', fontsize=20)
    ax3.set_xlabel('Number of epochs', fontsize=18)
    ax3.set_ylabel('accuracy', fontsize=18)
    ax3.tick_params(axis='both', which='major', labelsize=18)
    fig3.savefig(f'../Plots/accuracy_{act_func}')
    plt.close(fig3)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot()
    # plot validation accuracy
    ax4.plot(val_accuracy, label=f'Test data for {act_func}')
    ax4.legend(fontsize=18)
    ax4.set_title('Validation accuracy', fontsize=20)
    ax4.set_xlabel('Number of epochs', fontsize=18)
    ax4.set_ylabel('accuracy', fontsize=18)
    ax4.tick_params(axis='both', which='major', labelsize=18)
    fig4.savefig(f'../Plots/val_accuracy_{act_func}')
    plt.close(fig4)





