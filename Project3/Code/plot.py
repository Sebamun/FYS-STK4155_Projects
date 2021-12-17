import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_R2(R2, N_epochs, act_funcs):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title('R2 score for different activation functions', fontsize=20)
    ax1.set_xlabel('Number of epochs', fontsize=18)
    ax1.set_ylabel('Score', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    for i in range(len(act_funcs)):
        ax1.plot(R2[i], label=f'{act_funcs[i]}')
        ax1.legend(fontsize=18)

    fig1.savefig(f'../Plots/R2')

# def bias_variance_tradeoff(poly_degrees, MSE, bias, variance, N, title, fname):
#     poly_degrees_new = np.arange(1, len(poly_degrees), 2)
#     fig, ax = plt.subplots()
#     ax.set_title(title, fontsize=20)
#     ax.set_xticks(poly_degrees_new)
#     ax.tick_params(axis='both', which='major', labelsize=18)
#     ax.set_title(f'Test data, {N}x{N} datapoints', fontsize=18)
#     ax.plot(poly_degrees, MSE[1], label='MSE')
#     ax.plot(poly_degrees, bias[1], label='Bias')
#     ax.plot(poly_degrees, variance[1], label='Variance')
#     ax.set_xlabel('Polynomial Degree', fontsize=18)
#     ax.tick_params(axis='both', which='major', labelsize=18)
#     ax.legend(fontsize=18)
#     fig.savefig(fname)
#     plt.close(fig)


def plot_bias_accuracy(loss, val_loss, accuracy, val_accuracy, N_epochs, act_funcs):

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title('Training loss with k-folding', fontsize=20)
    ax1.set_xlabel('Number of epochs', fontsize=18)
    ax1.set_ylabel('Bias', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_title('Validation loss with k-folding', fontsize=20)
    ax2.set_xlabel('Number of epochs', fontsize=18)
    ax2.set_ylabel('Bias', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=18)


    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.set_title('Training accuracy with k-folding', fontsize=20)
    ax3.set_xlabel('Number of epochs', fontsize=18)
    ax3.set_ylabel('Accuracy', fontsize=18)
    ax3.tick_params(axis='both', which='major', labelsize=18)


    fig4 = plt.figure()
    ax4 = fig4.add_subplot()
    ax4.set_title('Validation accuracy with k-folding', fontsize=20)
    ax4.set_xlabel('Number of epochs', fontsize=18)
    ax4.set_ylabel('Accuracy', fontsize=18)
    ax4.tick_params(axis='both', which='major', labelsize=18)


    for i in range(len(act_funcs)):
        # plot loss during training
        ax1.plot(loss[i], label=f'Train for {act_funcs[i]}')
        ax1.legend(fontsize=18)
        # Plot validation loss
        ax2.plot(val_loss[i], label=f'Test for {act_funcs[i]}')
        ax2.legend(fontsize=18)
        # plot accuracy during training
        ax3.plot(accuracy[i], label=f'Train for {act_funcs[i]}')
        ax3.legend(fontsize=18)
        # plot validation accuracy
        ax4.plot(val_accuracy[i], label=f'Test for {act_funcs[i]}')
        ax4.legend(fontsize=18)

    fig1.savefig(f'../Plots/loss')
    fig2.savefig(f'../Plots/val_loss')
    fig3.savefig(f'../Plots/accuracy')
    fig4.savefig(f'../Plots/val_accuracy')

def plot_bias_accuracy_simple(loss, val_loss, accuracy, val_accuracy, N_epochs, act_funcs):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title('Training loss without k-folding', fontsize=20)
    ax1.set_xlabel('Number of epochs', fontsize=18)
    ax1.set_ylabel('Bias', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_title('Validation loss without k-folding', fontsize=20)
    ax2.set_xlabel('Number of epochs', fontsize=18)
    ax2.set_ylabel('Bias', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=18)


    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.set_title('Training accuracy without k-folding', fontsize=20)
    ax3.set_xlabel('Number of epochs', fontsize=18)
    ax3.set_ylabel('Accuracy', fontsize=18)
    ax3.tick_params(axis='both', which='major', labelsize=18)


def plot_test_train_loss(loss, val_loss, N_epochs, act_funcs, lmbd, n_comps_list, N_samples, error_std):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    title = r'MSE for training - and test data, $\sigma$ ='+f'{error_std}'
    ax1.set_title(title, fontsize=20)
    ax1.set_xlabel('Number of epochs', fontsize=18)
    ax1.set_ylabel('MSE', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    for i, n_comps in enumerate(n_comps_list):
        # plot loss during training
        ax1.plot(loss[i], label=f'Train, {n_comps} components')
        # Plot validation loss
        ax1.plot(val_loss[i], label=f'Test, {n_comps} components')
        ax1.set_yscale('log')
        ax1.legend(fontsize=16)

    fig1.savefig(f'../Plots/PCA_plots/Test_vs_train_Samples{N_samples}_err{error_std}.png')

def plot_expl_var(expl_var_list, error_std_list):
    colors = ['b--o', 'r--o', 'g--o']
    comps = np.arange(1,11)
    fig, ax = plt.subplots()
    for i, expl_var in enumerate(expl_var_list):
        ax.plot(comps, expl_var*100, colors[i], label=r'$\sigma$='+f'{error_std_list[i]}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(r'Amount of variance [\%]', fontsize=18)
    ax.set_xlabel('Principle Component', fontsize=18)
    ax.set_title(r'Explained Variance ratio', fontsize='20')
    plt.legend()
    fig.savefig(f'../Plots/PCA_plots/Expl_Var.png')
    for i in range(6, 10):
        print(f'Sum of the {i+1} first PCs = {np.sum(expl_var_list[0][0:i]*100)}')


def plot_test_loss(val_loss, N_epochs, n_comps_list, error_std):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    title = r'MSE for test data, $\sigma$ ='+f'{error_std}'
    ax1.set_title(title, fontsize=20)
    ax1.set_xlabel('Number of epochs', fontsize=18)
    ax1.set_ylabel('MSE', fontsize=18)
    # ax1.set_ylim(0, 100)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    for i, n_comps in enumerate(n_comps_list):
        # Plot validation loss
        ax1.plot(val_loss[i], label=f'{n_comps} components')
        ax1.set_yscale('log')
        ax1.legend(fontsize=16)

    fig1.savefig(f'../Plots/PCA_plots/Test_MSE_err{error_std}.png')

def plot_test_train_reduced(loss, val_loss, N_epochs, act_funcs, lmbd, N_samples, error_std):
    fig, axes = plt.subplots(1,2)

    title = r'MSE, $\sigma$ ='+f'{error_std}'
    axes[0].set_title('Uniformly distributed EEG-signals', fontsize=18)
    axes[0].set_xlabel('Number of epochs', fontsize=18)
    axes[0].set_ylabel('MSE', fontsize=18)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    # plot loss during training
    axes[0].plot(loss[0], label=f'Train')
    # Plot validation loss
    axes[0].plot(val_loss[0], label=f'Test')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=16)
    axes[1].set_title('First ten EEG-signals', fontsize=18)
    axes[1].set_xlabel('Number of epochs', fontsize=18)
    axes[1].set_ylabel('MSE', fontsize=18)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    # plot loss during training
    axes[1].plot(loss[1], label=f'Train')
    # Plot validation loss
    axes[1].plot(val_loss[1], label=f'Test')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=16)
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()

    fig.savefig(f'../Plots/PCA_plots/Test_vs_train_reduced_err{error_std}.png')


def plot_EEG(eeg):
    N = np.arange(1, eeg.shape[1] + 1)
    idx = np.random.randint(0, eeg.shape[1])
    plt.plot(N, eeg[idx])
    plt.show()
