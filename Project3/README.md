# Applied Data Analysis and Machine Learning, Project 2
## Biophysical modeling of EEG signals from current dipoles in human cortex

Kamilla Ida Julie Sulebakk, Marcus Berget and Sebastian Amundsen

In this project we have modeled EEG signals from current dipoles and utilized a feed-forward neural network (FFNN) to efficient locate the dipoles in the human cortex. We have studied the performance of the network with different activation functions, resampling methods and done a principal component analysis (PCA).

## The project is created with:
* Python version: 3
  * sklearn
  * numpy
  * matplotlib
  * autograd
  * mpl_toolkits

  * (NYHeadModel)

* LaTeX

## How to run the code:
Open terminal window, these commands compile and execute the programs:
```
# To compile and execute FFNN
python3 run_NN.py

# To compile and execute PCA
python3 run_PCA.py

# To model your own EEG signals with the desired number of patiens
python3 produce_eeg_data.py

```
Here;
* python3 run_NN.py fits EEG data from 1000 samples using FFNN with and without k-folding and:
  * provides plots of loss, accuracy and bias-variance-tradeoff
  * writes the loss and accuracy values given different noise deviations to file
    * with 231 features, batch size of 30 samples, 5 hidden layers, 100 neurons, 30 epochs and 10 folds.

* python3 run_PCA.py fits EEG data from 1000 patients using FFNN with k-folding and:
   * Train the NN varying the number of principle components. To change how many components one wants to use, edit the n_comps_list
   * Train the network w/o PCA, but with 10 extraced features, first 10 uniformly distributed, and then first 10.
   * Make a scree plot of the PC's, varying the noise.

* python3 produce_eeg_data.py places a dipole moment in cortex of human brain for N number of patients and:  
  * writes EEG and localizations to files 
