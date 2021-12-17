import numpy as np
from lfpykit.eegmegcalc import NYHeadModel

def prepare_data(num_samples):
    """
    Create an instance of the New York Head Model and produce eeg-signals.

    input:
        num_samples (int): the number of samples/patients

    returns:
        eeg (numpy array of floats, size (num_samples, 231)):
            electrical signals from the dipole in the cortex
        pos_list (numpy array of floats, size (3, num_samples)):
            the position of the dipole for each of the samples
    """
    nyhead = NYHeadModel()
    dipole_locations = nyhead.cortex      # Der svulsten kan plasseres
    num_positions = dipole_locations.shape[1]

    pos_list = np.zeros((3, num_samples))   # Posisjonene til svulstene
    for i in range(num_samples): # 0 til 1000
        idx = np.random.randint(0, num_positions) # Tilfeldig valgt posisjon blant X
        pos_list[:, i] = dipole_locations[:, idx] # Legger til posisjonen til svulsten

    eeg = np.zeros((num_samples, 231))

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

<<<<<<< HEAD:Project3/Code/produce_eeg_data.py
if __name__ == '__main__':
    for num_samples in [500, 1_000, 10_000]:
        eeg, pos_list = prepare_data(num_samples)
        np.save(f'data/eeg_{num_samples}', eeg)
        np.save(f'data/pos_list_{num_samples}', pos_list)
=======
def R2(y_data, y_model):
    """Calculate R2 score"""
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data, y_model):
    """Calculate MSE"""
    return np.mean((y_data - y_model)**2)
>>>>>>> 70fbb0b7cada5507123011678113ee2be507f947:Project3/Code/common.py
