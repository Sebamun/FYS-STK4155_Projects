import numpy as np
import LFPy
from lfpykit.eegmegcalc import NYHeadModel
from lfpykit import CellGeometry, CurrentDipoleMoment

def prepare_data(num_samples):

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
