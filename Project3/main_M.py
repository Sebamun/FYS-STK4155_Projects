import numpy as np
import LFPy
from lfpykit.eegmegcalc import NYHeadModel
from lfpykit import CellGeometry, CurrentDipoleMoment

def prepare_data():
    nyhead = NYHeadModel()
    dipole_locations = nyhead.cortex[:, 700] #0:74382:10]
    samples = dipole_locations.shape[1]
    eeg = np.zeros((231, samples))

    for i in range(samples):
        nyhead.set_dipole_pos(dipole_locations[:,i])
        M = nyhead.get_transformation_matrix()
        p = np.array(([0.0], [0.0], [1.0]))
        p = nyhead.rotate_dipole_to_surface_normal(p)
        eeg_i = M @ p
        eeg[:, i] = np.reshape(eeg_i, (231))
    return eeg


eeg = prepare_data()
print(eeg.shape)
