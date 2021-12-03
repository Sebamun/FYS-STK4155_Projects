import numpy as np
import LFPy
from lfpykit.eegmegcalc import NYHeadModel
from lfpykit import CellGeometry, CurrentDipoleMoment
from sklearn.preprocessing import StandardScaler
from common import FrankeFunction

np.random.seed(1234)

def prepare_data(num_signals):
    nyhead = NYHeadModel()
    step_length = nyhead.cortex.shape[1]//num_signals + 1
    dipole_locations = nyhead.cortex[:, ::step_length] # 8 positions
    samples = 1000
    pos_list = np.zeros((3, samples))
    for i in range(samples):
        idx = np.random.randint(0,8)
        pos_list[:, i] = dipole_locations[:, idx]
    eeg = np.zeros((samples, 231))

    for i in range(samples):
        nyhead.set_dipole_pos(pos_list[:,i])
        M = nyhead.get_transformation_matrix()
        p = np.array(([0.0], [0.0], [1.0]))
        p = nyhead.rotate_dipole_to_surface_normal(p)
        eeg_i = M @ p
        eeg[i, :] = eeg_i.T
    return eeg, pos_list

for num in [8, 20, 50, 100]:
    eeg, pos_list = prepare_data(num_signals=num)
    np.save(f'data/eeg_{num}', eeg)
    np.save(f'data/pos_list_{num}', pos_list)
quit()