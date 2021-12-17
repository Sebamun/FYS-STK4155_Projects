import numpy as np

def load_data(num_samples):
    try:
        eeg = np.load(f'data/eeg_{num_samples}.npy')
        pos_list = np.load(f'data/pos_list_{num_samples}.npy')
    except FileNotFoundError as e:
        print(f'Eeg data has not been produced for {num_samples} samples.')
        raise e
    pos_list = pos_list.T
    return eeg, pos_list
