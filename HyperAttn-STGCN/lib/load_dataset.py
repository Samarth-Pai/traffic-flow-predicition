import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset_path):
    # output B, N, D
    # Backward compatibility with original code
    print("DEBUG dataset_path =", repr(dataset_path))
    if dataset_path == 'PEMSD4':
        dataset_path = os.path.join('../data/PeMSD4/pems04.npz')
    elif dataset_path == 'PEMSD8':
        dataset_path = os.path.join('../data/PeMSD8/pems08.npz')
        
    if dataset_path.endswith('.h5'):
        # For HDF5 files (METR-LA and PEMS-BAY), read via pandas
        df = pd.read_hdf(dataset_path)
        data = df.values
    elif dataset_path.endswith('.npz'):
        # For NPZ files, load using numpy
        data = np.load(dataset_path)['data']
        # If it has 3 dimensions (B, N, D), extract the first feature (traffic flow)
        if len(data.shape) == 3:
            data = data[:, :, 0]
    else:
        raise ValueError(f"Unsupported dataset format or path: {dataset_path}")

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load Dataset from %s shaped: ' % dataset_path, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
