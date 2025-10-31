import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class Galaxies_ML_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        with h5py.File(self.data_dir, 'r') as f:
            self.length = f['image'].shape[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        with h5py.File(self.data_dir, 'r') as f:
            data = np.array(f['image'][idx])
            label_g = np.array(f['g_ellipticity'][idx])
            label_i = np.array(f['i_ellipticity'][idx])
            label_r = np.array(f['r_ellipticity'][idx])
            label_y = np.array(f['y_ellipticity'][idx])
            label_z = np.array(f['z_ellipticity'][idx])
            label = np.array([label_g, label_i, label_r, label_y, label_z])

        if self.transform:
            data = self.transform(data)
        else:
            data = torch.from_numpy(data)

        label = torch.from_numpy(label).float()
        return data, label