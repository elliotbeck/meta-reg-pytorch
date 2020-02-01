import torch.utils.data as data
import numpy as np
import torch
import h5py

class HDF5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(HDF5Dataset, self).__init__()
        with h5py.File(file_path) as hf:
            self.data = np.asarray(hf['images'])
            self.target = (np.expand_dims(np.asarray(hf['labels']), -1)-1)

    def __getitem__(self, index):
        return torch.from_numpy(np.asarray(self.data[index,:,:,:]/255., dtype=np.float16)).float(), torch.from_numpy(np.asarray(self.target[index], dtype=np.int16)).float()
        
    def __len__(self):
        return list(self.data.shape)[0]