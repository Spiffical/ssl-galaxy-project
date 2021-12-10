from vissl.data.data_helper import get_mean_image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image
import h5py
import torch
import glob
import numpy as np

class Multih5DataSource(Dataset):
    """
    add documentation on how this dataset works

    Args:
        add docstrings for the parameters
    """

    def __init__(self, cfg, data_source, path, split, dataset_name):
        super(Multih5DataSource, self).__init__()
        assert data_source in [
            "disk_filelist",
            "disk_folder",
            "my_data_source"
        ], "data_source must be either disk_filelist or disk_folder or my_data_source"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        self.filepaths = glob.glob(path + '/*')

        # implement anything that data source init should do'
        self.n_files = len(self.filepaths)
        with h5py.File(self.filepaths[0], 'r') as _f:
            self.n_samples_per_file = _f['images'].shape[0]

        self.files = [None for _ in range(self.n_files)]

    def _open_file(self, ifile):
        self.files[ifile] = h5py.File(self.filepaths[ifile], 'r')

    def num_samples(self):
        """
        Size of the dataset
        """
        n_samples = self.n_files * self.n_samples_per_file
        return n_samples

    def __len__(self):
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx: int):
        """
        implement how to load the data corresponding to idx element in the dataset
        from your data source
        """
        ifile = int(idx / self.n_samples_per_file)
        local_idx = int(idx % self.n_samples_per_file)

        if not self.files[ifile]:
            self._open_file(ifile)

        try:
            #with h5py.File(self.filenames[ifile], 'r') as _f:
            image = self.files[ifile]['images'][local_idx]
            image = np.swapaxes(image, 0, 2)
            image = np.float32(image)
            #print(image.dtype)
            is_success = True
        except Exception as e:
            print(e)
            is_success = False

        # is_success should be True or False indicating whether loading data was successful or failed
        # loaded data should be Image.Image if image data
        return image, is_success
