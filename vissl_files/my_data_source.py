from torch.utils.data import Dataset
import h5py
import glob
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from astropy import units as u

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


        #for fp in self.filepaths:
        #    with h5py.File(fp, 'r') as _f:
        #        n_samples = _f['images'].shape[0]

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

        image = None
        try:
            # Retrieve data
            image = self.files[ifile]['images'][local_idx]
            ra = self.files[ifile]['ra'][local_idx]
            dec = self.files[ifile]['dec'][local_idx]
            image = np.swapaxes(image, 0, 2)  # Needs to be flipped for input to SimCLR
            image = np.float32(image)

            # De-redden the image
            coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
            sfd = SFDQuery()
            sfd_ebv = sfd(coords)
            # Taking conversion factors of sdss-u, sdss-g, sdss-r, ps-i, ps-z from:
            # https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103#apj398709t6
            conversion_factor = np.array([4.239, 3.303, 2.285, 1.682, 1.322])
            true_ext = conversion_factor * sfd_ebv
            deredden_factor = 10. ** (true_ext / 2.5)
            image = np.float32(image * deredden_factor[None, None, :])
            #image = np.float32([image[j, :, :] * deredden_factor[j] for j in range(5)])  # deredden

            is_success = True
        except Exception as e:
            print(e)
            is_success = False

        # is_success should be True or False indicating whether loading data was successful or failed
        # loaded data should be Image.Image if image data
        return image, is_success
