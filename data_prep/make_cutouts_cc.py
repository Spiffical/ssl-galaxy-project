import os
import shutil
import h5py
import random
import glob
import argparse
import numpy as np
from astropy.nddata.utils import Cutout2D
from shutil import copyfile
import fitsio
from astropy.io import fits
from astropy.table import Table
import pandas as pd
from astropy.visualization import (ZScaleInterval, ImageNormalize)
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--tiles', nargs='+', required=True,
                    help='list of tiles to make cutouts from')

args = parser.parse_args()
tiles = args.tiles

# important parameters:
cutout_size = 200
download_dir = '/scratch/merileo/unions/tiles/'
cutout_save_folder = '/scratch/merileo/unions/cutouts'
h5_filename = 'cutout_stacks_ugriz_lsb_200x200'
SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')

cam_dict = {'cfis-u': {'name': 'cfis', 'band': 'u'},
            'hsc-g': {'name': 'hsc', 'band': 'g'},
            'cfis_lsb': {'name': 'cfis_lsb', 'band': 'r'},
            'ps1-i': {'name': 'ps1', 'band': 'i'},
            'ps1-z': {'name': 'ps1', 'band': 'z'}}


def make_cutout(img, x, y):
    ''' Creates the image cutout given a tile, the position of the center and the band '''

    img_cutout = Cutout2D(img.data, (x, y), cutout_size, mode="partial", fill_value=0).data

    if np.count_nonzero(np.isnan(img_cutout)) >= 0.05 * cutout_size ** 2 or np.count_nonzero(
            img_cutout) == 0:  # Don't use this cutout
        return np.zeros((cutout_size, cutout_size))

    img_cutout[np.isnan(img_cutout)] = 0

    return img_cutout


def ra_dec_to_xy(fits, ra, dec, band):
    if band in ['i', 'z']:
        head = fits[1].read_header()
    else:
        head = fits[0].read_header()
    w = WCS(head)
    return skycoord_to_pixel(SkyCoord(ra, dec, unit="deg"), w)


# Rename PS1 tiles to be consistent with CFIS and HSC tiles, and also to make clear whether it's i or z band
ps1_tile_filepaths = glob.glob(os.path.join(download_dir, 'ps1/*skycell*'))
num_files = len(ps1_tile_filepaths)
for i, filepath in enumerate(ps1_tile_filepaths):
    print('Renamed {} of {} tiles'.format(i, num_files))
    # get band
    fits_ = fitsio.FITS(filepath)
    band = fits_[1].read_header()['HIERARCH FPA.FILTER'].split('.')[0]

    # get tile + type of file info
    filename = os.path.basename(filepath)
    key_words = filename.split('.')
    tile = key_words[3] + '.' + key_words[4]
    img_or_wt = key_words[8]

    if img_or_wt == 'wt':
        new_name = 'PS1.{}.{}.wt.fits'.format(tile, band)
    elif img_or_wt == 'fits':
        new_name = 'PS1.{}.{}.fits'.format(tile, band)

    new_filepath = os.path.join(os.path.dirname(filepath), new_name)

    os.rename(filepath, new_filepath)

# Data that has already been processed to ensure tiles in common for HSC-g, CFIS-r, and PS-i
print('Loading in unions data...')
unions_data = pd.read_parquet('/scratch/merileo/unions/unions.matchingbands.ugriz.parquet', engine='fastparquet')

# Get only the rows that are included in the training set
indices = np.load('/scratch/merileo/unions/unions.indices.training.npy')
num_train = 1000000
unions_data = unions_data.iloc[indices]

print('starting cutout extraction')

count = 0
num_sources_extracted = 0
total_num_sources = len(unions_data)

for tile in tiles:
    print(tile)
    save_path = os.path.join(cutout_save_folder, h5_filename + '_{}.h5'.format(tile))
    if os.path.exists(save_path):
        print('Already done tile: {}\nSkipping!'.format(tile))
        continue

    stacked_cutout_array = []
    tile_id_array = []
    cfis_id_array = []
    ra_array = []
    dec_array = []

    # Gather all sources in this tile
    partitioned_data = unions_data[unions_data.tile == tile]

    ra_dec_list = partitioned_data[['ra', 'dec']].values
    cfis_id_list = partitioned_data['id'].values

    n_sources = len(ra_dec_list)
    print('There are {} sources to cut out'.format(n_sources))

    for cam in cam_dict.keys():
        tile_fitsfilename = '{}.{}.{}.fits'.format(cam_dict[cam]['name'].upper(), tile, cam_dict[cam]['band'])
        # tile_catfilename = '{}.{}.{}.cat'.format(cam.upper(), tile, cam_dict[cam]['band'])
        tile_fitsfilepath = os.path.join(download_dir, cam_dict[cam]['name'], tile_fitsfilename)
        cam_dict[cam]['tile_filepath'] = tile_fitsfilepath

    # If the CFIS tile doesn't exist in the download directory, it might exist in the projects dir
    if not os.path.exists(cam_dict['cfis-u']['tile_filepath']):
        cam_dict['cfis-u']['tile_filepath'] = '/home/merileo/projects/rrg-kyi/astro/cfis/W3/CFIS.{}.u.fits'.format(tile)

    # If any tile is missing, skip
    if not all([os.path.exists(cam_dict[cam]['tile_filepath']) for cam in cam_dict.keys()]):
        print('Missing tile from at least one of the cameras! \n{}'.format(cam_dict))
        continue

    # Copy files to slurm temp directory
    for cam in cam_dict.keys():
        tile_fitsfilepath = cam_dict[cam]['tile_filepath']
        filename = os.path.basename(tile_fitsfilepath)
        copyfile(tile_fitsfilepath, os.path.join(SLURM_TMPDIR, filename))
        cam_dict[cam]['tile_filepath'] = os.path.join(SLURM_TMPDIR, filename)

    for j, cam in enumerate(cam_dict.keys()):
        cam_dict[cam]['cutouts'] = []
        print(cam)

        with fits.open(cam_dict[cam]['tile_filepath'], memmap=True) as image:
            with fitsio.FITS(cam_dict[cam]['tile_filepath']) as fits_:
                for k, (ra, dec) in enumerate(ra_dec_list):
                    X, Y = ra_dec_to_xy(fits_, ra, dec, cam_dict[cam]['band'])
                    if cam_dict[cam]['name'] == 'ps1':
                        cutout = make_cutout(image[1], X.item(), Y.item())
                    else:
                        cutout = make_cutout(image[0], X.item(), Y.item())
                    cam_dict[cam]['cutouts'].append(cutout)
                    if j == len(cam_dict.keys()) - 1:
                        tile_id_array.append(tile)
                        ra_array.append(ra)
                        dec_array.append(dec)
                        cfis_id_array.append(cfis_id_list[k])
                    #    num_sources_extracted+=1
                    #    count+=1

    stacked_cutouts = np.stack([cam_dict[cam]['cutouts'] for cam in cam_dict.keys()], 1)
    stacked_cutout_array.extend(stacked_cutouts)

    save_path = os.path.join(cutout_save_folder, h5_filename + '_{}.h5'.format(tile))
    print('Saving file: {}'.format(save_path))
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('images', data=np.asarray(stacked_cutout_array))
        hf.create_dataset('tile', data=np.asarray(tile_id_array, dtype=dt))
        hf.create_dataset('cfis_id', data=np.asarray(cfis_id_array))
        hf.create_dataset('ra', data=np.asarray(ra_array))
        hf.create_dataset('dec', data=np.asarray(dec_array))
