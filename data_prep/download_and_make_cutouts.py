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
import pandas as pd
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
cutout_save_folder = '/scratch/merileo/unions/cutouts/ugriz_lsb/per_tile/'
h5_filename = 'cutout_stacks_ugriz_lsb_200x200'
SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')

cam_dict = {'cfis-u': {'name': 'cfis', 'band': 'u', 'vos': 'vcp -v vos:cfis/tiles_DR3/'},
            'hsc-g': {'name': 'hsc', 'band': 'g', 'vos': 'vcp -v vos:cfis/hsc/stacks2/'},
            'cfis_lsb': {'name': 'cfis_lsb', 'band': 'r', 'vos': 'vcp -v vos:cfis/tiles_LSB_DR3/'},
            'ps1-i': {'name': 'ps1', 'band': 'i', 'vos': 'vcp -L -v vos:cfis/panstarrs/DR2/skycell.'},
            'ps1-z': {'name': 'ps1', 'band': 'z', 'vos': 'vcp -L -v vos:cfis/panstarrs/DR2/skycell.'}}


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


def download_tile_for_each_cam(tile_id, cam_to_dir):
    skycell = tile_id[:3]

    for cam in cam_to_dir.keys():
        vos_copy_command = cam_to_dir[cam]['vos']
        band = cam_to_dir[cam]['band']
        cam_name = cam_to_dir[cam]['name']

        # Get fits filename
        if cam_to_dir[cam]['name'] == 'ps1':
            tile_fitsfilename = '{}/CFIS.V0.skycell.{}*unconv.fits'.format(skycell, tile)
        else:
            tile_fitsfilename = '{}.{}.{}.fits'.format(cam_name.upper(), tile, band)

        # Issue command to download for vospace
        try:
            os.system(vos_copy_command + tile_fitsfilename + ' {}/'.format(SLURM_TMPDIR))
        except Exception as e:
            print(e)
            return False

    return True


def make_cutouts_for_each_cam(tile_dir, cam_dict, ra_dec_list, cfis_id_list):
    tile_ids = []
    cfis_ids = []
    ras = []
    decs = []
    for j, cam in enumerate(cam_dict.keys()):
        tile_fitsfilename = '{}.{}.{}.fits'.format(cam_dict[cam]['name'].upper(), tile, cam_dict[cam]['band'])
        tile_fits_filepath = os.path.join(tile_dir, tile_fitsfilename)
        cam_dict[cam]['cutouts'] = []
        print(cam)

        with fits.open(tile_fits_filepath, memmap=True) as image:
            with fitsio.FITS(tile_fits_filepath) as fits_:
                for k, (ra, dec) in enumerate(ra_dec_list):
                    X, Y = ra_dec_to_xy(fits_, ra, dec, cam_dict[cam]['band'])
                    if cam_dict[cam]['name'] == 'ps1':
                        cutout = make_cutout(image[1], X.item(), Y.item())
                    else:
                        cutout = make_cutout(image[0], X.item(), Y.item())
                    cam_dict[cam]['cutouts'].append(cutout)
                    if j == len(cam_dict.keys()) - 1:
                        tile_ids.append(tile)
                        ras.append(ra)
                        decs.append(dec)
                        cfis_ids.append(cfis_id_list[k])
    return tile_ids, cfis_ids, ras, decs


def rename_ps1(download_dir):
    # Rename PS1 tiles to be consistent with CFIS and HSC tiles, and also to make clear whether it's i or z band
    ps1_tile_filepaths = glob.glob(os.path.join(download_dir, '*skycell*'))
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


# Load data that has already been cleaned to ensure tiles in common for ugriz
print('Loading in unions data...')
unions_data = pd.read_parquet('/scratch/merileo/unions/catalogs/unions.matchingbands.ugriz.parquet', engine='fastparquet')

count = 0
num_sources_extracted = 0
total_num_sources = len(unions_data)

for tile in tiles:
    skycell = tile[:3]
    print(tile)
    save_path = os.path.join(cutout_save_folder, h5_filename + '_{}.h5'.format(tile))
    if os.path.exists(save_path):
        print('Already done tile: {}\nSkipping!'.format(tile))
        continue

    # Gather all sources in this tile
    partitioned_data = unions_data[unions_data.tile == tile]

    ra_dec_list = partitioned_data[['ra', 'dec']].values
    cfis_id_list = partitioned_data['id'].values

    n_sources = len(ra_dec_list)
    print('There are {} sources to cut out'.format(n_sources))

    # Download each camera's tile
    success = download_tile_for_each_cam(tile, cam_dict)
    if not success:
        continue

    # Rename the PS1 files
    rename_ps1(SLURM_TMPDIR)

    # Make cutouts for this tile with each cam
    tile_id_array, cfis_id_array, ra_array, dec_array = \
        make_cutouts_for_each_cam(SLURM_TMPDIR, cam_dict, ra_dec_list, cfis_id_list)

    # Stack all the cutouts along the filter dimension
    stacked_cutout_array = np.stack([cam_dict[cam]['cutouts'] for cam in cam_dict.keys()], 1)

    print('Saving file: {}'.format(save_path))
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('images', data=np.asarray(stacked_cutout_array))
        hf.create_dataset('tile', data=np.asarray(tile_id_array, dtype=dt))
        hf.create_dataset('cfis_id', data=np.asarray(cfis_id_array))
        hf.create_dataset('ra', data=np.asarray(ra_array))
        hf.create_dataset('dec', data=np.asarray(dec_array))
