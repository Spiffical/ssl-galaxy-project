import os
import sys
import glob
import pandas as pd
import numpy as np

# directories
download_dir = '/scratch/merileo/unions/tiles/'
cfis_tiles_dir = '/home/merileo/projects/rrg-kyi/astro/cfis/W3'

prev_downloaded_files = glob.glob(cfis_tiles_dir + '/*')

# Data that has already been processed to ensure tiles in common for HSC-g, CFIS-r, and PS-i
print('Loading in unions data...')
unions_data = pd.read_parquet('/scratch/merileo/unions/unions.matchingbands.ugriz.parquet', engine='fastparquet')
# unions_data = pd.read_parquet('/scratch/merileo/unions/unions.ugriz2.matchingbands.ugriz.parquet', engine='pyarrow')
# unions_data = pd.read_csv('/scratch/merileo/unions/unions.ugriz2.matchingbands.ugriz.tsv',
#                           header=0,
#                           delim_whitespace=True,
#                           dtype={'tile': str})

tiles = unions_data.tile

cam_to_dir = {'cfis': {'vos': 'vcp -v vos:cfis/tiles_DR3/', 'band': 'u'},
              'hsc': {'vos': 'vcp -v vos:cfis/hsc/stacks2/', 'band': 'g'},
              'cfis_lsb': {'vos': 'vcp -v vos:cfis/tiles_LSB_DR3/', 'band': 'r'},
              'ps1': {'vos': 'vcp -L -v vos:cfis/panstarrs/DR2/skycell.', 'band': 'i'}}

indices = np.load('/scratch/merileo/unions/unions.indices.training.npy')
num_train = 1000000

indices_train = indices[:num_train]
tiles = np.unique(tiles.iloc[indices_train])
print(tiles)

# panstarrs_filename_format -> 'skycell.{skycell}/CFIS.V0.skycell.{tile}*unconv.fits'

for j, tile in enumerate(tiles):
    print('{} of {} tiles downloaded'.format(j, len(tiles)))
    skycell = tile[:3]

    for cam in cam_to_dir.keys():
        vos_copy_command = cam_to_dir[cam]['vos']
        band = cam_to_dir[cam]['band']

        save_dir = os.path.join(download_dir, cam)

        # Get fits filename
        if cam == 'ps1':
            tile_fitsfilename = '{}/CFIS.V0.skycell.{}*unconv.fits'.format(skycell, tile)
            tile_fitsfilename_saved = 'PS1.{}.i.fits'.format(tile)
            # tile_fitsfilename_saved = 'PS1.{}.z.fits'.format(tile)
            tile_catfilename = None
        else:
            tile_fitsfilename = tile_fitsfilename_saved = '{}.{}.{}.fits'.format(cam.upper(), tile, band)
            tile_catfilename = '{}.{}.{}.cat'.format(cam.upper(), tile, band)

        # Check if already downloaded
        if os.path.exists(
                os.path.join(save_dir, tile_fitsfilename_saved)) or tile_fitsfilename_saved in prev_downloaded_files:
            print('Already downloaded {}, moving on...'.format(tile_fitsfilename_saved))
            continue

        # Issue command to download for vospace
        try:
            os.system(vos_copy_command + tile_fitsfilename + ' {}/'.format(save_dir))
        except e as error:
            print(error)
            continue
        # if band in ['u','r','g']:
        if tile_catfilename is not None:
            if tile_catfilename not in prev_downloaded_files:
                os.system(vos_copy_command + tile_catfilename + ' {}/'.format(save_dir))