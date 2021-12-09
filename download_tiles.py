import os
import sys
import glob
import pandas as pd
import numpy as np

#directories
download_dir = '/scratch/merileo/unions/tiles/'
cfis_tiles_dir = '/home/merileo/projects/rrg-kyi/astro/cfis/W3'

prev_downloaded_files = glob.glob(cfis_tiles_dir + '/*')

# Data that has already been processed to ensure tiles in common for HSC-g, CFIS-r, and PS-i
print('Loading in unions data...')
unions_data = pd.read_csv('/scratch/merileo/unions.ugriz2_matchingbands_PSiHSCgCFISr.tsv', 
                           header=0, 
                           delim_whitespace=True, 
                           dtype={'tile': str})

tiles = np.unique(unions_data.tile)

cam_to_dir = { 'cfis' : {'vos': 'vcp -v vos:cfis/tiles_DR3/', 'band': 'r'},
                'ps1' : {'vos': 'vcp -L -v vos:cfis/panstarrs/DR2/skycell.', 'band': 'i'},
                'hsc' : {'vos': 'vcp -v vos:cfis/hsc/stacks2/', 'band': 'g'}}


# panstarrs_filename_format -> 'skycell.{skycell}/CFIS.V0.skycell.{tile}*unconv.fits'

for j, tile in enumerate(tiles):
    print('{} of {} tiles downloaded'.format(j, len(tiles)))
    skycell = tile[:3]
    
    for cam in cam_to_dir.keys():
        vos_copy_command = cam_to_dir[cam]['vos']
        band = cam_to_dir[cam]['band']
        save_dir = os.path.join(download_dir, cam)
        
        # Get fits filename
        if cam=='ps1':
            tile_fitsfilename = '{}/CFIS.V0.skycell.{}*unconv.fits'.format(skycell, tile)
            tile_catfilename = None
        else:
            tile_fitsfilename = '{}.{}.{}.fits'.format(cam.upper(), tile, band)
            tile_catfilename = '{}.{}.{}.cat'.format(cam.upper(), tile, band)
            
        # Check if already downloaded
        if os.path.exists(os.path.join(save_dir, tile_fitsfilename)) or tile_fitsfilename in prev_downloaded_files:
            print('Already downloaded this tile, moving on...')
            break
        
        # Issue command to download for vospace
        try:
            os.system(vos_copy_command + tile_fitsfilename + ' {}'.format(save_dir) )
        except e as error:
            print(error)
            continue
        #if band in ['u','r','g']:
        if tile_catfilename is not None:
            if tile_catfilename  not in prev_downloaded_files:
                os.system(vos_copy_command + tile_catfilename + ' {}'.format(save_dir) )
