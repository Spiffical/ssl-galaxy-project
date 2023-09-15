import numpy as np
import pandas as pd
import math
from astropy.wcs import WCS,utils

unions_data = pd.read_parquet('/scratch/merileo/unions/unions.ugriz.800deg2.xmatch.parquet',
                              engine='fastparquet')

# Get all the data that has simultaneous ugriz photometry
unions_data = unions_data[(unions_data['i_ps']>0) &
                          (unions_data['g_hsc']>0) &
                          (unions_data['r_cfht']>0) & (unions_data['r_cfht']<24.5) &
                          (unions_data['u_cfht']>0) &
                          (unions_data['z_ps']>0)]


# useful functions
def tile_to_coord(tile, return_ra=False, return_dec=False):
    '''correspondance in degrees of ra,dec of tile name'''
    xxx, yyy = tile.split('.')
    dec = int(yyy) / 2 - 90
    ra = int(xxx) / (2 * np.cos(math.radians(dec)))

    if return_ra is True:
        return ra
    if return_dec is True:
        return dec


def coord_to_tile(ra, dec, row):
    '''takes ra and dec coord (in degrees) and outputs the tile name centered there'''
    # convert to radians
    try:
        yyy = int(round((2 * (dec + 90))))
        cosf = np.cos(np.deg2rad(yyy / 2 - 90))
        xxx = int(round(ra * 2 * cosf))

        xxx = '{0:0=3d}'.format(xxx)
        yyy = '{0:0=3d}'.format(yyy)

        if xxx == '720':
            xxx = '000'

        return xxx + '.' + yyy
    except e as error:
        print(error)
        print('ra and dec are not a number', row)
        return 0

# Add in tile information
unions_data['tile'] = unions_data.apply(lambda row: coord_to_tile(row['ra'], row['dec'], row), axis=1)

unions_data.to_parquet('unions.ugriz2.matchingbands.ugriz.parquet', engine='fastparquet')