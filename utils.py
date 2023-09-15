import os
import fitsio
import numpy as np
import glob
from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.wcs.utils import skycoord_to_pixel



def coord_to_tile(ra, dec):
    '''takes ra and dec coord (in degrees) and outputs the tile name centered there'''
    #convert to radians
    yyy = int(round((2*(dec+90))))
    cosf = np.cos(np.deg2rad(yyy/2-90))
    xxx = int(round(ra*2*cosf))

    xxx = '{0:0=3d}'.format(xxx)
    yyy = '{0:0=3d}'.format(yyy)

    if xxx=='720':
        xxx='000'

    return xxx + '.' + yyy


def make_cutout(img, x, y, cutout_size):
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


def download_tile_for_each_cam(tile_id, cam_to_dir, download_dir):
    skycell = tile_id[:3]

    for cam in cam_to_dir.keys():
        vos_copy_command = cam_to_dir[cam]['vos']
        band = cam_to_dir[cam]['band']
        cam_name = cam_to_dir[cam]['name']

        # Get fits filename
        if cam_to_dir[cam]['name'] == 'ps1':
            tile_fitsfilename = '{}/CFIS.V0.skycell.{}*unconv.fits'.format(skycell, tile_id)
        else:
            tile_fitsfilename = '{}.{}.{}.fits'.format(cam_name.upper(), tile_id, band)

        if os.path.exists(os.path.join(download_dir, tile_fitsfilename)):
            print('Tile downloaded already.')
            return True

        # Issue command to download for vospace
        try:
            os.system(vos_copy_command + tile_fitsfilename + ' {}/'.format(download_dir))
        except Exception as e:
            print(e)
            return False

    return True


def make_cutouts_for_each_cam(tile_dir, cam_dict, ra, dec, tile_id):
    for j, cam in enumerate(cam_dict.keys()):
        tile_fitsfilename = '{}.{}.{}.fits'.format(cam_dict[cam]['name'].upper(), tile_id, cam_dict[cam]['band'])
        tile_fits_filepath = os.path.join(tile_dir, tile_fitsfilename)
        cam_dict[cam]['cutouts'] = []
        print(cam)

        with fits.open(tile_fits_filepath, memmap=True) as image:
            with fitsio.FITS(tile_fits_filepath) as fits_:
                X, Y = ra_dec_to_xy(fits_, ra, dec, cam_dict[cam]['band'])
                if cam_dict[cam]['name'] == 'ps1':
                    cutout = make_cutout(image[1], X.item(), Y.item())
                else:
                    cutout = make_cutout(image[0], X.item(), Y.item())
                cam_dict[cam]['cutouts'].append(cutout)
    return


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
