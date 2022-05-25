import os
import torch
import h5py
import glob
import random

import numpy as np
import pandas as pd
from torchvision.models import resnet50
from torch.nn import Identity
from scipy import spatial
import torchvision.transforms as T

from utils import coord_to_tile, make_cutout, make_cutouts_for_each_cam, rename_ps1, download_tile_for_each_cam


cropper = T.CenterCrop(size=140)
totensor = T.ToTensor()

base_model_dir = '/scratch/merileo/visslcheckpoints/ugriz_lsb'
chosen_model = 'n300000_replacewzeros2/model_phase0.torch'


# important parameters:
cutout_size = 200
SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')

unions_data = pd.read_parquet('/scratch/merileo/unions/catalogs/unions.ugriz.800deg2.xmatch.parquet',
                              engine='fastparquet', columns=['ra', 'dec', 'id'])

cam_dict = {'cfis-u': {'name': 'cfis', 'band': 'u', 'vos': 'vcp -v vos:cfis/tiles_DR3/'},
            'hsc-g': {'name': 'hsc', 'band': 'g', 'vos': 'vcp -v vos:cfis/hsc/stacks2/'},
            'cfis_lsb': {'name': 'cfis_lsb', 'band': 'r', 'vos': 'vcp -v vos:cfis/tiles_LSB_DR3/'},
            'ps1-i': {'name': 'ps1', 'band': 'i', 'vos': 'vcp -L -v vos:cfis/panstarrs/DR2/skycell.'},
            'ps1-z': {'name': 'ps1', 'band': 'z', 'vos': 'vcp -L -v vos:cfis/panstarrs/DR2/skycell.'}}


rn50 = resnet50(pretrained=False)
rn50.conv1 = torch.nn.Conv2d(5, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
rn50.fc = Identity()

weights = torch.load(os.path.join(base_model_dir, chosen_model))
trunk_weights = weights["classy_state_dict"]["base_model"]["model"]["trunk"]

prefix = "_feature_blocks."
trunk_weights = {k[len(prefix):] : w for k, w in trunk_weights.items()}
rn50.load_state_dict(trunk_weights)

rn50 = rn50.cuda()
rn50.eval()

for i in range(73e6):
    ra = unions_data.iloc[i]['ra']
    dec = unions_data.iloc[i]['dec']
    cfis_id = unions_data.iloc[i]['id']
    tile = coord_to_tile(ra, dec)

    skycell = tile[:3]
    print(tile)

    # Download each camera's tile
    success = download_tile_for_each_cam(tile, cam_dict, download_dir)
    if not success:
        continue

    # Rename the PS1 files
    rename_ps1(SLURM_TMPDIR)

    # Make cutouts for this tile with each cam
    make_cutouts_for_each_cam(SLURM_TMPDIR, cam_dict, ra, dec, tile)

    # Stack all the cutouts along the filter dimension
    stacked_cutout_array = np.stack([cam_dict[cam]['cutouts'] for cam in cam_dict.keys()], 1)


#save_path = os.path.join(scratch, 'unions_simCLR_CFISr_HSCg_PS1i.h5')

if os.path.exists(save_path):
    with h5py.File(save_path, 'r') as hf:
        cfis_ids_saved = hf['cfis_id'][:]
else:
    cfis_ids_saved = []

for j, filepath in enumerate(all_files):
    output_representations_list = []
    print('{} of {} done'.format(j, len(all_files)))
    with h5py.File(filepath, 'r') as f:
        cfis_ids = f['cfis_id'][:]
        if cfis_ids[0] in cfis_ids_saved:
            print('Already processed, moving on...')
            continue
        images = f['images'][:]
        tiles = f['tile'][:]
        decs = f['dec'][:]
        ras = f['ra'][:]
        images = np.float32(images)


    images = np.swapaxes(images, 1, 3)
    images = np.asarray(([cropper(totensor(image)).detach().numpy() for image in images]))
    for i in range(0, len(images), 10):
        reps = rn50(torch.from_numpy(images[i:i + 10]).cuda()).detach().cpu().numpy()
        output_representations_list.extend(reps)

    output_representations_list = np.asarray(output_representations_list)
    if os.path.exists(save_path):
        hf = h5py.File(save_path, 'a')
    else:
        hf = h5py.File(save_path, 'w')

    if 'simclr_reps' not in hf.keys():
        print('Creating simclr dataset')
        maxshape = (None, output_representations_list.shape[1])
        hf.create_dataset('simclr_reps', data=output_representations_list, maxshape=maxshape)
    else:
        hf['simclr_reps'].resize((hf['simclr_reps'].shape[0]) + output_representations_list.shape[0], axis=0)
        hf['simclr_reps'][-output_representations_list.shape[0]:] = output_representations_list

    dt = h5py.special_dtype(vlen=str)
    tiles = np.array(tiles[:], dtype=dt)

    if 'cfis_id' not in hf.keys():
        print('Creating {} dataset'.format('cfis_id'))
        maxshape = (None,)
        hf.create_dataset('cfis_id', data=cfis_ids, maxshape=maxshape)
    else:
        hf['cfis_id'].resize((hf['cfis_id'].shape[0]) + np.shape(cfis_ids)[0], axis=0)
        hf['cfis_id'][-np.shape(cfis_ids)[0]:] = cfis_ids

    if 'dec' not in hf.keys():
        print('Creating {} dataset'.format('dec'))
        maxshape = (None,)
        hf.create_dataset('dec', data=decs, maxshape=maxshape)
    else:
        hf['dec'].resize((hf['dec'].shape[0]) + np.shape(decs)[0], axis=0)
        hf['dec'][-np.shape(decs)[0]:] = decs

    if 'ra' not in hf.keys():
        print('Creating {} dataset'.format('ra'))
        maxshape = (None,)
        hf.create_dataset('ra', data=ras, maxshape=maxshape)
    else:
        hf['ra'].resize((hf['ra'].shape[0]) + np.shape(ras)[0], axis=0)
        hf['ra'][-np.shape(ras)[0]:] = ras

    if 'tile' not in hf.keys():
        print('Creating {} dataset'.format('tile'))
        maxshape = (None,)
        hf.create_dataset('tile', data=tiles, maxshape=maxshape)
    else:
        hf['tile'].resize((hf['tile'].shape[0]) + np.shape(tiles)[0], axis=0)
        hf['tile'][-np.shape(tiles)[0]:] = tiles

    hf.close()