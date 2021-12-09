import os
import glob
import numpy as np
import h5py

cutout_save_folder = '/scratch/merileo/unions/cutouts'
h5_basefilename = 'cutout_stacks_CFISr_PS1i_HSCg_200x200'

all_cutout_files = glob.glob('/scratch/merileo/unions/cutouts/all_tile_cutouts/*.*.h5')

num_cutout_stacks_per_file = 10000

collected_images = []
collected_cfis_ids = []
collected_ra = []
collected_dec = []
collected_tiles = []

count = 0
save_filenumber = 0
for i in range(len(all_cutout_files)):
    with h5py.File(all_cutout_files[i]) as f:
        cfis_ids = f['cfis_id'][:]
        images = f['images'][:]
        ra = f['ra'][:]
        dec = f['dec'][:]
        tiles = f['tile'][:]
        
        collected_images.extend(images)
        collected_cfis_ids.extend(cfis_ids)
        collected_ra.extend(ra)
        collected_dec.extend(dec)
        collected_tiles.extend(tiles)
        
        count += len(ra)
        
    while count >= num_cutout_stacks_per_file:
        save_path = os.path.join(cutout_save_folder, h5_basefilename + 'new_{}.h5'.format(save_filenumber))
        print('Saving file: {}'.format(save_path))
        dt = h5py.special_dtype(vlen=str)
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('images', data=np.asarray(collected_images[:num_cutout_stacks_per_file]))
            hf.create_dataset('tile', data=np.asarray(collected_tiles[:num_cutout_stacks_per_file], dtype=dt))
            hf.create_dataset('cfis_id', data=np.asarray(collected_cfis_ids[:num_cutout_stacks_per_file]))
            hf.create_dataset('ra', data=np.asarray(collected_ra[:num_cutout_stacks_per_file]))
            hf.create_dataset('dec', data=np.asarray(collected_dec[:num_cutout_stacks_per_file]))

        count = count - num_cutout_stacks_per_file
        leftover_images = collected_images[num_cutout_stacks_per_file:]
        leftover_tile_ids = collected_tiles[num_cutout_stacks_per_file:]
        leftover_cfis_ids = collected_cfis_ids[num_cutout_stacks_per_file:]
        leftover_ra = collected_ra[num_cutout_stacks_per_file:]
        leftover_dec = collected_dec[num_cutout_stacks_per_file:]
        if count > 1:
            collected_images = leftover_images
            collected_tiles = leftover_tile_ids
            collected_cfis_ids = leftover_cfis_ids
            collected_ra = leftover_ra
            collected_dec = leftover_dec
        else:
            stacked_cutout_array = []
            tile_id_array = []
            cfis_id_array = []
            ra_array = []
            dec_array = []
            
        save_filenumber+=1
