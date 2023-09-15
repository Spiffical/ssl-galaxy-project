import os
import glob
import numpy as np
import h5py
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cutout_dir', type=str, default=os.getenv('SLURM_TMPDIR'),
                    help='Path where cutouts to be merged are stored')
parser.add_argument('--save_dir', type=str, default='/scratch/merileo/unions/cutouts/ugriz_lsb/merged',
                    help='Path where merged cutouts will be saved')
parser.add_argument('--cutout_basename', type=str, default='cutout.stacks.ugriz.lsb.200x200',
                    help='Base name of all h5 files to save')

args = parser.parse_args()
cutout_dir = args.cutout_dir
save_dir = args.save_dir
h5_basefilename = args.cutout_basename

all_cutout_files = glob.glob(os.path.join(cutout_dir, '*.h5'))

num_cutout_stacks_per_file = 10000

saved_cfisids_file = os.path.join(save_dir, 'saved_cfis_ids.npy')
saved_tileids_file = os.path.join(save_dir, 'saved_tile_ids.npy')

collected_images = []
collected_cfis_ids = []
collected_ra = []
collected_dec = []
collected_tiles = []

count = 0
saved_files = glob.glob(os.path.join(save_dir, '*.h5'))
save_filenumber = 0 if len(saved_files) == 0 else len(saved_files)

for i in range(len(all_cutout_files)):
    if os.path.exists(saved_tileids_file):
        tilename = all_cutout_files[i].split('0_')[1][:-3]
        if tilename in np.load(saved_tileids_file).astype(str):
            continue
    os.system('cp {} {}'.format(all_cutout_files[i], '$SLURM_TMPDIR'))
    try:
        with h5py.File(os.path.join(os.getenv('SLURM_TMPDIR'),
                                    os.path.basename(all_cutout_files[i])), 'r') as f:
            cfis_ids = f['cfis_id'][:]
            if os.path.exists(saved_cfisids_file):
                saved_cfis_ids = np.load(saved_cfisids_file)
                for j, cfis_id in enumerate(cfis_ids):
                    if cfis_id not in saved_cfis_ids:
                        collected_images.append(f['images'][j])
                        collected_cfis_ids.append(cfis_id)
                        collected_ra.append(f['ra'][j])
                        collected_dec.append(f['dec'][j])
                        collected_tiles.append(f['tile'][j])
                        count += 1
            else:
                collected_images.extend(f['images'][:])
                collected_cfis_ids.extend(cfis_ids)
                collected_ra.extend(f['ra'][:])
                collected_dec.extend(f['dec'][:])
                collected_tiles.extend(f['tile'][:])
                count += len(f['ra'][:])

    except Exception as e:
        print(e)
        print('problem with {}'.format(all_cutout_files[i]))
        continue
    finally:
        print('deleting...')
        os.system('rm {}'.format(os.path.join(os.getenv('SLURM_TMPDIR'),
                                              os.path.basename(all_cutout_files[i]))))
        
    while count >= num_cutout_stacks_per_file or i == len(all_cutout_files)-1:
        if count >= num_cutout_stacks_per_file:
            num_cutouts_saved = num_cutout_stacks_per_file
        else:
            num_cutouts_saved = count
        save_path = os.path.join(save_dir, h5_basefilename + '.{}.{}.h5'.format(save_filenumber,
                                                                               num_cutouts_saved))
        print('Saving file: {}'.format(save_path))
        dt = h5py.special_dtype(vlen=str)
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('images', data=np.asarray(collected_images[:num_cutout_stacks_per_file]))
            hf.create_dataset('tile', data=np.asarray(collected_tiles[:num_cutout_stacks_per_file], dtype=dt))
            hf.create_dataset('cfis_id', data=np.asarray(collected_cfis_ids[:num_cutout_stacks_per_file]))
            hf.create_dataset('ra', data=np.asarray(collected_ra[:num_cutout_stacks_per_file]))
            hf.create_dataset('dec', data=np.asarray(collected_dec[:num_cutout_stacks_per_file]))

        # Keep track of which CFIS IDs have been saved already
        if os.path.exists(saved_cfisids_file):
            np.save(saved_cfisids_file,
                    np.append(np.load(saved_cfisids_file),
                              collected_cfis_ids[:num_cutout_stacks_per_file]))
        else:
            np.save(saved_cfisids_file, collected_cfis_ids[:num_cutout_stacks_per_file])
        print(count)
        count = count - num_cutout_stacks_per_file
        print(str(count)+'\n\n')
        leftover_images = collected_images[num_cutout_stacks_per_file:]
        leftover_tile_ids = collected_tiles[num_cutout_stacks_per_file:]
        leftover_cfis_ids = collected_cfis_ids[num_cutout_stacks_per_file:]
        leftover_ra = collected_ra[num_cutout_stacks_per_file:]
        leftover_dec = collected_dec[num_cutout_stacks_per_file:]

        # Keep track of which tile IDs have been fully processed
        processed_tile_ids = []
        for tile_id in np.unique(collected_tiles[:num_cutout_stacks_per_file]):
            if tile_id not in leftover_tile_ids:
                processed_tile_ids.append(tile_id)
        if os.path.exists(saved_tileids_file):
            np.save(saved_tileids_file,
                    np.append(np.load(saved_tileids_file),
                              processed_tile_ids))
        else:
            np.save(saved_tileids_file, processed_tile_ids)

        if count >= 1:
            collected_images = leftover_images
            collected_tiles = leftover_tile_ids
            collected_cfis_ids = leftover_cfis_ids
            collected_ra = leftover_ra
            collected_dec = leftover_dec
        else:
            collected_images = []
            collected_tiles = []
            collected_cfis_ids = []
            collected_ra = []
            collected_dec = []

        if i == len(all_cutout_files)-1 and count <= 0:
            break
        save_filenumber+=1
