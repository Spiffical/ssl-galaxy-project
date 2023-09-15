import glob
import os
import numpy as np
import pandas as pd
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

saved_cfisids_file = os.path.join(save_dir, 'saved_cfis_ids.npy')
saved_index_file = os.path.join(save_dir, 'saved_index_ids.npy')


ugriz_data = pd.read_parquet('/scratch/merileo/unions/catalogs/unions.matchingbands.ugriz.parquet',
                            engine='fastparquet')

ugriz_14_15 = ugriz_data.loc[(ugriz_data['r_cfht'] > 14) & (ugriz_data['r_cfht'].values < 15)]
ugriz_15_16 = ugriz_data.loc[(ugriz_data['r_cfht'] > 15) & (ugriz_data['r_cfht'].values < 16)]
ugriz_16_17 = ugriz_data.loc[(ugriz_data['r_cfht'] > 16) & (ugriz_data['r_cfht'].values < 17)]
ugriz_17_18 = ugriz_data.loc[(ugriz_data['r_cfht'] > 17) & (ugriz_data['r_cfht'].values < 18)]
ugriz_18_19 = ugriz_data.loc[(ugriz_data['r_cfht'] > 18) & (ugriz_data['r_cfht'].values < 19)]
ugriz_19_20 = ugriz_data.loc[(ugriz_data['r_cfht'] > 19) & (ugriz_data['r_cfht'].values < 20)]
ugriz_20_21 = ugriz_data.loc[(ugriz_data['r_cfht'] > 20) & (ugriz_data['r_cfht'].values < 21)]
ugriz_21_22 = ugriz_data.loc[(ugriz_data['r_cfht'] > 21) & (ugriz_data['r_cfht'].values < 22)]
ugriz_22_above = ugriz_data.loc[(ugriz_data['r_cfht'] > 22)]

frames = [ugriz_14_15,
          ugriz_15_16.head(124060),
          ugriz_16_17.head(124115),
          ugriz_17_18.head(124060),
          ugriz_18_19.head(124060),
          ugriz_19_20.head(124060),
          ugriz_20_21.head(124060),
          ugriz_21_22.head(124060),
          ugriz_22_above.head(124000)]

result = pd.concat(frames)

curated_ids = result['id'].values
st = set(curated_ids)

num_cutout_stacks_per_file = 10000
#save_dir = '/scratch/merileo/unions/cutouts/curated'
#h5_basefilename = 'curated_ugriz_lsb'

# saved_cfisids_file = os.path.join(save_dir, 'saved_cfis_ids.npy')
# saved_tileids_file = os.path.join(save_dir, 'saved_tile_ids.npy')

collected_images = []
collected_cfis_ids = []
collected_ra = []
collected_dec = []
collected_tiles = []
collected_idxes = []

count = 0
save_filenumber = 0

for i in range(len(all_cutout_files)):
    idx = all_cutout_files[i].split('.')[-3]
    if os.path.exists(saved_index_file):
        if idx in np.load(saved_index_file).astype(str):
            continue
    os.system('cp {} {}'.format(all_cutout_files[i], '$SLURM_TMPDIR'))
    try:
        with h5py.File(os.path.join(os.getenv('SLURM_TMPDIR'),
                                    os.path.basename(all_cutout_files[i])), 'r') as f:
            cfis_ids = f['cfis_id'][:]

            keep_indices = [j for j, e in enumerate(cfis_ids) if e in st]

            print('# of indices to keep: {}'.format(len(keep_indices)))
            if os.path.exists(saved_cfisids_file):
                saved_cfis_ids = np.load(saved_cfisids_file)
                for k, cfis_id in enumerate(cfis_ids):
                    if k in keep_indices:
                        if cfis_id not in saved_cfis_ids:
                            collected_images.append(f['images'][k])
                            collected_cfis_ids.append(cfis_id)
                            collected_ra.append(f['ra'][k])
                            collected_dec.append(f['dec'][k])
                            collected_tiles.append(f['tile'][k])
                            collected_idxes.append(idx)
                            count += 1
            else:
                if len(keep_indices) > 0:
                    collected_images.extend(f['images'][keep_indices])
                    collected_cfis_ids.extend(cfis_ids[keep_indices])
                    collected_ra.extend(f['ra'][keep_indices])
                    collected_dec.extend(f['dec'][keep_indices])
                    collected_tiles.extend(f['tile'][keep_indices])
                    collected_idxes.extend([idx]*len(keep_indices))
                    count += len(keep_indices)

    except Exception as e:
        print(e)
        print('problem with {}'.format(all_cutout_files[i]))
        continue
    finally:
        print('deleting...')
        os.system('rm {}'.format(os.path.join(os.getenv('SLURM_TMPDIR'),
                                              os.path.basename(all_cutout_files[i]))))

    while count >= num_cutout_stacks_per_file or i == len(all_cutout_files) - 1:
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
        print(str(count) + '\n\n')
        leftover_images = collected_images[num_cutout_stacks_per_file:]
        leftover_tile_ids = collected_tiles[num_cutout_stacks_per_file:]
        leftover_cfis_ids = collected_cfis_ids[num_cutout_stacks_per_file:]
        leftover_ra = collected_ra[num_cutout_stacks_per_file:]
        leftover_dec = collected_dec[num_cutout_stacks_per_file:]
        leftover_idxes = collected_idxes[num_cutout_stacks_per_file:]

        # Keep track of which file IDs have been fully processed
        processed_indx_ids = []
        for ind in np.unique(collected_idxes[:num_cutout_stacks_per_file]):
            if ind not in leftover_idxes:
                processed_indx_ids.append(ind)
        if os.path.exists(saved_index_file):
            np.save(saved_index_file,
                    np.append(np.load(saved_index_file),
                              processed_indx_ids))
        else:
            np.save(saved_index_file, processed_indx_ids)

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

        if i == len(all_cutout_files) - 1 and count <= 0:
            break
        save_filenumber += 1