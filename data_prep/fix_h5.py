import h5py
import glob

all_files = glob.glob('/scratch/merileo/unions/cutouts/ugriz_lsb/per_tile/*.h5')

for i, f in enumerate(all_files):
    print('Done {} of {} files'.format(i, len(all_files)))
    print(f)
    try:
        with h5py.File(f, 'r+') as hf:

            # Make placeholders with correct values
            hf['dec1'] = hf['ra']
            hf['ra1'] = hf['cfis_id']
            hf['cfis_ids1'] = hf['dec']

            # Delete old datasets
            del hf['ra']
            del hf['dec']
            del hf['cfis_id']

            # Remake them with correct data
            hf['ra'] = hf['ra1']
            hf['dec'] = hf['dec1']
            hf['cfis_id'] = hf['cfis_ids1']

            # Delete placeholders
            del hf['ra1']
            del hf['dec1']
            del hf['cfis_ids1']
    except Exception as e:
        print(e)
        print('Problem! Moving on...')
        continue
