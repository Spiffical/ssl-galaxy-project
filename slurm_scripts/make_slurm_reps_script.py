import argparse
import os
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--job_path', type=str, default='./jobs/todo/make_reps.sh',
                    help='Path where job script will go')
parser.add_argument('--script', type=str, default='/scratch/merileo/unions/make_reps_h5.py',
                    help='Python script to run')
parser.add_argument('--cutout_dir', type=str, default=os.getenv('SLURM_TMPDIR'),
                    help='Path where cutouts are stored')
parser.add_argument('--model', type=str,
                    help='Path of saved model (.torch file)')
parser.add_argument('--save_path', type=str, default='/scratch/merileo/',
                    help='Path of folder where representation files will be stored')
parser.add_argument('--catalog_path', type=str, default='',
                    help='Path of catalog to retrieve redshift information from (if desired)')

args = parser.parse_args()
job_path = args.job_path
make_reps_script = args.script
cutouts_dir = args.cutout_dir
model_path = args.model
save_path = args.save_path
catalog_path = args.catalog_path

#cutouts_to_merge = glob.glob(os.path.join(cutout_dir, '*.h5'))

def write_script(output_path, cutouts_path, save_path, model_path, make_reps_script, catalog_path,
                 indices, order_num):
    if not output_path.endswith('.sh'):
        output_path += '.sh'

    output_path = output_path[:-2] + '{}.'.format(order_num) + output_path[-2:]
    save_path = os.path.join(save_path, '{}.h5'.format(order_num))#save_path[:-2] + '{}.'.format(order_num) + save_path[-2:]

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load python/3.7\n')
        writer.write('source $HOME/vissl-env/bin/activate\n')
        writer.write("export HDF5_USE_FILE_LOCKING='FALSE'\n")
        writer.write('\n')
        #for path in cutout_paths:
        #    writer.write('cp {} {}\n'.format(path, '$SLURM_TMPDIR'))
        writer.write('python {} \\\n'.format(make_reps_script))
        #writer.write('--cutout_dir {} \\\n'.format('$SLURM_TMPDIR'))
        writer.write('--cutout_dir {} \\\n'.format(cutouts_path))
        writer.write('--save_path {} \\\n'.format(save_path))
        writer.write('--model {} \\\n'.format(model_path))
        if catalog_path:
            writer.write('--catalog_path {} \\\n'.format(catalog_path))
        writer.write('--indices')
        for ind in indices:
            writer.write(' {}'.format(ind))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

all_indices = np.arange(579)
chunked_indices = list(chunks(all_indices, 50))

if ~os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(len(chunked_indices)):
    write_script(job_path, cutouts_dir, save_path, model_path, make_reps_script, catalog_path,
                 chunked_indices[i], i)
