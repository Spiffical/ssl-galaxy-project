import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--job_path', type=str, default='./jobs/todo/merge_script.sh',
                    help='Path where job script will go')
parser.add_argument('--cutout_dir', type=str, default=os.getenv('SLURM_TMPDIR'),
                    help='Path where cutouts to be merged are stored')
parser.add_argument('--merge_script', type=str,
                    default='/scratch/merileo/unions/data_prep/merge_cutout_files.py',
                    help='Path of merge python script')
parser.add_argument('--save_dir', type=str, default='/scratch/merileo/unions/cutouts/ugriz_lsb/merged',
                    help='Path where merged cutouts will be saved')
parser.add_argument('--cutout_basename', type=str, default='cutout.stacks.ugriz.lsb.200x200',
                    help='Path where merged cutouts will be saved')

args = parser.parse_args()
job_path = args.job_path
cutouts_dir = args.cutout_dir
save_dir = args.save_dir
h5_basefilename = args.cutout_basename
merge_path = args.merge_script

#cutouts_to_merge = glob.glob(os.path.join(cutout_dir, '*.h5'))


def write_script(output_path, cutouts_path, save_path, basename, merge_script):
    if not output_path.endswith('.sh'):
        output_path += '.sh'

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load python/3.7\n')
        writer.write('source $HOME/vissl-env/bin/activate\n')
        writer.write("export HDF5_USE_FILE_LOCKING='FALSE'\n")
        writer.write('\n')
        #for path in cutout_paths:
        #    writer.write('cp {} {}\n'.format(path, '$SLURM_TMPDIR'))
        writer.write('python {} \\\n'.format(merge_script))
        #writer.write('--cutout_dir {} \\\n'.format('$SLURM_TMPDIR'))
        writer.write('--cutout_dir {} \\\n'.format(cutouts_path))
        writer.write('--save_dir {} \\\n'.format(save_path))
        writer.write('--cutout_basename {} \\\n'.format(basename))


write_script(job_path, cutouts_dir, save_dir, h5_basefilename, merge_path)
