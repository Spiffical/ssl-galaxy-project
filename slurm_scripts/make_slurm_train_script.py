import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./jobs/todo/submit_job.sh',
                    help='Path of shell script')
parser.add_argument('--training_path', type=str, default='/scratch/merileo/unions/cutouts/train',
                    help='Path where cutout training h5 files are stored')
parser.add_argument('--env', type=str, default='vissl-env',
                    help='Name of virtual environment used')
parser.add_argument('--config', type=str, default='simclr_UNIONS_resnet_slurm',
                    help='Name of config file used with VISSL')
parser.add_argument('--checkpoint', type=str, default='/scratch/merileo/visslcheckpoints/',
                    help='Path to where model checkpoints will be saved')
parser.add_argument('--num_train', type=int, default=100000,
                    help='Number of cutouts to train on')
parser.add_argument('--num_gpu', type=int, default=4,
                    help='Number of GPUs to train on')
parser.add_argument('--num_cutouts_per_file', type=int, default=10000,
                    help='Number of cutouts saved in each h5 file')

args = parser.parse_args()

# Collect args
output_job_path = args.output_path
training_path = args.training_path
num_train = args.num_train
num_gpu = args.num_gpu
num_cutouts_per_file = args.num_cutouts_per_file
env = args.env
config = args.config
checkpoint_dir = args.checkpoint

# Define paths for VISSL wrappers
home = os.getenv('HOME')
wrapper_path = os.path.join(home, env, 'lib/python3.7/site-packages/dev/low_resource_1gpu_train_wrapper.sh')
new_wrapper_path = os.path.join(home, env, 'lib/python3.7/site-packages/dev/low_resource_1gpu_train_wrapper_slurm.sh')

# If continuing from a saved model, make sure we use the same training data.
# Look for saved training data, and collect the file paths if the file exists
train_files_path = os.path.join(checkpoint_dir, 'training_files_used.npy')
if os.path.exists(train_files_path):
    files_for_training = np.load(train_files_path)
else:
    # Determine number of files needed for training (by rounding up)
    num_files_for_training = num_train // num_cutouts_per_file + (num_train % num_cutouts_per_file > 0)

    # Collect training files
    all_train_files = glob.glob(os.path.join(training_path, '*.h5'))
    files_for_training = all_train_files[:num_files_for_training]

    # Keep a record of which files were used for training
    np.save(train_files_path, np.asarray(files_for_training))


def edit_wrapper_file(wrapper_in, wrapper_out, n_train, n_gpu, checkpoint):
    # input file
    fin = open(wrapper_in, "rt")
    # output file to write the result to
    fout = open(wrapper_out, "wt")
    # for each line in the input file
    for line in fin:
        # read replace the string and write to output file
        if 'TRAIN.DATA_LIMIT' in line:
            line_break = line.split('=')
            line_break[1] = str(n_train)
            fout.write(line.replace(line, line_break[0] + '=' + line_break[1] + ' \\\n'))
        elif 'NUM_PROC_PER_NODE' in line:
            line_break = line.split('=')
            line_break[1] = str(n_gpu)
            fout.write(line.replace(line, line_break[0] + '=' + line_break[1] + ' \\\n'))
        elif 'CHECKPOINT.DIR' in line:
            line_break = line.split('=')
            line_break[1] = str(checkpoint)
            fout.write(line.replace(line, line_break[0] + '=' + '"'+line_break[1]+'"' + ' \\\n'))
        else:
            fout.write(line)
    # close input and output files
    fin.close()
    fout.close()


def write_script(output_path, training_files, wrapper, config):
    if not output_path.endswith('.sh'):
        output_path += '.sh'

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load python/3.7\n')
        writer.write('source $HOME/vissl-env/bin/activate')
        writer.write('\n')
        for f in training_files:
            writer.write('cp {} {}\n'.format(f, '$SLURM_TMPDIR'))
        writer.write('python /home/merileo/ssl-unions/vissl_files/append_slurmtmpdir_to_json.py')
        writer.write('\nbash {} config=pretrain/simclr/{}.yaml'.format(wrapper, config))


# Execute functions
edit_wrapper_file(wrapper_path, new_wrapper_path, num_train, num_gpu, checkpoint_dir)
write_script(output_job_path, files_for_training, new_wrapper_path, config)

