import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./jobs/todo/submit_job.sh',
                    help='Path of shell script')
args = parser.parse_args()
output_job_path = args.output_path


# Data that has already been processed to ensure tiles in common for HSC-g, CFIS-r, and PS-i
print('Loading in unions data...')
unions_data = pd.read_csv('/scratch/merileo/unions/unions.ugriz2_matchingbands_PSiHSCgCFISr_fixed.tsv', 
                           header=0, 
                           delim_whitespace=True, 
                           dtype={'tile': str})

all_tiles = np.unique(unions_data.tile)

num_tiles_per_job = 10


def write_script(output_path, tiles):

    if not output_path.endswith('.sh'):
        output_path += '.sh'

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load python/3.7\n')
        writer.write('source $HOME/jupyter_py3/bin/activate')
        writer.write('\n')
        writer.write('python /scratch/merileo/unions/make_cutouts_cc.py ')
        writer.write('--tiles')
        for tile in tiles:
            writer.write(' %s' % tile)
            
            
for i in range(0, len(all_tiles), num_tiles_per_job):
    selected_tiles = all_tiles[i:i+num_tiles_per_job]
    output_path_temp = output_job_path[:-3] + '_{}.sh'.format(i)
    write_script(output_path_temp, selected_tiles)
