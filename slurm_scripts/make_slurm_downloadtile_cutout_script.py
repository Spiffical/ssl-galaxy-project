import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./jobs/todo/submit_job.sh',
                    help='Path of shell script')
parser.add_argument('--catalog', type=str,
                    default='/scratch/merileo/unions/catalogs/unions.matchingbands.ugriz.parquet',
                    help='Path of catalog')
parser.add_argument('--num_tiles_per_job', type=int,
                    default=10,
                    help='Number of tiles to be processed by each job')


args = parser.parse_args()
output_job_path = args.output_path
catalog_path = args.catalog
num_tiles_per_job = args.num_tiles_per_job


# Load data that has already been cleaned to ensure tiles in common for ugriz
print('Loading in unions data...')
unions_data = pd.read_parquet(catalog_path, engine='fastparquet')

# Collect tile names
all_tiles = np.unique(unions_data.tile)


def write_script(output_path, tiles):
    if not output_path.endswith('.sh'):
        output_path += '.sh'

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load python/3.7\n')
        writer.write('source $HOME/vissl-env/bin/activate')
        writer.write('\n')
        writer.write('python /scratch/merileo/unions/data_prep/download_and_make_cutouts.py ')
        writer.write('--tiles')
        for tile in tiles:
            writer.write(' %s' % tile)


for i in range(0, len(all_tiles), num_tiles_per_job):
    selected_tiles = all_tiles[i:i + num_tiles_per_job]
    output_path_temp = output_job_path[:-3] + '_{}.sh'.format(i)
    write_script(output_path_temp, selected_tiles)