import os
import torch
import h5py
import glob
import argparse
import numpy as np
import torchvision.transforms as T
import pandas as pd
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights


parser = argparse.ArgumentParser()
parser.add_argument('--job_path', type=str, default='./jobs/todo/make_reps.sh',
                    help='Path where job script will go')
parser.add_argument('--cutout_dir', type=str, default=os.getenv('SLURM_TMPDIR'),
                    help='Path where cutouts are stored')
parser.add_argument('--model', type=str,
                    help='Path of saved model (.torch file)')
parser.add_argument('--save_path', type=str, default='/scratch/merileo/',
                    help='Path of file where representations will be stored')
parser.add_argument('--catalog_path', type=str, default='',
                    help='Path of catalog to retrieve redshift information from (if desired)')
parser.add_argument('-ind', '--indices', nargs='+', help='List of indices of files to process', required=True)

args = parser.parse_args()
job_path = args.job_path
cutouts_dir = args.cutout_dir
model_path = args.model
save_path = args.save_path
catalog_path = args.catalog_path
file_indices = args.indices

cropper = T.CenterCrop(size=140)
totensor = T.ToTensor()

scratch = os.getenv('SCRATCH')


all_files = np.asarray(glob.glob(os.path.join(cutouts_dir, '*.h5')))
file_num_list = np.asarray([f.split('.')[-3] for f in all_files])
indices_to_collect = [np.where(i == file_num_list)[0][0] for i in file_indices]
files_to_process = all_files[indices_to_collect]
print('Processing the following files: {}'.format(files_to_process))


cfg = [
  'config=pretrain/simclr/simclr_UNIONS_resnet_slurm.yaml',
  'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={}'.format(model_path), # Specify path for the model weights.
  'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
  'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk.
  'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
  'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
  'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]' # Extract only the res5avg features.
]

# Compose the hydra configuration.
cfg = compose_hydra_configuration(cfg)
# Convert to AttrDict. This method will also infer certain config options
# and validate the config is valid.
_, cfg = convert_to_attrdict(cfg)

model = build_model(cfg.MODEL, cfg.OPTIMIZER).to('cuda')

# Load the checkpoint weights.
weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

# Initialize the model with the simclr model weights.
init_model_from_consolidated_weights(
    config=cfg,
    model=model,
    state_dict=weights,
    state_dict_key_name="classy_state_dict",
    skip_layers=[],  # Use this if you do not want to load all layers
)

print("Weights have loaded")

if catalog_path:
    unions_data = pd.read_parquet(catalog_path, engine='fastparquet')
    z_cols = ['z_ps', 'z_err_ps', 'zspec_sdss', 'zspec_err_sdss', 'zspec_sdss_warning',
              'z_mq', 'zspec_lamost', 'zspec_err_lamost', 'z_ls', 'z_err_ls', 'zphot_ls',
              'zphot_err_ls', 'zspec_ls']


def extract_features(img, trained_model):
    x = cropper(totensor(img)).float().unsqueeze(dim=0)
    x = torch.transpose(x, 1, 2)
    x = trained_model(x.cuda())[0].squeeze().detach().cpu().numpy()

    return x

if os.path.exists(save_path):
    with h5py.File(save_path, 'r') as hf:
        cfis_ids_saved = hf['cfis_id'][:]
else:
    cfis_ids_saved = []

for j, filepath in enumerate(files_to_process):
    output_representations_list = []
    print('{} of {} done'.format(j, len(files_to_process)))
    with h5py.File(filepath, 'r') as f:
        cfis_ids = f['cfis_id'][:]
        if cfis_ids[0] in cfis_ids_saved:
            print('Already processed, moving on...')
            continue
        tiles = f['tile'][:]
        decs = f['dec'][:]
        ras = f['ra'][:]

        for i in range(0, len(tiles)):
            rep = extract_features(f['images'][i], model)
            output_representations_list.append(rep)
        output_representations_list = np.asarray(output_representations_list)

    if catalog_path:
        print('Collecting redshift data...')
        # Collect all rows with ids that match the loaded cfis_ids, then change the order
        # of the rows so it matches the order of cfis_ids
        unions_cut = unions_data.query('id in @cfis_ids')
        unions_cut = unions_cut.set_index('id').reindex(cfis_ids).reset_index()
        # Load z data
        z_dic = {}
        for z_col in z_cols:
            z_data = unions_cut[z_col].values
            print('{}: {}'.format(z_col, z_data))
            z_dic[z_col] = z_data

    if os.path.exists(save_path):
        hf = h5py.File(save_path, 'a')
    else:
        hf = h5py.File(save_path, 'w')

    if catalog_path:
        for z_col in z_cols:
            data_z = z_dic[z_col].astype(np.float64)
            if z_col not in hf.keys():
                print('Creating {} dataset'.format(z_col))
                maxshape = (None,)
                hf.create_dataset(z_col, data=data_z, maxshape=maxshape)
            else:
                hf[z_col].resize((hf[z_col].shape[0]) + np.shape(data_z)[0], axis=0)
                hf[z_col][-np.shape(data_z)[0]:] = data_z

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