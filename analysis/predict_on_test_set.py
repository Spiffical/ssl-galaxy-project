import torch
import os
import h5py
import glob
import argparse
import numpy as np
from torchvision.models import resnet50
from torch.nn import Identity
import torchvision.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='/scratch/merileo/visslcheckpoints/n500000/',
                    help='Path of trained model')
parser.add_argument('--model_name', type=str, default='model_phase50.torch',
                    help='Path of trained model')
parser.add_argument('--test_dir', type=str, default='/scratch/merileo/unions/cutouts/ugriz_lsb/per_tile/',
                    help='Path of test data')
parser.add_argument('--save_path', type=str, default='/scratch/merileo/unions/results/simCLRreps.ugriz.lsb.h5',
                    help='Path where results will be saved')

args = parser.parse_args()
checkpoint_dir = args.checkpoint_dir
model_name = args.model_name
test_dir = args.test_dir
save_path = args.save_path

cropper = T.CenterCrop(size=140)
totensor = T.ToTensor()

# Get file names
train_files = np.load(os.path.join(checkpoint_dir, 'training_files_used.npy'))
test_files_temp = glob.glob(os.path.join(test_dir, '*.h5'))

# Collect only the test files that don't appear in the training set
test_files = []
for tf in test_files_temp:
    if tf not in train_files:
        test_files.append(tf)

# Instantiate resnet model
rn50 = resnet50(pretrained=False)
rn50.fc = Identity()

# Load in saved weights
model_path = os.path.join(checkpoint_dir, model_name)
weights = torch.load(model_path)
trunk_weights = weights["classy_state_dict"]["base_model"]["model"]["trunk"]
prefix = "_feature_blocks."
trunk_weights = {k[len(prefix):] : w for k, w in trunk_weights.items()}
rn50.load_state_dict(trunk_weights)
rn50 = rn50.cuda()
rn50.eval()  # This command must be executed before predictions are made

cfis_id_list = []
ra_list = []
dec_list = []
filenum_list = []
filename_list = []
indx_list = []
output_representations_list = []

if os.path.exists(save_path):
    with h5py.File(save_path, 'r') as hf:
        cfis_ids_saved = hf['cfis_id'][:]
else:
    cfis_ids_saved = []

for j, filepath in enumerate(test_files):
    output_representations_list = []
    print('{} of {} done'.format(j, len(all_files)))
    with h5py.File(filepath, 'r') as f:
        cfis_ids = f['cfis_id'][:]
        if cfis_ids[0] in cfis_ids_saved:
            print('Already processed, moving on...')
            continue
        images = f['images'][:]
        decs = f['dec'][:]
        ras = f['ra'][:]
        images = np.float32(images)

    images = np.swapaxes(images, 1, 3)
    images = np.asarray(([cropper(totensor(image)).detach().numpy() for image in images]))
    for i in range(0, len(images), 10):
        reps = rn50(torch.from_numpy(images[i:i + 10]).cuda()).detach().cpu().numpy()
        output_representations_list.extend(reps)
        filename_list.extend([os.path.basename(filepath)] * len(reps))
        indx_list.extend(np.arange(i, i + len(reps)))

    output_representations_list = np.asarray(output_representations_list)

    if os.path.exists(save_path):
        hf = h5py.File(save_path, 'a')
    else:
        hf = h5py.File(save_path, 'w')

    dt = h5py.special_dtype(vlen=str)
    tiles = np.array(tiles[:], dtype=dt)

    data_to_save = {'simclr_reps': output_representations_list,
                    'cfis_id': cfis_ids,
                    'dec': decs,
                    'ra': ras,
                    'tile': tiles,
                    'filename': filename_list,
                    'indx': indx_list}

    for key in data_to_save:
        data = data_to_save[key]

        if len(np.shape(data)) > 1:
            maxshape = (None, data.shape[1])
        else:
            maxshape = (None, )

        if key not in hf.keys():
            print('Creating {} dataset'.format(key))
            hf.create_dataset(key, data=data, maxshape=maxshape)
        else:
            hf[key].resize((hf[key].shape[0]) + np.shape(data)[0], axis=0)
            hf[key][-np.shape(data)[0]:] = data

    hf.close()


