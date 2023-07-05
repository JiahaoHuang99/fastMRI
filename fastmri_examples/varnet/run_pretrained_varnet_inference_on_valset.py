"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os
import os.path
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm
import h5py
import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.models import VarNet

VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        # print(f'create {path}')
    else:
        pass
        # print(f'{path} already exists.')


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


def run_varnet_model(batch, model, device):
    crop_size = batch.crop_size

    output = model(batch.masked_kspace.to(device), batch.mask.to(device)).cpu()

    # detect FLAIR 203
    if output.shape[-1] < crop_size[1]:
        crop_size = (output.shape[-1], output.shape[-1])

    output = T.center_crop(output, crop_size)[0]

    return output, int(batch.slice_num[0]), batch.fname[0]


def get_sens_map(batch, model, device):
    crop_size = batch.crop_size

    output = model(batch.masked_kspace.to(device), batch.mask.to(device)).cpu()
    sens_map = model.sens_net(batch.masked_kspace.to(device), batch.mask.to(device), None).cpu()
    sens_map = torch.complex(sens_map[..., 0], sens_map[..., 1])

    # detect FLAIR 203
    if output.shape[-1] < crop_size[1]:
        crop_size = (output.shape[-1], output.shape[-1])

    output = T.center_crop(output, crop_size)[0]
    sens_map = T.center_crop(sens_map, crop_size)[0]

    return sens_map, int(batch.slice_num[0]), batch.fname[0]


def run_inference(challenge, state_dict_file, data_path, output_path, device, mask):
    model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
    # download the state_dict if we don't have it
    if state_dict_file is None:
        if not Path(MODEL_FNAMES[challenge]).exists():
            url_root = VARNET_FOLDER
            download_model(url_root + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])

        state_dict_file = MODEL_FNAMES[challenge]

    model.load_state_dict(torch.load(state_dict_file))
    model = model.eval()

    # data loader setup
    data_transform = T.VarNetDataTransformM2(mask=mask)  # need to modify here
    dataset = SliceDataset(root=data_path,
                           transform=data_transform,
                           challenge="multicoil",
                           )

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    # run the model
    start_time = time.perf_counter()
    outputs = defaultdict(list)
    sens_maps = defaultdict(list)
    gts = defaultdict(list)
    zfs = defaultdict(list)
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_varnet_model(batch, model, device)
            # sens_map, slice_num, fname = get_sens_map(batch, model, device)  # FIXME: DELETE
            gt = batch.gt[0]
            zf = batch.zf[0]
        gts[fname].append((slice_num, gt))
        zfs[fname].append((slice_num, zf))
        outputs[fname].append((slice_num, output))
        # sens_maps[fname].append((slice_num, sens_map))  # FIXME: DELETE

    # save outputs
    import matplotlib.pyplot as plt
    for fname in outputs:
        assert len(outputs[fname]) == 20
        for idx, (idx_name, image_slice) in enumerate(outputs[fname]):
            img_info = '{}_{:03d}'.format(fname[:-3], idx_name)
            image_slice = image_slice.numpy()
            mkdir(os.path.join(output_path, 'recon'))
            plt.imsave(os.path.join(output_path, 'recon', '{}.png'.format(img_info)), np.abs(image_slice), cmap='gray')

            _, gt = gts[fname][idx]
            gt = gt.numpy()
            mkdir(os.path.join(output_path, 'gt'))
            plt.imsave(os.path.join(output_path, 'gt', '{}.png'.format(img_info)), np.abs(gt), cmap='gray')

            _, zf = zfs[fname][idx]
            zf = zf.numpy()
            mkdir(os.path.join(output_path, 'zf'))
            plt.imsave(os.path.join(output_path, 'zf', '{}.png'.format(img_info)), np.abs(zf), cmap='gray')

            # FIXME: DELETE
            # _, sens_map = sens_maps[fname][idx]
            # sens_map = sens_map.numpy()
            # mkdir(os.path.join(output_path, 'sens_map'))
            # for idx_c in range(15):
            #     plt.imsave(os.path.join(output_path, 'sens_map', '{}_{}.png'.format(img_info, idx_c)), np.abs(sens_map[idx_c]), cmap='gray')

            mkdir(os.path.join(output_path, 'h5'))
            with h5py.File(os.path.join(output_path, 'h5', '{}.h5'.format(img_info)), "w") as file:
                file['recon'] = image_slice
                file['gt'] = gt
                file['zf'] = zf
                file.attrs['img_info'] = img_info

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":

    import pathlib

    # mask_name = 'fMRI_Ran_AF4_CF0.08_PE320'
    # mask_name = 'fMRI_Ran_AF8_CF0.04_PE320'
    mask_name = 'fMRI_Ran_AF16_CF0.02_PE320'

    from select_mask import define_Mask
    # from fastmri_examples.varnet.select_mask import define_mask

    opt = {}
    opt['mask'] = mask_name
    # get mask
    if 'fMRI' in opt['mask']:
        mask = define_Mask(opt)
        mask = mask[:, np.newaxis]
    else:
        raise NotImplementedError

    data_path = pathlib.Path(f'/media/NAS03/fastMRI/knee/d.2.0.complex.mc.ori640.VarNet/val_mini/PD/h5raw')
    task_name = f'varnet_knee_mc.d.2.0.complex.mc.ori640.VarNet.{mask_name}'

    output_path = pathlib.Path('/home/jh/fastMRI/jh/results/{}'.format(task_name))
    mkdir(output_path)

    run_inference(challenge='varnet_knee_mc',
                  state_dict_file=None,
                  data_path=data_path,
                  output_path=output_path,
                  device='cuda',
                  mask=mask,  # (320, 1)
                  )

