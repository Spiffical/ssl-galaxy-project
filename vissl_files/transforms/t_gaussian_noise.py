# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict

import numpy as np
import torchvision.transforms as T
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import ImageFilter, Image



@register_transform("TensorGaussianNoise")
class TensorGaussianNoise(ClassyTransform):
    """
    Apply Gaussian Noise to the (tensor) image.
    """

    def __init__(self, p, noise_scale_cfis_u, noise_scale_hsc_g,
                 noise_scale_cfis_lsb_r, noise_scale_ps1_i, noise_scale_ps1_z):
        """
        Args:
            prob (float): probably threshold for applying this transformation
            noise_scale_cfis_u (float) MAD of cfis_u images
            noise_scale_hsc_g (float): MAD of hsc_g images
            noise_scale_cfis_r (float): MAD of cfis_lsb_r images
            noise_scale_ps1_i (float): MAD of ps1_i images
            noise_scale_ps1_z (float): MAD of ps1_z images

        """
        self.prob = p
        self.cfis_u = noise_scale_cfis_u
        self.hsc_g = noise_scale_hsc_g
        self.cfis_lsb_r = noise_scale_cfis_lsb_r
        self.ps1_i = noise_scale_ps1_i
        self.ps1_z = noise_scale_ps1_z

    def __call__(self, img):

        should_add_noise = np.random.rand() <= self.prob
        if not should_add_noise:
            return img

        scale_factor = random.uniform(1, 3)
        noise_cfis_u = torch.empty(img[0].shape).normal_(mean=0, std=(scale_factor * self.cfis_u) ** 0.5)
        noise_hsc_g = torch.empty(img[0].shape).normal_(mean=0, std=(scale_factor * self.hsc_g) ** 0.5)
        noise_cfis_lsb_r = torch.empty(img[0].shape).normal_(mean=0, std=(scale_factor*self.cfis_lsb_r) ** 0.5)
        noise_ps1_i = torch.empty(img[0].shape).normal_(mean=0, std=(scale_factor*self.ps1_i) ** 0.5)
        noise_ps1_z = torch.empty(img[0].shape).normal_(mean=0, std=(scale_factor * self.ps1_z) ** 0.5)

        img[0, :, :] += noise_cfis_u
        img[1, :, :] += noise_hsc_g
        img[2, :, :] += noise_cfis_lsb_r
        img[3, :, :] += noise_ps1_i
        img[4, :, :] += noise_ps1_z
        return img

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TensorGaussianNoise":
        """
        Instantiates TensorGaussianNoise from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            TensorGaussianNoise instance.
        """
        prob = config.get("p", 0.5)
        cfis_u = config.get("cfis_u", 2.0)
        hsc_g = config.get("hsc_g", 1.4)
        cfis_lsb_r = config.get("cfis_lsb_r", 1.6)
        ps1_i = config.get("ps1_i", 0.34)
        ps1_z = config.get("ps1_z", 0.34)
        return cls(p=prob, noise_scale_cfis_u=cfis_u, noise_scale_hsc_g=hsc_g,
                   noise_scale_cfis_lsb_r=cfis_lsb_r, noise_scale_ps1_i=ps1_i, noise_scale_ps1_z=ps1_z)

