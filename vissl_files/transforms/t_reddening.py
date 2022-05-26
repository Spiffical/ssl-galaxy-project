# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from scipy.stats import lognorm


@register_transform("TensorReddening")
class TensorReddening(ClassyTransform):
    """
    Add reddening to the (tensor) image. A log-normal distribution was fit to the data with
    scipy.stats.lognorm.fit to get the shape, loc, and scale, parameters. We randomly sample from
    this distribution
    """

    def __init__(self, shape, loc, scale):
        """
        Args:
            shape (float): log-normal shape parameter
            loc (float): log-normal loc parameter
            scale (float): log-normal scale parameter
        """
        self.shape = shape
        self.loc = loc
        self.scale = scale

    def __call__(self, img):

        sample_ebv = torch.tensor(lognorm.rvs(self.shape, self.loc, self.scale, size=1)[0])
        #new_ebv = torch.FloatTensor(1).uniform_(0, self.max_ebv)
        # Taking conversion factors of sdss-u, sdss-g, sdss-r, ps-i, ps-z from:
        # https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103#apj398709t6
        conversion_factor = torch.tensor([4.239, 3.303, 2.285, 1.682, 1.322])
        reddening_factor = 10. ** (-conversion_factor * sample_ebv / 2.5)
        img = img * reddening_factor[:, None, None]
        return img

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TensorReddening":
        """
        Instantiates TensorReddening from configuration.

        Args:
            config (Dict): arguments for the transform

        Returns:
            TensorReddening instance.
        """
        shape = config.get("shape", 0.39)
        loc = config.get("shape", -0.02)
        scale = config.get("shape", 0.067)
        return cls(shape=shape, loc=loc, scale=scale)
