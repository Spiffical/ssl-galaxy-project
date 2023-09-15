# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict

import numpy as np
import torchvision.transforms as T
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import ImageFilter, Image



@register_transform("TensorGaussianBlur")
class TensorGaussianBlur(ClassyTransform):
    """
    Apply Gaussian Blur to the (tensor) image. Take the radius and probability of
    application as the parameter.
    """

    def __init__(self, radius_min, radius_max):
        """
        Args:
            p (float): probability of applying gaussian blur to the image
            radius_min (float): blur kernel minimum radius used by ImageFilter.GaussianBlur
            radius_max (float): blur kernel maximum radius used by ImageFilter.GaussianBlur
        """
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):

        return T.GaussianBlur(kernel_size=21, sigma=(self.radius_min, self.radius_max))(img)


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TensorGaussianBlur":
        """
        Instantiates TensorGaussianBlur from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            TensorGaussianBlur instance.
        """
        #prob = config.get("p", 0.5)
        radius_min = config.get("radius_min", 0.1)
        radius_max = config.get("radius_max", 2.0)
        return cls(radius_min=radius_min, radius_max=radius_max)
