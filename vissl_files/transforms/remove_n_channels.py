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
import random


def make_list_of_channels_to_replace(n_replace, n_channels):
    list = []
    list.extend([1]*n_replace)
    list.extend([0]*(n_channels-n_replace))
    random.shuffle(list)
    return list


@register_transform("RemoveNChannels")
class RemoveNChannels(ClassyTransform):
    """
    Replace N (tensor) channels with noise or zeros.
    """

    def __init__(self, p, max_n, noise):
        """
        Args:
            prob (float): probably threshold for applying this transformation
            max_n (int): maximum number of channels to replace with noise
            noise (bool): if True, replace channel with noise, if False, replace with zeros

        """
        self.prob = p
        self.max_n = max_n
        self.noise = noise

    def __call__(self, img):

        should_remove_channels = np.random.rand() <= self.prob
        if not should_remove_channels:
            return img

        if self.noise:
            replace_channel = torch.empty(img[0].shape).normal_(mean=0, std=0.1 ** 0.5)
        else:
            replace_channel = torch.zeros(img[0].shape)

        num_channels_replace = np.random.randint(1, self.max_n+1)
        total_num_channels = img.shape[0]
        replace_channels = make_list_of_channels_to_replace(num_channels_replace, total_num_channels)

        for i in range(len(replace_channels)):
            if replace_channels[i]:
                img[i, :, :] = replace_channel

        return img


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RemoveNChannels":
        """
        Instantiates RemoveNChannels from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            RemoveNChannels instance.
        """
        prob = config.get("p", 0.5)
        max_n = config.get("max_n", 4)
        noise = config.get("noise", True)
        return cls(p=prob, max_n=max_n, noise=noise)

