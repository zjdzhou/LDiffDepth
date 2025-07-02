# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import random

import numpy as np
import torch

# from trainer import Trainer
from trainer import Trainer
from options import MonodepthOptions


def set_seed(seed):
    if seed is None:
        seed = 1
    # print("Random Seed: {}".format(seed))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


options = MonodepthOptions()
opts = options.parse()
set_seed(opts.random_seed)


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()

