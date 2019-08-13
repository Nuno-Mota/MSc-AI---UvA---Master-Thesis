"""
##################################################
##################################################
## This is just a wrapper for the CE loss.      ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch.nn as nn

# *** Own modules imports. *** #

# Import here!





#################
##### CLASS #####
#################

class CrossEntropyLoss(object):
    """
    Defines a wrapper around Pytorch's nn.CrossEntropyLoss() so that the appropriate parameters and labels can be
    selected.
    """

    def __init__(self, **params_dict):
        self._loss_fn = nn.CrossEntropyLoss(reduction='sum')


    def __call__(self, input, target):
        return (self._loss_fn(input[0], target[0][0]), )


    def annealing_step(self):
        pass