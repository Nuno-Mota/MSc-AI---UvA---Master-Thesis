"""
##################################################
##################################################
## This file contains dicts that at run time    ##
## facilitates the creation of unknown          ##
## components.                                  ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

import torch.nn as nn
import torch.optim as optim


# *** Import own functions. *** #

from helpers.trainer.metrics.accuracy import Accuracy
from helpers.trainer.metrics.f1 import F1
from helpers.trainer.metrics.macroF1 import MacroF1
from helpers.trainer.metrics.harmonicMacroF1 import HarmonicMacroF1
from helpers.trainer.losses.cross_entropy import CrossEntropyLoss
from helpers.trainer.losses.bag_of_words import BagOfWordsLogLoss
from helpers.trainer.losses.re_bow_loss import RE_BoW_LOSS
from helpers.trainer.losses.mse import MeanSquaredErrorLoss
from models.mlp import MLP
from models.baseline_BiLSTM import Baseline_BiLSTM
from models.esim_sts import ESIM_StS
from models.re_bow_decoder import RE_BOW_DECODER
from models.esim_embed_layer_AE import ESIM_Embed_Layer_AE
from models.re_bow import RE_BOW





##########################
##### CLASS MAPPINGS #####
##########################

# *** Activation Functions *** #
AFS = {'sigmoid'   : nn.Sigmoid,   # The sigmoid activation function.
       'tanh'      : nn.Tanh,      # The hyperbolic tangent activation function.
       'relu'      : nn.ReLU,      # The rectified linear unit activation function.
       'leaky_relu': nn.LeakyReLU, # The leaky (does not map negative to 0) rectified linear unit activation function.
       'softmax'   : nn.Softmax,   # The softmax activation function.
      }



# *** Loss Functions *** #
LOSSES = {'Cross Entropy'        : CrossEntropyLoss,     # The cross entropy loss
          'Bag of Words Log Loss': BagOfWordsLogLoss,    # The bag of words loss
          're_bow_loss'          : RE_BoW_LOSS,          # The Loss used to train the RE_BOW model
          'MSE'                  : MeanSquaredErrorLoss, # The mean squared error loss
         }



# *** Optimisers *** #
OPTIMS = {'adadelta': optim.Adadelta, # The adadelta optimiser
          'adagrad' : optim.Adagrad,  # The adagrad optimiser
          'adam'    : optim.Adam,     # The adam optimiser
          'adamax'  : optim.Adamax,   # The adamax optimiser
          'RMSprop' : optim.RMSprop,  # The RMSprop optimiser
          'sgd'     : optim.SGD,      # The SGD optimiser
         }

METRICS = {
    'Accuracy'       : Accuracy,
    'F1'             : F1,
    'MacroF1'        : MacroF1,
    'HarmonicMacroF1': HarmonicMacroF1,
}

MODELS = {
    'MLP'                : MLP,
    'ESIM_StS'           : ESIM_StS,
    'Baseline_BiLSTM'    : Baseline_BiLSTM,
    'RE_BoW_DECODER'     : RE_BOW_DECODER,
    'RE_BoW'             : RE_BOW,
    'ESIM_Embed_Layer_AE': ESIM_Embed_Layer_AE,
}