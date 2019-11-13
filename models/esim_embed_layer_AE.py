"""
##################################################
##################################################
## This file implements an auto-encoder using   ##
## ESIM's first BiLSTM layer as the encoder and ##
## a second single layer BiLSTM as the decoder. ##
## The idea was to test whether pre-training    ##
## ESIM's first BiLSTM, which essentially just  ##
## projects ELMo embeddings to a lower          ##
## dimensional manifold, would yield better     ##
## results, as we could use the entire dataset  ##
## to learn the representation, instead of      ##
## using just the data corresponding to the     ##
## Any-Shot learning setting being considered.  ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import torch.nn as nn
from   torch.nn.utils.rnn import PackedSequence
#TODO: Remove "import os" if we can use trainer_helpers to load a model. Do we want to, though? Then the model is
#TODO: dependant on the trainer. Write model saving/loading function in the general trainers?
import os


# *** Own modules imports. *** #

from   models.mlp import MLP
import helpers.trainer.helpers as trainer_helpers





#################
##### CLASS #####
#################

class ESIM_Embed_Layer_AE(nn.Module):
    """
    This class implements a simple seq2seq auto-encoder model, which allows learning the ELMo embeddings lower
    dimensional representation (i.e. after passing through ESIM's first BiLSTM stage) using the full dataset, instead
    of using simply the dataset associated with a specific Any-Shot Learning experiment.
    """

    def __init__(self, word_embedding_size=None, first_bilstm_hidden_size=None, first_bilst_num_layers=1,
                 load_path=None, epoch_or_best=None, loaded_arch=False, device=torch.device('cpu'), **params_dict):
        """
        Instantiates an ESIM_Embed_Layer_AE model object.

        :param word_embedding_size     : The size of the word embeddings.

        :param first_bilstm_hidden_size: The hidden size of the BiLSTM. The name is to keep it in line with esim_sts.

        :param first_bilst_num_layers  : The number of BiLSTM layers. The name is to keep it in line with esim_sts.

        :param load_path               : The path to a previously saved state.

        :param epoch_or_best           : A number indicating which epoch to load ('-1' for the best performing epoch).

        :param loaded_arch             : A boolean that allows for a simple previous state loading pattern.

        :param device                  : The device (CPU, GPU-n) on which the model is meant to be run.

        :param params_dict             : Allows passing some of the previous parameters as a dictionary that gets
                                         unpacked.
        """

        super(ESIM_Embed_Layer_AE, self).__init__()

        # Input parameters.
        self._load_path     = load_path
        self._epoch_or_best = epoch_or_best


        # Simple loading pattern that allows for a parameter dictionary to be read directly from file.
        if (not loaded_arch and load_path is not None):
            with open(load_path + "architecture.act", 'rb') as f:
                self.__init__(load_path=load_path, epoch_or_best=epoch_or_best, loaded_arch=True,
                              device=device, **torch.load(f))
        else:
            self._word_embedding_size      = word_embedding_size
            self._first_bilstm_hidden_size = first_bilstm_hidden_size
            self._first_bilst_num_layers   = first_bilst_num_layers
            if (loaded_arch):
                return

        self._device = device


        ###########
        # ENCODER #
        ###########

        # Input --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
        # Output --> Packed_Sequence([Batch_Size, Max_Seq_Len, first_bilstm_hidden_size])
        self._encoder = nn.LSTM(self._word_embedding_size, self._first_bilstm_hidden_size,
                                self._first_bilst_num_layers, bidirectional=True, batch_first=True)


        ###########
        # DECODER #
        ###########

        # Input --> Packed_Sequence([Batch_Size, Max_Batch_Seq_Len, first_bilstm_hidden_size])
        # Output --> Packed_Sequence([Batch_Size, Max_Batch_Seq_Len, word_embedding_size])
        self._decoder = nn.LSTM(2*self._first_bilstm_hidden_size, self._word_embedding_size,
                                self._first_bilst_num_layers, bidirectional=True, batch_first=True)


        #*** 1 layer (0 hidden) MLP that reduces the dimensionality of the decoder output back to the original size ***#
        # Input --> [sum(Batch_Seq_Lens), 2*first_bilstm_hidden_size]
        # Output --> [sum(Batch_Seq_Lens), first_bilstm_hidden_size]
        self._dim_reduction_mlp = MLP([2*self._word_embedding_size, self._word_embedding_size],
                                      [nn.ReLU()], device=self._device)


        #################
        # MISCELLANEOUS #
        #################

        # Load previous state, if adequate.
        previous_weights = trainer_helpers.load_checkpoint_state(self._load_path, "weights", self._epoch_or_best,
                                                                 device=self._device)

        if (previous_weights is not None):
            self.load_state_dict(previous_weights)

        self.to(device=self._device)





    def forward(self, batch, data_loader_type=None):
        """
        Performs the network's forward pass.


        :param x_batch         : The input batch to the network.

        :param data_loader_type: Allows the identification of whether the model is being trained or simply evaluated,
                                 which can be important due to different behaviour on either stage.


        :return: The output of the network, for the input batch.
        """

        # TODO: This is a workaround to solve the issue of when the dataloader's 'pin_memory=True' PackedSequences
        # TODO: get 'destroyed' and only the 'data' and 'batch_sizes' tensors remain. THIS IS A PyTorch BUG.
        if (str(self._device)[:3] != 'cpu'):
            sentences = PackedSequence(batch[0], batch[1].to(device='cpu'))
        else:
            sentences = batch


        ###########
        # ENCODER #
        ###########

        # Input --> Packed_Sequence([Batch_Size, Max_Batch_Seq_Len, word_embedding_size])
        # Output --> Packed_Sequence([Batch_Size, Max_Batch_Seq_Len, first_bilstm_hidden_size])
        embeded_sentences, (_, _) = self._encoder(sentences)


        ###########
        # DECODER #
        ###########

        # Input --> Packed_Sequence([Batch_Size, Max_Batch_Seq_Len, word_embedding_size])
        # Output --> Packed_Sequence([Batch_Size, Max_Batch_Seq_Len, first_bilstm_hidden_size])
        reconstructed_sentences, (_, _) = self._decoder(embeded_sentences)
        reconstructed_embeddings = self._dim_reduction_mlp(reconstructed_sentences.data)[0]


        return (reconstructed_embeddings, )




    def save(self, current_epoch, path, file_type, save_weights=True):
        """
        Saves the network to files.


        :param current_epoch: Identifies the current epoch, which will be used to identify the saved files.

        :param path         : The path to the directory where the state will be saved as files.

        :param file_type    : Identifies whether the save pertains a regular checkpoint, or the best performing model,
                              so far.

        :param save_weights : A boolean indicating whether the weights should be saved or not.


        :return: Nothing
        """

        if (current_epoch == 0):
            # Saves the architecture
            architecture = {
                "word_embedding_size"                   : self._word_embedding_size,
                "first_bilstm_hidden_size"              : self._first_bilstm_hidden_size,
                "first_bilst_num_layers"                : self._first_bilst_num_layers,
            }
            torch.save(architecture, path + "architecture.act")

        # Saves the weights for the current_epoch, associated either with a checkpoint or the best performing model.
        if (save_weights):
            torch.save(self.state_dict(), path + "weights" + file_type)




    @property
    def word_embedding_size(self):
        return self._word_embedding_size

    @property
    def first_bilstm_hidden_size(self):
        return self._first_bilstm_hidden_size

    @property
    def first_bilst_num_layers(self):
        return self._first_bilst_num_layers

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def dim_reduction_mlp(self):
        return self._dim_reduction_mlp