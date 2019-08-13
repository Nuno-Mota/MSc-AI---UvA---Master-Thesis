"""
##################################################
##################################################
## TODO ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import torch.nn as nn
from   torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, PackedSequence
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
    TODO
    """

    def __init__(self, word_embedding_size=None, first_bilstm_hidden_size=None, first_bilst_num_layers=1,
                 load_path=None, epoch_or_best=None, loaded_arch=False, device=torch.device('cpu'), **params_dict):
        """
        Instantiates an ESIM_Embed_Layer_AE model object.
        """

        super(ESIM_Embed_Layer_AE, self).__init__()

        # Input parameters.
        self._load_path     = load_path
        self._epoch_or_best = epoch_or_best


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


        # Encoder

        # Input --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
        # Output --> Packed_Sequence([Batch_Size, Max_Seq_Len, First_BiLSTM_Stage_Hidden_Size])
        self._encoder = nn.LSTM(self._word_embedding_size, self._first_bilstm_hidden_size,
                                self._first_bilst_num_layers, bidirectional=True, batch_first=True)


        # Decoder

        # Input --> Packed_Sequence([Batch_Size, Max_Seq_Len, First_BiLSTM_Stage_Hidden_Size])
        # Output --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
        self._decoder = nn.LSTM(2*self._first_bilstm_hidden_size, self._word_embedding_size,
                                self._first_bilst_num_layers, bidirectional=True, batch_first=True)


        ##########################################################################################################
        # 1 layer (0 hidden) MLP that reduces the dimensionality of the decoder output back to the original size #
        ##########################################################################################################
        # Input --> [sum(seq_lengths), 2*First_BiLSTM_Stage_Hidden_Size]
        # Output --> [sum(seq_lengths), First_BiLSTM_Stage_Hidden_Size]
        self._dim_reduction_mlp = MLP([2*self._word_embedding_size, self._word_embedding_size],
                                      [nn.ReLU()], device=self._device)



        # Load previous state, if adequate.
        previous_weights = trainer_helpers.load_checkpoint_state(self._load_path, "weights", self._epoch_or_best,
                                                                 device=self._device)

        if (previous_weights is not None):
            self.load_state_dict(previous_weights)

        self.to(device=self._device)





    def forward(self, batch, data_loader_type=None):
        """
        Performs the network's forward pass.


        :param x_batch: The input batch to the network.
        TODO: Add missing parameters.


        :return: The output of the network, for the input batch.
        """

        # TODO: This is a workaround to solve the issue that when the dataloader's 'pin_memory=True' PackedSequences
        # TODO: get 'destroyed' and only the 'data' and 'batch_sizes' tensors remain.
        if (str(self._device)[:3] != 'cpu'):
            sentences = PackedSequence(batch[0], batch[1].to(device='cpu'))
        else:
            sentences = batch


        ################
        # Forward Pass #
        ################

        # Encoder.

        # In --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
        # Out --> Packed_Sequence([Batch_Size, Max_Seq_Len, First_BiLSTM_Stage_Hidden_Size])
        embeded_sentences, (_, _) = self._encoder(sentences)



        # Decoder.

        # In --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
        # Out --> Packed_Sequence([Batch_Size, Max_Seq_Len, First_BiLSTM_Stage_Hidden_Size])
        reconstructed_sentences, (_, _) = self._decoder(embeded_sentences)
        reconstructed_embeddings = self._dim_reduction_mlp(reconstructed_sentences.data)[0]


        return (reconstructed_embeddings, )




    def save(self, current_epoch, path, file_type, save_parameters=True):
        """
        Saves the network to files.TODO: Missing parameters.


        :param current_epoch: Identifies the current epoch, which will be used to identify the saved files.

        :param path         : The path to the directory where the thework will be saved as files.

        :param file_type    : Identifies whether the save pertains a regular checkpoint, or the best performing model,
                              so far.


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
        if (save_parameters):
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