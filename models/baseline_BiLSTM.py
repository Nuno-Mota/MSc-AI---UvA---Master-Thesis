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
from   torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, PackedSequence
import h5py
#TODO: Remove "import os" if we can use trainer_helpers to load a model. Do we want to, though? Then the model is
#TODO: dependant on the trainer. Write model saving/loading function in the general trainers?
import os


# *** Own modules imports. *** #

import helpers.trainer.helpers as trainer_helpers





#################
##### CLASS #####
#################

class Baseline_BiLSTM(nn.Module):
    """
    TODO
    """

    def __init__(self, classes_descs_embed_file=None, word_embedding_size=None,
                 single_embedding_bilstm=None, first_bilstm_hidden_size=None, first_bilst_num_layers=1,
                 load_path=None, epoch_or_best=None, loaded_arch=False, device=torch.device('cpu'), **params_dict):
        """
        Instantiates an Baseline_BiLSTM model object.
        """

        super(Baseline_BiLSTM, self).__init__()

        # Input parameters.
        self._classes_descs_embed_file = classes_descs_embed_file

        self._load_path     = load_path
        self._epoch_or_best = epoch_or_best


        if (not loaded_arch and load_path is not None):
            with open(load_path + "architecture.act", 'rb') as f:
                self.__init__(classes_descs_embed_file=classes_descs_embed_file, load_path=load_path,
                              epoch_or_best=epoch_or_best, loaded_arch=True, device=device, **torch.load(f))
        else:
            self._word_embedding_size      = word_embedding_size
            self._single_embedding_bilstm  = single_embedding_bilstm
            self._first_bilstm_hidden_size = first_bilstm_hidden_size
            self._first_bilst_num_layers   = first_bilst_num_layers
            if (loaded_arch):
                return

        self._device = device


        #########################
        # Relation Descriptions #
        #########################

        # Here we get the relations' descriptions' embeddings.
        y_sentences_dataset = h5py.File(self._classes_descs_embed_file, 'r')
        self._y_sentences = [None] * (len(y_sentences_dataset.keys()) - 1)
        for key in y_sentences_dataset:
            if (key != 'sentence_to_index'):
                y = torch.from_numpy(y_sentences_dataset[key][:])
                self._y_sentences[int(key)] = torch.reshape(y.permute(1, 0, 2), (y.shape[1], y.shape[0] * y.shape[2]))
                self._y_sentences[int(key)] = self._y_sentences[int(key)].to(device=self._device)


        #########################
        # Input Embedding Layer #
        #########################

        if (self._single_embedding_bilstm):
            # Input --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
            # Output --> Packed_Sequence([Batch_Size, Max_Seq_Len, First_BiLSTM_Stage_Hidden_Size])
            self._embedding_bilstm = nn.LSTM(self._word_embedding_size, self._first_bilstm_hidden_size,
                                             self._first_bilst_num_layers, bidirectional=True, batch_first=True)

        else:
            # Input --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
            # Output --> Packed_Sequence([Batch_Size, Max_Seq_Len, First_BiLSTM_Stage_Hidden_Size])
            self._first_x_bilstm = nn.LSTM(self._word_embedding_size, self._first_bilstm_hidden_size,
                                           self._first_bilst_num_layers, bidirectional=True, batch_first=True)

            # Input --> Packed_Sequence([Num_Descriptions, Max_Description_Len, Word_Embedding_Size])
            # Output --> Packed_Sequence([Num_Descriptions, Max_Description_Len, First_BiLSTM_Stage_Hidden_Size])
            self._first_y_bilstm = nn.LSTM(self._word_embedding_size, self._first_bilstm_hidden_size,
                                           self._first_bilst_num_layers, bidirectional=True, batch_first=True)



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
            x_sentences = PackedSequence(batch[0][0], batch[0][1].to(device='cpu'))
        else:
            x_sentences = batch[0]


        #########################
        # Relation Descriptions #
        #########################

        # Extract the relation descriptions against which the sentences, 'x', in the batch will be compared against.
        rel_descs_for_batch = [self._y_sentences[idx] for idx in batch[1][0]]
        batch_rel_descs_lengths = torch.LongTensor([rel_desc.shape[0] for rel_desc in rel_descs_for_batch])

        # Order sequences in order of descending length
        batch_rel_descs_lengths, perm_idx = batch_rel_descs_lengths.sort(0, descending=True)
        relation_perm_unsort_idxs = perm_idx.sort()[1]
        self._relation_perm_unsort_idxs_batch = relation_perm_unsort_idxs

        # Stack embeddings and order them, from longest sequence to shortest sequence
        y_sentences = pad_sequence(rel_descs_for_batch, batch_first=True)[perm_idx]

        # Pack the relations, into a PackedSequence structure, for optimised RNN operations.
        y_sentences = pack_padded_sequence(y_sentences, batch_rel_descs_lengths, batch_first=True)
        self._y_sentences_batch = y_sentences



        #########################
        # Input Embedding Layer #
        #########################

        # In --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
        # Out --> Packed_Sequence([Batch_Size, Max_Seq_Len, First_BiLSTM_Stage_Hidden_Size])
        if (self._single_embedding_bilstm):
            _, (x_h_n, _) = self._embedding_bilstm(x_sentences)
        else:
            _, (x_h_n, _) = self._first_x_bilstm(x_sentences)

        x_h_n = x_h_n.permute(1, 0, 2).reshape(x_h_n.shape[1], x_h_n.shape[0] * x_h_n.shape[2])

        # In --> Packed_Sequence([Num_Descriptions, Max_Description_Len, word_embedding_size])
        # Out --> Packed_Sequence([Num_Descriptions, Max_Description_Len, First_BiLSTM_Stage_Hidden_Size])
        if (self._single_embedding_bilstm):
            _, (y_h_n, _) = self._embedding_bilstm(y_sentences)
        else:
            _, (y_h_n, _) = self._first_y_bilstm(y_sentences)
        y_h_n = y_h_n.permute(1, 0, 2).reshape(y_h_n.shape[1], y_h_n.shape[0] * y_h_n.shape[2])
        y_h_n = y_h_n[relation_perm_unsort_idxs, :]


        logits = torch.matmul(x_h_n, torch.transpose(y_h_n, 1, 0))

        # When evaluating, we want to aggregate over the multiple descriptions for each relation.
        if (data_loader_type != 'train' and batch[1][1] is not None):
            aggregated_logits = torch.max(logits[:, batch[1][1][0]], dim=1)[0].unsqueeze(1)
            for aggregate_idxs in batch[1][1][1:]:
                aggregated_logits = torch.cat((aggregated_logits,
                                               torch.max(logits[:, aggregate_idxs], dim=1)[0].unsqueeze(1)),
                                              dim=1)
            logits = aggregated_logits

        return (logits, )




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
                "single_embedding_bilstm"               : self._single_embedding_bilstm,
                "first_bilstm_hidden_size"              : self._first_bilstm_hidden_size,
                "first_bilst_num_layers"                : self._first_bilst_num_layers,
            }
            torch.save(architecture, path + "architecture.act")

        # Saves the weights for the current_epoch, associated either with a checkpoint or the best performing model.
        if (save_parameters):
            torch.save(self.state_dict(), path + "weights" + file_type)