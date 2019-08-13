"""
##################################################
##################################################
## This file contains the implementation of the ##
## Enhanced Sequential Inference Model, by      ##
## Chen et al. (2017), with some small          ##
## variations.                                  ##
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
import h5py
#TODO: Remove "import os" if we can use trainer_helpers to load a model. Do we want to, though? Then the model is
#TODO: dependant on the trainer. Write model saving/loading function in the general trainers?
import os


# *** Own modules imports. *** #

from   models.mlp import MLP
from models.esim_embed_layer_AE import ESIM_Embed_Layer_AE
import helpers.trainer.helpers as trainer_helpers





#################
##### CLASS #####
#################

class ESIM_StS(nn.Module):
    """
    This class implements an ESIM (Chen et al. (2017)) Natural Language Inference Model, with small variations to the
    one in the original paper. In specific we regress a score, instead of performing a classification task and we
    perform a set to set attention (i.e. we compute attention between two independent sets of sentences. For each
    sentence we compute the attention against every other sentence in the other set.).
    Throughout the model implementation, x refers to the sentences which we want to classify and y refers to labels'
    descriptions.
    """

    def __init__(self, classes_descs_embed_file=None, word_embedding_size=None, pretrained_embedding_layer_path=None,
                 single_embedding_bilstm=None, first_bilstm_hidden_size=None, post_attention_size=None,
                 second_bilstm_hidden_size=None,
                 score_mlp_post_first_layer_sizes=None, score_mlp_post_first_layer_activations=None,
                 leakyReLU_negative_slope=None, first_bilst_num_layers=1, second_bilst_num_layers=1,
                 load_path=None, epoch_or_best=None, loaded_arch=False, device=torch.device('cpu'), **params_dict):
        """
        Instantiates an ESIM_StS model object.
        """

        super(ESIM_StS, self).__init__()

        # Input parameters.
        self._classes_descs_embed_file = classes_descs_embed_file

        self._load_path     = load_path
        self._epoch_or_best = epoch_or_best


        if (not loaded_arch and load_path is not None):
            with open(load_path + "architecture.act", 'rb') as f:
                self.__init__(classes_descs_embed_file=classes_descs_embed_file, load_path=load_path,
                              epoch_or_best=epoch_or_best, loaded_arch=True, device=device, **torch.load(f))
        else:
            self._word_embedding_size                    = word_embedding_size
            self._single_embedding_bilstm                = single_embedding_bilstm
            self._first_bilstm_hidden_size               = first_bilstm_hidden_size
            self._post_attention_size                    = post_attention_size
            self._second_bilstm_hidden_size              = second_bilstm_hidden_size
            self._score_mlp_post_first_layer_sizes       = score_mlp_post_first_layer_sizes
            self._score_mlp_post_first_layer_activations = score_mlp_post_first_layer_activations
            self._leakyReLU_negative_slope               = leakyReLU_negative_slope
            self._first_bilst_num_layers                 = first_bilst_num_layers
            self._second_bilst_num_layers                = second_bilst_num_layers
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
            if (pretrained_embedding_layer_path is not None):
                model_files = os.listdir(pretrained_embedding_layer_path + os.sep + "Model" + os.sep)
                best_epoch = [int(file.split('.')[-2]) for file in model_files if file.endswith('.bst')]
                if (not bool(best_epoch)):
                    last_epoch = max([int(file.split('.')[-2]) for file in model_files if file.endswith('.ckp')])
                pretrained_model_path = pretrained_embedding_layer_path + os.sep + "Model" + os.sep
                pretrained_epoch_or_best = -1 if bool(best_epoch) else last_epoch

                self._embedding_bilstm = ESIM_Embed_Layer_AE(load_path=pretrained_model_path,
                                                             epoch_or_best=pretrained_epoch_or_best).encoder
            else:
                # Input --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
                # Output --> Packed_Sequence([Batch_Size, Max_Seq_Len, First_BiLSTM_Stage_Hidden_Size])
                self._embedding_bilstm = nn.LSTM(self._word_embedding_size, self._first_bilstm_hidden_size,
                                                 self._first_bilst_num_layers, bidirectional=True, batch_first=True)

        else:
            if (pretrained_embedding_layer_path is not None):
                model_files = os.listdir(pretrained_embedding_layer_path + os.sep + "Model" + os.sep)
                best_epoch = [int(file.split('.')[-2]) for file in model_files if file.endswith('.bst')]
                if (not bool(best_epoch)):
                    last_epoch = max([int(file.split('.')[-2]) for file in model_files if file.endswith('.ckp')])
                pretrained_model_path = pretrained_embedding_layer_path + os.sep + "Model" + os.sep
                pretrained_epoch_or_best = -1 if bool(best_epoch) else last_epoch

                self._first_x_bilstm = ESIM_Embed_Layer_AE(load_path=pretrained_model_path,
                                                           epoch_or_best=pretrained_epoch_or_best).encoder

                self._first_y_bilstm = ESIM_Embed_Layer_AE(load_path=pretrained_model_path,
                                                           epoch_or_best=pretrained_epoch_or_best).encoder
            else:
                # Input --> Packed_Sequence([Batch_Size, Max_Seq_Len, word_embedding_size])
                # Output --> Packed_Sequence([Batch_Size, Max_Seq_Len, First_BiLSTM_Stage_Hidden_Size])
                self._first_x_bilstm = nn.LSTM(self._word_embedding_size, self._first_bilstm_hidden_size,
                                               self._first_bilst_num_layers, bidirectional=True, batch_first=True)

                # Input --> Packed_Sequence([Num_Descriptions, Max_Description_Len, Word_Embedding_Size])
                # Output --> Packed_Sequence([Num_Descriptions, Max_Description_Len, First_BiLSTM_Stage_Hidden_Size])
                self._first_y_bilstm = nn.LSTM(self._word_embedding_size, self._first_bilstm_hidden_size,
                                               self._first_bilst_num_layers, bidirectional=True, batch_first=True)



        #############################################################
        # Here We Compute the Attention Weights in the Forward Pass #
        #############################################################

        # Here we unpack the PackedSequences.
        # Compute the attention weights.                  --- Equations (12) & (13) of Chen et al. (2017).
        # Compute the reweighted versions os 'a' and 'b'. --- Equations (12) & (13) of Chen et al. (2017).
        # Compute the 'm' vectors.                        --- Equations (14) & (15) of Chen et al. (2017).
        # Repack the sequences.



        #####################################################################################################
        # 1 layer (0 hidden) MLP that reduces the dimensionality of the post-attention concatenated vectors #
        #####################################################################################################
        # Input --> [sum(seq_lengths in ("Batch" or "Relations' Descriptions")), First_BiLSTM_Stage_Hidden_Size]
        # Output --> [sum(seq_lengths in ("Batch" or "Relations' Descriptions")), Post_Attention_Embeddings_Size]
        self._dim_reduction_mlp = MLP([8*self._first_bilstm_hidden_size, self._post_attention_size],
                                      [nn.LeakyReLU(negative_slope=self._leakyReLU_negative_slope)],
                                      device=self._device)



        ###############################
        # Inference Composition Layer #
        ###############################

        # Input --> [Batch_Size, Max_Seq_Len, Post_Attention_Embeddings_Size]
        # Output --> [Batch_Size, Max_Seq_Len, Second_BiLSTM_Stage_Hidden_Size]
        self._second_x_bilstm = nn.LSTM(self._post_attention_size, self._second_bilstm_hidden_size,
                                        self._second_bilst_num_layers, bidirectional=True, batch_first=True)

        # Input --> [Num_Descriptions, Max_Description_Len, Post_Attention_Embeddings_Size]
        # Output --> [Num_Descriptions, Max_Description_Len, Second_BiLSTM_Stage_Hidden_Size]
        self._second_y_bilstm = nn.LSTM(self._post_attention_size, self._second_bilstm_hidden_size,
                                        self._second_bilst_num_layers, bidirectional=True, batch_first=True)



        # Avg & Max pooling happen at this stage, in the forward pass.



        ##########################################################
        # 2 layers (1 hidden) MLP that estimates the final score #
        ##########################################################
        # We split the MLP into 2 different MLPs, so that we have the possibility of training ESIM on SNLI & MultiNLI
        # and then perform transfer learning to our task. This way, we load the entire ESIM model for NLI and load all
        # the weights, except for the very last layer.

        # Input --> [Batch_Size * Num_Descriptions, Second_BiLSTM_Stage_Hidden_Size]
        # Output --> [Batch_Size * Num_Descriptions, score_mlp_after_first_layer_sizes[-1] (this will be 0 or 3)]

        # All layers but the last.
        layers = [8*self._second_bilstm_hidden_size] + self._score_mlp_post_first_layer_sizes[:-1]
        activations = [nn.LeakyReLU(negative_slope=self._leakyReLU_negative_slope)]
        activations += self._score_mlp_post_first_layer_activations[:-1]
        self._score_mlp_but_last = MLP(layers, activations, device=self._device)

        # Last layer.
        self._score_mlp_last = MLP(self._score_mlp_post_first_layer_sizes[-2:],
                                   self._score_mlp_post_first_layer_activations[-1:], device=self._device)

        # self._softmax_across_relations = torch.nn.Softmax(dim=1)


        # Load previous state, if adequate.
        previous_weights = trainer_helpers.load_checkpoint_state(self._load_path, "weights", self._epoch_or_best,
                                                                 device=self._device)

        if (previous_weights is not None):
            self.load_state_dict(previous_weights)

        self.to(device=self._device)




    # TODO: Implement memory freeing methods.
    # TODO: remake docstring to account for data_loader_type.
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
            x_sentences_post_first_lstm, (_, _) = self._embedding_bilstm(x_sentences)
        else:
            x_sentences_post_first_lstm, (_, _) = self._first_x_bilstm(x_sentences)

        # In --> Packed_Sequence([Num_Descriptions, Max_Description_Len, word_embedding_size])
        # Out --> Packed_Sequence([Num_Descriptions, Max_Description_Len, First_BiLSTM_Stage_Hidden_Size])
        if (self._single_embedding_bilstm):
            y_sentences_post_first_lstm, (y_h_n, _) = self._embedding_bilstm(y_sentences)
        else:
            y_sentences_post_first_lstm, (y_h_n, _) = self._first_y_bilstm(y_sentences)
        self._encoded_rels_batch = y_h_n.permute(1, 0, 2).reshape(y_h_n.shape[1], y_h_n.shape[0] * y_h_n.shape[2])
        self._encoded_rels_batch = self._encoded_rels_batch[relation_perm_unsort_idxs, :]




        #############################
        # Compute Attention Weights #
        #############################

        # Unpack packed sequences, in order to be able to compute the tensor products necessary to compute the
        # attention weights.
        unpacked_x_sentences, x_lengths = pad_packed_sequence(x_sentences_post_first_lstm, batch_first=True)
        unpacked_y_sentences, y_lengths = pad_packed_sequence(y_sentences_post_first_lstm, batch_first=True)
        num_x_sentences = x_lengths.shape[0]
        num_y_sentences = y_lengths.shape[0]
        # TODO: This can be take care of before? So that at this point nothing needs to be loaded into GPU memory.
        device_x_lengths = torch.empty(x_lengths.shape, dtype=torch.int64, device=self._device).copy_(x_lengths,
                                                                                                      non_blocking=True)
        device_y_lengths = torch.empty(y_lengths.shape, dtype=torch.int64, device=self._device).copy_(y_lengths,
                                                                                                      non_blocking=True)


        # Create a mask for the attention elements which are not meant to be included in the normalisation factors, due
        # to variable sequence length.
        max_len_x = torch.max(device_x_lengths)
        max_len_y = torch.max(device_y_lengths)

        mask_x = torch.arange(max_len_x, device=self._device).expand(num_x_sentences, max_len_x.data)
        mask_x = mask_x < device_x_lengths.unsqueeze(1)
        mask_y = torch.arange(max_len_y, device=self._device).expand(num_y_sentences, max_len_y.data)
        mask_y = mask_y < device_y_lengths.unsqueeze(1)

        mask_x = mask_x.unsqueeze(1).unsqueeze(3)
        mask_y = mask_y.unsqueeze(0).unsqueeze(2)

        # Here we compute the masks (based on setting some values to -inf) used in computing both x_tilde and y_tilde.
        attention_values_mask_for_x_tilde = torch.matmul(torch.ones_like(mask_x).float(), mask_y.float()).byte()
        attention_values_mask_for_y_tilde = torch.matmul(mask_x.float(), torch.ones_like(mask_y).float()).byte()


        # Permute last two axis of the y sentences, in order to allow the inter sentences matrix multiplications.
        unpacked_y_sentences = unpacked_y_sentences.permute(0, 2, 1)

        # Unsqueeze dimensions for easy broadcasting in torch.matmul.
        unpacked_x_sentences = unpacked_x_sentences.unsqueeze(1)
        unpacked_y_sentences = unpacked_y_sentences.unsqueeze(0)


        # Compute the Attention Elements (e_{ij}).
        attention_elements = torch.matmul(unpacked_x_sentences, unpacked_y_sentences)


        # Get y sentences' tensor's axes back to the correct shape.
        unpacked_y_sentences = unpacked_y_sentences.permute(0, 1, 3, 2)




        #######################################################################################################
        # Compute Attention Weighted Embeddings ('m' vectors --- Equations (14) & (15) of Chen et al. (2017)) #
        #######################################################################################################
        # Here we attempt to minimize the GPU's memory usage. For that effect we first compute m_x (m_a) and compress
        # it, passing it through the first MLP. Only then will we compute m_y (m_b), and then compress it.

        # We start by computing the weights that the words of the y sentences will have when building x_tilde. For that,
        # we first clone 'attention_elements' so that we can replace, in-place, some of its elements (masked ones) by
        # '-inf', to avoid numerical errors caused by the masking process.
        attention_for_x_tilde = torch.clone(attention_elements)
        attention_for_x_tilde[~attention_values_mask_for_x_tilde] = -float("inf")

        x_tilde_weights = torch.softmax(attention_for_x_tilde, dim=3)

        # We compute 'x tilde' at this stage.
        x_tilde = torch.matmul(x_tilde_weights, unpacked_y_sentences)

        # Compute the 'x difference' and 'x element wise multiplication' elements of 'm_x', and concatenate with
        # 'x tilde'.
        m_x = torch.cat((x_tilde, unpacked_x_sentences - x_tilde, unpacked_x_sentences * x_tilde), dim=3)

        # x_tilde is no longer needed, so release it from memory.
        del x_tilde
        #TODO: If cuda
        torch.cuda.empty_cache()

        # Add the normal x encoded sentences to 'm_x'.
        m_x = torch.cat((unpacked_x_sentences.expand(-1, unpacked_y_sentences.shape[1], -1, -1), m_x), dim=3)

        # Reshape m_x:
        # From: [Batch_Size, Num_Descriptions, Max_Seq_Len, 4 * First_BiLSTM_Stage_Hidden_Size]
        # To:   [Batch_Size * Num_Descriptions, Max_Seq_Len, 4 * First_BiLSTM_Stage_Hidden_Size]
        m_x = m_x.reshape(m_x.shape[0] * m_x.shape[1], m_x.shape[2], m_x.shape[3])

        # Compute m_x's sequence lengths and transform m_x into a PackedSequence. We pad it before reducing the
        # embeddings' size so as to avoid having to pass the padding through the MLP.
        m_x_lens = x_lengths.unsqueeze(1).expand(-1, y_lengths.shape[0])
        m_x_lens = m_x_lens.reshape(m_x_lens.shape[0] * m_x_lens.shape[1])
        m_x = pack_padded_sequence(m_x, m_x_lens, batch_first=True)

        # Reduce the dimension of m_x.
        m_x = PackedSequence(self._dim_reduction_mlp(m_x.data)[0], m_x.batch_sizes)



        # *** Compute the m_y vectors *** #

        # We start by computing the weights that the words of the x sentences will have when building y_tilde. For that,
        # we first clone 'attention_elements' so that we can replace, in-place, some of its elements (masked ones) by
        # '-inf', to avoid numerical errors caused by the masking process.
        attention_for_x_tilde = torch.clone(attention_elements)
        attention_for_x_tilde[~attention_values_mask_for_y_tilde] = -float("inf")

        # Compute the normalised attention weights, for 'y tilde'.
        y_tilde_weights = torch.softmax(attention_for_x_tilde, dim=2)

        # We compute 'y tilde' at this stage.
        y_tilde = torch.matmul(y_tilde_weights.permute(0, 1, 3, 2), unpacked_x_sentences)

        # Compute the 'y difference' and 'y element wise multiplication' elements of 'm_y', and concatenate with
        # 'y tilde'. torch.cat makes a copy of the input vectors.
        m_y = torch.cat((y_tilde, unpacked_y_sentences - y_tilde, unpacked_y_sentences * y_tilde), dim=3)

        # y_tilde is no longer needed, so release it from memory.
        del y_tilde
        # TODO: If cuda
        torch.cuda.empty_cache()

        # Add the normal y encoded sentences to 'm_y'.
        m_y = torch.cat((unpacked_y_sentences.squeeze(2).expand(unpacked_x_sentences.shape[0], -1, -1, -1), m_y), dim=3)

        # In order to apply the same PackedSequence trick as we did for m_x, in order to avoid making computations for
        # masked values, we need to first rotate m_y to get it in the shape:
        # [Num_Descriptions, Batch_Size, Max_Description_Len, 4 * First_BiLSTM_Stage_Hidden_Size]
        # This will allow us to correctly compute the sentence lengths of m_y (because we now need to account for each
        # y sentence having been paired with each x sentence).
        m_y = m_y.permute(1, 0, 2, 3)

        # Reshape m_y:
        # From: [Num_Descriptions, Batch_Size, Max_Description_Len, 4 * First_BiLSTM_Stage_Hidden_Size]
        # To:   [Num_Descriptions * Batch_Size, Max_Description_Len, 4 * First_BiLSTM_Stage_Hidden_Size]
        m_y = m_y.reshape(num_y_sentences * num_x_sentences, m_y.shape[2], m_y.shape[3])

        # Compute m_y's sequence lengths and transform m_y into a PackedSequence. We pad it before reducing the
        # embeddings' size so as to avoid having to pass the padding through the MLP.
        m_y_lens = y_lengths.unsqueeze(1).expand(-1, x_lengths.shape[0])
        m_y_lens = m_y_lens.reshape(m_y_lens.shape[0] * m_y_lens.shape[1])
        m_y = pack_padded_sequence(m_y, m_y_lens, batch_first=True)

        # Reduce the dimension of m_y.
        m_y = PackedSequence(self._dim_reduction_mlp(m_y.data)[0], m_y.batch_sizes)




        #######################
        # Compositional Layer #
        #######################

        # *** Compute the 'v_x' and 'v_y' vectors (result of second stage BiLSTM pass) *** #

        # In --> Packed_Sequence([Batch_Size * Num_Descriptions, Max_Seq_Len, 4*First_BiLSTM_Stage_Hidden_Size])
        # Out --> Packed_Sequence([Batch_Size * Num_Descriptions, Max_Seq_Len, Post_Attention_Embeddings_Size])
        v_x, (_, _) = self._second_x_bilstm(m_x)

        # In --> Packed_Sequence([Batch_Size * Num_Descriptions, Max_Description_Len, 4*First_BiLSTM_Stage_Hidden_Size])
        # Out --> Packed_Sequence([Batch_Size * Num_Descriptions, Max_Description_Len, Post_Attention_Embeddings_Size])
        v_y, (_, _) = self._second_y_bilstm(m_y)


        unpacked_v_x, m_x_lens = pad_packed_sequence(v_x, batch_first=True)
        device_m_x_lens = torch.empty(m_x_lens.shape, dtype=torch.int64, device=self._device).copy_(m_x_lens,
                                                                                                    non_blocking=True)

        v_x_avg = torch.div(torch.sum(unpacked_v_x, dim=1), device_m_x_lens.unsqueeze(1).float())
        v_x_max, _ = torch.max(unpacked_v_x, dim=1)


        unpacked_v_y, m_y_lens = pad_packed_sequence(v_y, batch_first=True)
        device_m_y_lens = torch.empty(m_y_lens.shape, dtype=torch.int64, device=self._device).copy_(m_y_lens,
                                                                                                    non_blocking=True)

        v_y_avg    = torch.div(torch.sum(unpacked_v_y, dim=1), device_m_y_lens.unsqueeze(1).to(torch.float32))
        v_y_max, _ = torch.max(unpacked_v_y, dim=1)
        v_y        = torch.cat((v_y_avg, v_y_max), dim=1)

        # In the case of v_y (which originates from m_y) we need to reshape and permute it, so that we can once more
        # have the elements in the order:
        # [Batch_Size, Num_Descriptions, Max_Description_Len, Post_Attention_Embeddings_Size]
        # thus allowing us to pair them correctly with v_x, to form v.
        v_y = v_y.reshape(num_y_sentences, num_x_sentences, v_y.shape[1])
        v_y = v_y.permute(1, 0, 2)
        v_y = v_y.reshape(num_x_sentences * num_y_sentences, v_y.shape[2])


        v = torch.cat((v_x_avg, v_x_max, v_y), dim=1)


        logits = self._score_mlp_last(self._score_mlp_but_last(v)[0])[0]

        logits = logits.squeeze(1).reshape(num_x_sentences, num_y_sentences)

        logits = logits[:, relation_perm_unsort_idxs]


        # When evaluating, we want to aggregate over the multiple descriptions for each relation.
        if (data_loader_type != 'train' and batch[1][1] is not None):
            aggregated_logits = torch.max(logits[:, batch[1][1][0]], dim=1)[0].unsqueeze(1)
            for aggregate_idxs in batch[1][1][1:]:
                aggregated_logits=torch.cat((aggregated_logits,
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
                "post_attention_size"                   : self._post_attention_size,
                "second_bilstm_hidden_size"             : self._second_bilstm_hidden_size,
                "score_mlp_post_first_layer_sizes"      : self._score_mlp_post_first_layer_sizes,
                "score_mlp_post_first_layer_activations": self._score_mlp_post_first_layer_activations,
                "leakyReLU_negative_slope"              : self._leakyReLU_negative_slope,
                "first_bilst_num_layers"                : self._first_bilst_num_layers,
                "second_bilst_num_layers"               : self._second_bilst_num_layers,
            }
            torch.save(architecture, path + "architecture.act")

        # Saves the weights for the current_epoch, associated either with a checkpoint or the best performing model.
        if (save_parameters):
            torch.save(self.state_dict(), path + "weights" + file_type)




    @property
    def y_sentences(self):
        return self._y_sentences

    @property
    def y_sentences_batch(self):
        return self._y_sentences_batch

    @property
    def relation_perm_unsort_idxs_batch(self):
        return self._relation_perm_unsort_idxs_batch

    @property
    def encoded_rels_batch(self):
        return self._encoded_rels_batch