"""
##################################################
##################################################
## This file contains the implementation of a   ##
## Bag of Words VAE-like model, with our        ##
## ESIM_StS as the encoder and our              ##
## re_bow_decoder defined in re_bow_decoder.py  ##
##                                              ##
## DISCLAIMER: This file ended up not being     ##
## part of our work, as it was mainly used for  ##
## exploratory experiments. At some point we    ##
## realised that some of our earlier ideas were ##
## actually non-sensical, but, since we also    ##
## decided to drop this line of research, we    ##
## never got around to removing the code used   ##
## by those earlier ideas. As such, this file   ##
## requires some major refactoring in order     ##
## to represent our latest findings.            ##
## TODO: This is left for future work.          ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import torch.nn as nn
#TODO: Remove "import os" if we can use trainer_helpers to load a model. Do we want to, though? Then the model is
#TODO: dependant on the trainer. Write model saving/loading function in the general trainers?
import os


# *** Own modules imports. *** #

from   models.esim_sts import ESIM_StS
from   models.re_bow_decoder import RE_BOW_DECODER
import helpers.trainer.helpers as trainer_helpers





#################
##### CLASS #####
#################

class RE_BOW(nn.Module):
    """
    This class implements our Relation Classification VAE-Like Unsupervised Learning model, based on a Bag of
    Words decoder.
    """

    def __init__(self, classes_descs_embed_file=None, word_embedding_size=None, pretrained_embedding_layer_path=None,
                 single_embedding_bilstm=None, first_bilstm_hidden_size=None, post_attention_size=None,
                 second_bilstm_hidden_size=None,
                 score_mlp_post_first_layer_sizes=None, score_mlp_post_first_layer_activations=None,
                 leakyReLU_negative_slope=None, first_bilst_num_layers=1, second_bilst_num_layers=1,
                 encoder_params_dict={},
                 rels_descs_embed_type=None, rels_embedder_hidden_size=None,
                 decoder_mlp_layers=None, decoder_mlp_activations=None, decoder_mlp_dropout_values=None,
                 decoder_mlp_params_dict={}, decoder_params_dict={},
                 load_path=None, epoch_or_best=None, loaded_arch=False,
                 device=torch.device('cpu'), **params_dict):
        """
        Instantiates a RE_BOW object.


        :param classes_descs_embed_file              : The path to the file of the ELMo embeddings of the classes'
                                                       descriptions.

        :param word_embedding_size                   : The size of the word embeddings.

        :param pretrained_embedding_layer_path       : Boolean indicating whether to use a pretrained embedding layer or
                                                       not.

        :param single_embedding_bilstm               : A boolean parameter to indicate whether to use a single BiLSTM
                                                       for both x and y, or whether to use distinct ones.

        :param first_bilstm_hidden_size              : The hidden size of the first BiLSTM stage.

        :param post_attention_size                   : The word embedding size (projected with a MLP) after attention
                                                       has been computed.

        :param second_bilstm_hidden_size             : The hidden size of the second BiLSTM stage.

        :param score_mlp_post_first_layer_sizes      : The score MLP last layer sizes (i.e., after the first layer,
                                                       which has a size dependent on the previous model components).

        :param score_mlp_post_first_layer_activations: The score MLP last activation functions (i.e., after the first
                                                       layer, which has a fixed LeakyReLU).

        :param leakyReLU_negative_slope              : The negative slope of the LeakyReLUs used in the model.

        :param first_bilst_num_layers                : The number of layers of the first BiLSTM stage.

        :param second_bilst_num_layers               : The number of layers of the second BiLSTM stage.

        :param encoder_params_dict                   : Allows passing some of the previous encoder parameters as a
                                                       dictionary that gets unpacked.

        # TODO: Disclaimer: The next two parameters are nonsensical
        :param rels_descs_embed_type                 : The type of embedding computed for the descriptions.

        :param rels_embedder_hidden_size             : The hidden size of a separate BiLSTM used to compute the
                                                       relations' descriptions lower dimensional word representation.

        :param decoder_mlp_layers                    : A list that specifies the layer sizes of the decoder's MLP.

        :param decoder_mlp_activations               : A list that specifies the activation function applied at each
                                                       layer of the decoder's MLP.

        :param decoder_mlp_dropout_values            : A list that specifies the dropout values of each layer of the
                                                       decoder's MLP.

        :param decoder_mlp_params_dict               : Allows passing some of the previous decoder's MLP parameters as a
                                                       dictionary that gets unpacked.

        :param decoder_params_dict                   : Allows passing some of the previous decoder parameters as a
                                                       dictionary that gets unpacked.

        :param load_path                             : The path to a previously saved state.

        :param epoch_or_best                         : A number indicating which epoch to load ('-1' for the best
                                                       performing epoch).

        :param loaded_arch                           : A boolean that allows for a simple previous state loading
                                                       pattern.

        :param device                                : The device (CPU, GPU-n) on which the model is meant to be run.

        :param params_dict                           : Allows passing some of the previous parameters as a dictionary
                                                       that gets unpacked.
        """

        super(RE_BOW, self).__init__()

        # Input parameters.
        self._classes_descs_embed_file = classes_descs_embed_file

        self._load_path     = load_path
        self._epoch_or_best = epoch_or_best


        # Simple loading pattern that allows for a parameter dictionary to be read directly from file.
        if (not loaded_arch and load_path is not None):
            with open(load_path + "architecture.act", 'rb') as f:
                self.__init__(classes_descs_embed_file=classes_descs_embed_file, load_path=load_path,
                              epoch_or_best=epoch_or_best, loaded_arch=True, device=device, **torch.load(f))
        else:
            self._word_embedding_size                    = word_embedding_size
            self._first_bilstm_hidden_size               = first_bilstm_hidden_size
            self._post_attention_size                    = post_attention_size
            self._second_bilstm_hidden_size              = second_bilstm_hidden_size
            self._score_mlp_post_first_layer_sizes       = score_mlp_post_first_layer_sizes
            self._score_mlp_post_first_layer_activations = score_mlp_post_first_layer_activations
            self._first_bilst_num_layers                 = first_bilst_num_layers
            self._second_bilst_num_layers                = second_bilst_num_layers
            self._encoder_params_dict                    = encoder_params_dict
            self._rels_descs_embed_type                  = rels_descs_embed_type
            self._rels_embedder_hidden_size              = rels_embedder_hidden_size
            self._decoder_mlp_layers                     = decoder_mlp_layers
            self._decoder_mlp_activations                = decoder_mlp_activations
            self._decoder_mlp_dropout_values             = decoder_mlp_dropout_values
            self._decoder_mlp_params_dict                = decoder_mlp_params_dict
            self._decoder_params_dict                    = decoder_params_dict
            if (loaded_arch):
                return

        # Set correct device variables
        self._device = device
        self._encoder_params_dict['device'] = self._device
        self._decoder_params_dict['device'] = self._device


        ###########
        # ENCODER #
        ###########

        if (bool(self._encoder_params_dict)):
            self._encoder = ESIM_StS(**self._encoder_params_dict)
        else:
            self._encoder = ESIM_StS(self._word_embedding_size, self._first_bilstm_hidden_size, self._post_attention_size,
                                     self._second_bilstm_hidden_size, self._score_mlp_post_first_layer_sizes,
                                     self._score_mlp_post_first_layer_activations, self._first_bilst_num_layers,
                                     self._second_bilst_num_layers)

        self._softmax_across_relations = torch.nn.Softmax(dim=1)


        ###########
        # DECODER #
        ###########

        # Create the decoder that will produce the BoW probabilities.
        if (bool(self._decoder_params_dict)):
            self._decoder = RE_BOW_DECODER(**self._decoder_params_dict)
        else:
            if (bool(self._decoder_mlp_params_dict)):
                self._decoder = RE_BOW_DECODER(self._word_embedding_size, self._rels_descs_embed_type,
                                               self._rels_embedder_hidden_size, **self._decoder_mlp_params_dict)
            else:
                self._decoder = RE_BOW_DECODER(self._word_embedding_size, self._rels_descs_embed_type,
                                               self._rels_embedder_hidden_size, self._bow_mlp_layers,
                                               self._bow_mlp_activations, self._mlp_dropout_values)


        #################
        # MISCELLANEOUS #
        #################

        # Load previous state, if adequate.
        previous_weights = trainer_helpers.load_checkpoint_state(self._load_path, "weights", self._epoch_or_best,
                                                                 device=self._device)

        if (previous_weights is not None):
            self.load_state_dict(previous_weights)

        self.to(device=self._device)




    def forward(self, batch, data_loader_type):
        """
        Performs the network's forward pass.


        :param x_batch         : The input batch to the network.

        :param data_loader_type: Allows the identification of whether the model is being trained or simply evaluated,
                                 which can be important due to different behaviour on either stage.


        :return: The output of the network, for the input batch.
        """

        # Here we get the indices of the relation descriptions used for this training batch, to produce averaged
        # relations' descriptions' embeddings. This way we avoid having to keep the batch around, when computing.
        if (data_loader_type == 'train' and self._rels_descs_embed_type == 'avg'):
            rels_indices = batch[1][0]

        # Get relation predictions and relation descriptions sentence embeddings, produced by the encoder.
        logits = self._encoder.forward(batch, data_loader_type)[0]
        rel_probs = self._softmax_across_relations(logits)

        if (data_loader_type == 'train'):
            if (self._decoder.vector_relation_encoding):
                words_probs = self._decoder()[0]
            else:
                # Get an embedding of the relation descriptions, with which the word probabilities will be estimated.
                if (self._decoder.rels_descs_embed_type == 'bilstm'):
                    rel_descs_batch = self._encoder.y_sentences_batch

                elif (self._decoder.rels_descs_embed_type == 'avg'):
                    rel_descs_batch = torch.cat([self._encoder.y_sentences[idx].mean(dim=0).unsqueeze(0)
                                                    for idx in rels_indices])

                elif (self._decoder.rels_descs_embed_type == 'esim'):
                    rel_descs_batch = self._encoder.encoded_rels_batch
                else:
                    raise NotImplementedError

                # Here we perform the unsupervised part of the training procedure. From each relation's description's
                # sentence embedding we estimate the probabilities of each word present in the training set. Afterwards,
                # when computing the model's loss, for each individual sentence x we use the relation predictions,
                # 'rel_probs', to marginalize the word probabilities over the relations.
                relation_unsort_idxs = self._encoder.relation_perm_unsort_idxs_batch
                words_probs = self._decoder(rel_descs_batch, relation_unsort_idxs=relation_unsort_idxs)[0]

            return (rel_probs, words_probs)

        else:
            rel_probs = self._encoder.forward(batch, data_loader_type)[0]

            return (rel_probs, )




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
            architecture = {}
            if (bool(self._encoder_params_dict)):
                architecture['encoder_params_dict'] = self._encoder_params_dict
            else:
                architecture['encoder_params_dict'] = {
                    "word_embedding_size"                   : self._word_embedding_size,
                    "first_bilstm_hidden_size"              : self._first_bilstm_hidden_size,
                    "post_attention_size"                   : self._post_attention_size,
                    "second_bilstm_hidden_size"             : self._second_bilstm_hidden_size,
                    "score_mlp_post_first_layer_sizes"      : self._score_mlp_post_first_layer_sizes,
                    "score_mlp_post_first_layer_activations": self._score_mlp_post_first_layer_activations,
                    "first_bilst_num_layers"                : self._first_bilst_num_layers,
                    "second_bilst_num_layers"               : self._second_bilst_num_layers,
                }
            if (bool(self._decoder_params_dict)):
                architecture['decoder_params_dict'] = self._decoder_params_dict
            else:
                architecture['decoder_params_dict'] = {
                    "word_embedding_size"      : self._word_embedding_size,
                    "rels_descs_embed_type"    : self._rels_descs_embed_type,
                    "rels_embedder_hidden_size": self._rels_embedder_hidden_size,
                }
                if (bool(self._decoder_mlp_params_dict)):
                    architecture['decoder_params_dict']["mlp_params_dict"] = self._decoder_mlp_params_dict
                else:
                    architecture['decoder_params_dict']["mlp_params_dict"] = {
                        "layers_sizes": self._decoder_mlp_layers,
                        "activation_functions": self._decoder_mlp_activations,
                        "dropout_values"      : self._decoder_mlp_dropout_values,
                    }
            torch.save(architecture, path + "architecture.act")

        # Saves the weights for the current_epoch, associated either with a checkpoint or the best performing model.
        if (save_weights):
            torch.save(self.state_dict(), path + "weights" + file_type)

        # Save both the encoder and the decoder separately, in case only one of them is meant to be used.
        # Make sure the 'Encoder' Directory exists
        directory = os.path.dirname(path + 'Encoder' + os.sep)
        if (not os.path.exists(directory)):
            os.makedirs(directory)
        self._encoder.save(current_epoch, path + 'Encoder' + os.sep, file_type, save_weights)

        # Make sure the 'Decoder' Directory exists
        directory = os.path.dirname(path + 'Decoder' + os.sep)
        if (not os.path.exists(directory)):
            os.makedirs(directory)
        self._decoder.save(current_epoch, path + 'Decoder' + os.sep, file_type, save_weights)




    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder