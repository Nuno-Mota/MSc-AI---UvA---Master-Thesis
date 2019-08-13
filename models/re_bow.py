"""
##################################################
##################################################
## TODO                                         ##
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
    """"""

    def __init__(self, relations_embeds_file=None, word_embedding_size=None, first_bilstm_hidden_size=None,
                 post_attention_size=None, second_bilstm_hidden_size=None, score_mlp_post_first_layer_sizes=None,
                 score_mlp_post_first_layer_activations=None, first_bilst_num_layers=1, second_bilst_num_layers=1,
                 encoder_params_dict={},
                 rels_descs_embed_type=None, rels_embedder_hidden_size=None,
                 decoder_mlp_layers=None, decoder_mlp_activations=None, decoder_mlp_dropout_values=None,
                 decoder_mlp_params_dict={}, decoder_params_dict={},
                 load_path=None, epoch_or_best=None, loaded_arch=False,
                 device=torch.device('cpu'), **model_params):
        """
        Instantiates a RE model object.
        """

        super(RE_BOW, self).__init__()


        # Input parameters.
        # TODO: Rename to classes.
        self._relations_embeddings_file = relations_embeds_file

        self._load_path     = load_path
        self._epoch_or_best = epoch_or_best

        if (not loaded_arch and load_path is not None):
            with open(load_path + "architecture.act", 'rb') as f:
                self.__init__(relations_embeddings_file=relations_embeds_file, load_path=load_path,
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

        # Test_params
        self._step = 0




    # TODO: make docstring.
    def forward(self, batch, data_loader_type):

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
        if (save_parameters):
            torch.save(self.state_dict(), path + "weights" + file_type)

        # Save both the encoder and the decoder separately, in case only one of them is meant to be used.
        # Make sure the 'Encoder' Directory exists
        directory = os.path.dirname(path + 'Encoder/')
        if (not os.path.exists(directory)):
            os.makedirs(directory)
        self._encoder.save(current_epoch, path + 'Encoder/', file_type, save_parameters)

        # Make sure the 'Decoder' Directory exists
        directory = os.path.dirname(path + 'Decoder/')
        if (not os.path.exists(directory)):
            os.makedirs(directory)
        self._decoder.save(current_epoch, path + 'Decoder/', file_type, save_parameters)




    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder