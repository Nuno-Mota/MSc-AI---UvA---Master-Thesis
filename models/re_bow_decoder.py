"""
##################################################
##################################################
## This file contains the implementation of a   ##
## Bag of Words decoder, projecting from a      ##
## representation of each class to the          ##
## vocabulary size.                             ##
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

class RE_BOW_DECODER(nn.Module):
    """"""

    def __init__(self, word_embedding_size=None, vector_relation_encoding=None, num_relations=None, vocab_size=None,
                 rels_descs_embed_type=None, rels_embedder_hidden_size=None,
                 mlp_layers=None, mlp_activations=None, mlp_dropout_values=None, mlp_params_dict={},
                 load_path=None, epoch_or_best=None, loaded_arch=False, device=torch.device('cpu'), **model_params):
        """
        Instantiates a RE_BOW_DECODER model object.
        TODO: Make docstring.
        """

        super(RE_BOW_DECODER, self).__init__()


        self._load_path     = load_path
        self._epoch_or_best = epoch_or_best

        # Input parameters.
        if (not loaded_arch and load_path is not None):
            with open(load_path + "architecture.act", 'rb') as f:
                self.__init__(load_path=load_path, epoch_or_best=epoch_or_best, loaded_arch=True,
                              device=device, **torch.load(f))
        else:
            self._word_embedding_size       = word_embedding_size
            self._vector_relation_encoding  = vector_relation_encoding
            self._num_relations             = num_relations
            self._vocab_size                = vocab_size
            self._rels_descs_embed_type     = rels_descs_embed_type
            self._rels_embedder_hidden_size = rels_embedder_hidden_size
            self._mlp_layers                = mlp_layers
            self._mlp_activations           = mlp_activations
            self._mlp_dropout_values        = mlp_dropout_values
            self._mlp_params_dict           = mlp_params_dict
            if (loaded_arch):
                return

        # Set the load model variables.
        self._load_path     = load_path
        self._epoch_or_best = epoch_or_best

        # Set correct device variables.
        self._device = device
        self._mlp_params_dict['device'] = self._device


        # If we are using a simple relation vector parameterization (a |R| x |V| matrix), we implement it here.
        if (self._vector_relation_encoding):
            self._vector_encodings   = torch.nn.Parameter(torch.rand(self._num_relations, self._vocab_size))
            self._softmax_over_vocab = torch.nn.Softmax(dim=1)

        # If we are using a more complex decoder, we implement it here.
        else:
            # Here we define the relations' descriptions' embedder.
            if (self._rels_descs_embed_type == 'bilstm'):
                embedder_num_layers = 1
                self._descriptions_embedder = nn.LSTM(self._word_embedding_size, self._rels_embedder_hidden_size,
                                                      embedder_num_layers, bidirectional=True, batch_first=True)
            elif (self._rels_descs_embed_type == 'avg'):
                pass
            elif (self._rels_descs_embed_type == 'esim'):
                pass
            else:
                raise NotImplementedError


            # Create the MLP that will produce the BoW probabilities.
            if (bool(self._mlp_params_dict)):
                self._mlp = MLP(**self._mlp_params_dict)
            else:
                self._mlp = MLP(self._mlp_layers, self._mlp_activations, self._mlp_dropout_values)


        #################
        # MISCELLANEOUS #
        #################

        # Load previous state, if adequate.
        previous_weights = trainer_helpers.load_checkpoint_state(self._load_path, "weights", self._epoch_or_best,
                                                                 device=self._device)

        if (previous_weights is not None):
            self.load_state_dict(previous_weights)

        self.to(device=self._device)




    # TODO: The parameter relation_unsort_idxs does not reflect our latest findings and is in fact non-sensical. Read
    # TODO: this file's docstring.
    def forward(self, batch=None, data_loader_type='train', relation_unsort_idxs=None):
        """
        Performs the network's forward pass.


        :param x_batch         : The input batch to the network.

        :param data_loader_type: Allows the identification of whether the model is being trained or simply evaluated,
                                 which can be important due to different behaviour on either stage.


        :return: The output of the network, for the input batch.
        """

        if (self._vector_relation_encoding):
            word_probs = self._softmax_over_vocab(self._vector_encodings)

        else:
            # TODO: This is a workaround to solve the issue that when the dataloader's 'pin_memory=True' PackedSequences
            # TODO: get 'destroyed' and only the 'data' and 'batch_sizes' tensors remain. In this case we only need to
            # TODO: do it when pre training the decoder (which can be identified when 'relation_unsort_idxs=None').
            if (relation_unsort_idxs is None):
                if (str(self._device)[:3] != 'cpu'):
                    batch = PackedSequence(batch[0], batch[1].to(device='cpu'))


            # Get an embedding of the relation descriptions, with which the word probabilities will be estimated.
            if (self._rels_descs_embed_type == 'bilstm'):
                _, (y_h_n, _) = self._descriptions_embedder(batch)

                # We also retrieve the descriptions' last hidden states, 'y_h_n', to use as relation encodings. These
                # will be used for unsupervised training. We concatenate both LSTM's directions' last hidden states:
                batch = y_h_n.permute(1, 0, 2).reshape(y_h_n.shape[1], y_h_n.shape[0] * y_h_n.shape[2])

                if (relation_unsort_idxs is not None):
                    # Restore the correct relations' descriptions' order.
                    batch = batch[relation_unsort_idxs, :]


            # Here we perform the unsupervised part of the training procedure. From each relation's description's
            # sentence embedding we estimate the probabilities of each word present in the training set. Afterwards,
            # when computing the model's loss, for each individual sentence x we use the relation predictions,
            # 'rel_probs', to marginalize the word probabilities over the relations.
            word_probs = self._mlp(batch)[0]

        return (word_probs, )




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
                "word_embedding_size"      : self._word_embedding_size,
                'rels_descs_embed_type'    : self._rels_descs_embed_type,
                'rels_embedder_hidden_size': self._rels_embedder_hidden_size,
            }
            if (bool(self._mlp_params_dict)):
                architecture['mlp_params_dict'] = self._mlp_params_dict
            else:
                architecture['mlp_params_dict'] = {
                    "layers_sizes"        : self._mlp_layers,
                    "activation_functions": self._mlp_activations,
                    "dropout_values"      : self._mlp_dropout_values,
                }
            torch.save(architecture, path + "architecture.act")

        # Saves the weights for the current_epoch, associated either with a checkpoint or the best performing model.
        if (save_weights):
            torch.save(self.state_dict(), path + "weights" + file_type)




    @property
    def word_embedding_size(self):
        return self._word_embedding_size

    @property
    def vector_relation_encoding(self):
        return self._vector_relation_encoding

    @property
    def num_relations(self):
        return self._num_relations

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def rels_descs_embed_type(self):
        return self._rels_descs_embed_type

    @property
    def rels_embedder_hidden_size(self):
        return self._rels_embedder_hidden_size

    @property
    def mlp(self):
        return self._mlp