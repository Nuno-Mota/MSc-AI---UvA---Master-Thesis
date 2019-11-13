"""
##################################################
##################################################
## This file contains the implementation of a   ##
## Multi Layer Perceptron Model.                ##
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
#TODO: dependant on the trainer.
import os


# *** Own modules imports. *** #

import helpers.trainer.helpers as trainer_helpers





#################
##### CLASS #####
#################

class MLP(nn.Module):
    """
    This class implements a Multi Layer Perceptron Neural Network.
    """

    def __init__(self, layers_sizes=None, activation_functions=None, dropout_values=None, bias=True,
                 weight_initialization=None, load_path=None, epoch_or_best=None, device=torch.device('cpu'),
                 **param_dict):
        """
        Instantiates a MLP model object.


        :param layers_sizes         : A list that specifies the size of each layer.

        :param activation_functions : A list that specifies the activation function applied at each layer.

        :param dropout_values       : A list that specifies the dropout values of each layer.

        :param bias                 : A boolean that determines whether to use bias terms or not.

        :param weight_initialization: Specifies the weight initialization function. NOT IMPLEMENTED TODO

        :param load_path               : The path to a previously saved state.

        :param epoch_or_best           : A number indicating which epoch to load ('-1' for the best performing epoch).

        :param device                  : The device (CPU, GPU-n) on which the model is meant to be run.

        :param params_dict             : Allows passing some of the previous parameters as a dictionary that gets
                                         unpacked.
        """

        # Test whether it is possible to instantiate a MLP model with the provided input parameters.
        if ((layers_sizes is None or activation_functions is None) and load_path is None):
            raise ValueError("Not enough parameters with which to build a MLP model.")

        super(MLP, self).__init__()

        self._device = device
        # TODO: Should implement the simple loading pattern.

        # If there is a load_path, load the MLP's architecture from memory.
        arch = None if load_path is None else torch.load(load_path + "architecture.act")

        # Define the layers' sizes, in order to save the architecture of the MLP to file.
        self._layers_sizes = layers_sizes if arch is None else arch["layers_sizes"]

        # Define the activation functions, in order to save the architecture of the MLP to file, and to call in the
        # forward pass.
        self._activation_functions = activation_functions if arch is None else arch["activation_functions"]

        # Define the dropout values, to be considered for the input of each layer, in order to save the architecture of
        # the MLP to file, and to call in the forward pass.
        self._dropout_values = dropout_values if arch is None else arch["dropout_values"]
        if (self._dropout_values is not None):
            self._dropout = nn.ModuleList([nn.Dropout(p=dropout_value) for dropout_value in self._dropout_values])


        # Define whether the model's layers will have bias terms or not, in order to save the architecture of the MLP
        # to file.
        self._bias = bias if arch is None else arch["bias"]

        # Define whether the model's layers weight iniliatisation function, in order to save the architecture of the MLP
        # to file. TODO: add them
        self._weight_initialization = weight_initialization if arch is None else arch["weight_initialization"]

        # Build the MLP's layers.
        self._layers = nn.ModuleList(
            [nn.Linear(dims[0], dims[1], bias=self._bias) for dims in zip(self._layers_sizes, self._layers_sizes[1:])])


        #################
        # MISCELLANEOUS #
        #################

        # Load previous state, if adequate.
        previous_weights = trainer_helpers.load_checkpoint_state(load_path, "weights", epoch_or_best,
                                                                 device=self._device)

        if (previous_weights is not None):
            self.load_state_dict(previous_weights)

        self.to(device=self._device)




    def forward(self, x_batch, data_loader_type=None):
        """
        Performs the network's forward pass.


        :param x_batch         : The input batch to the network.

        :param data_loader_type: Allows the identification of whether the model is being trained or simply evaluated,
                                 which can be important due to different behaviour on either stage.


        :return: The output of the network, for the input batch.
        """

        if (self._dropout_values is None):
            for i, activation_function in enumerate(self._activation_functions):
                x_batch = activation_function(self._layers[i](x_batch))
        else:
            for i, activation_function in enumerate(self._activation_functions):
                x_batch = activation_function(self._layers[i](self._dropout[i](x_batch)))

        return (x_batch, )





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
                "layers_sizes"         : self._layers_sizes,
                "activation_functions" : self._activation_functions,
                "bias"                 : self._bias,
                "weight_initialization": self._weight_initialization
            }
            torch.save(architecture, path + "architecture.act")

        # Saves the weights for the current_epoch, associated either with a checkpoint or the best performing model.
        if (save_weights):
            torch.save(self.state_dict(), path + "weights" + file_type)