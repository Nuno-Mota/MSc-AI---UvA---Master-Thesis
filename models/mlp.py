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
        TODO: Add dropout parameter docstring. Add batchnorm?
        Instantiates a MLP model object. NOTE: Arguments can be either directly specified on object creation or they
                                               can be unpacked using a parameter dictionary (**param_dict). For each
                                               argument only one of the methods can be used, otherwise an error occurs.
                                               It is, however, possible to define some arguments specifically and others
                                               by defining them in the parameter dictionary. E.g:
                                               param_dict = {'layers_sizes': [784, 300, 10], 'bias': False}
                                               mlp = MLP(activation_functions=[Relu(), Softmax()], **param_dict)


        :param layers_sizes         : A list that specifies the size of each layer.

        :param activation_functions : A list that specifies the activation function applied at each layer.

        :param bias                 : A boolean that determines whether to use bias terms or not.

        :param weight_initialization: TODO:

        :param load_path            : The path that leads to the directory where the states pertaining a previously
                                      trained model are located.

        :param epoch_or_best   : Determines whether to load a specific checkpoint or whether to load the epoch
                                      where the model performed the best.
        """

        # Test whether it is possible to instantiate a MLP model with the provided input parameters.
        if ((layers_sizes is None or activation_functions is None) and load_path is None):
            raise ValueError("Not enough parameters with which to build a MLP model.")

        super(MLP, self).__init__()

        self._device = device

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

        # Define whether the model's layers weight iniliatization function, in order to save the architecture of the MLP
        # to file.
        self._weight_initialization = weight_initialization if arch is None else arch["weight_initialization"]  # TODO: add them

        # Build the MLP's layers.
        self._layers = nn.ModuleList(
            [nn.Linear(dims[0], dims[1], bias=self._bias) for dims in zip(self._layers_sizes, self._layers_sizes[1:])])


        # Load previously saved weights, if necessary. TODO: Change this part so the model is independent of the trainer.
        previous_weights = trainer_helpers.load_checkpoint_state(load_path, "weights", epoch_or_best,
                                                                 device=self._device)

        if (previous_weights is not None):
            self.load_state_dict(previous_weights)

        # if (load_epoch_or_best is not None and isinstance(load_epoch_or_best, int)):
        #     existing_files = os.listdir(load_path)
        #     if (load_epoch_or_best == -1 and # -1 identifies the best model
        #             ".bst" in {existing_file[-4:] for existing_file in existing_files}):
        #         # Get the correct path to where the weights are stored
        #         path_to_load = load_path
        #         path_to_load += [existing_file for existing_file in existing_files if existing_file.endswith(".bst")][0]
        #         self.load_state_dict(torch.load(path_to_load))
        #     elif (load_epoch_or_best >= 0 and os.path.exists(
        #             load_path + "weights." + str(load_epoch_or_best) + ".ckp")):
        #         self.load_state_dict(torch.load(load_path + "weights." + str(load_epoch_or_best) + ".ckp"))
        #     else:
        #         raise ValueError("Tried to load model's state dict for invalid or inexisting epoch (epoch num: " + str(
        #             load_epoch_or_best) + ").")

        self.to(device=self._device)




    # TODO: remake docstring to account for data_loader_type.
    def forward(self, x_batch, data_loader_type=None):
        """
        Performs the network's forward pass.


        :param x_batch: The input batch to the network.


        :return: The output of the network, for the input batch.
        """

        if (self._dropout_values is None):
            for i, activation_function in enumerate(self._activation_functions):
                x_batch = activation_function(self._layers[i](x_batch))
        else:
            for i, activation_function in enumerate(self._activation_functions):
                x_batch = activation_function(self._layers[i](self._dropout[i](x_batch)))

        return (x_batch, )





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
                "layers_sizes"         : self._layers_sizes,
                "activation_functions" : self._activation_functions,
                "bias"                 : self._bias,
                "weight_initialization": self._weight_initialization
            }
            torch.save(architecture, path + "architecture.act")

        # Saves the weights for the current_epoch, associated either with a checkpoint or the best performing model.
        if (save_parameters):
            torch.save(self.state_dict(), path + "weights" + file_type)