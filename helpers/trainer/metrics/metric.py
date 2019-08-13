"""
##################################################
##################################################
## This file implements the parent class of any ##
## metric. It implements the basic functions    ##
## that are general across different metrics.   ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import numpy as np
from   abc import abstractmethod
import os



# *** Own modules imports. *** #

import helpers.trainer.helpers as trainer_helpers
from   helpers.general_helpers import join_path, get_path_components





#################
##### CLASS #####
#################

class Metric(object):
    """
    The parent class of any implemented metric used to evaluate a model.
    """

    def __init__(self, short_name, evaluated_on_batches, max_num_batches, accumulate_n_batch_grads,
                 training_batch_sizes, lower_is_better, load_path, load_epoch_or_best, starting_batch_num):
        """
        Instantiates a Metric (or child of Metric) object.TODO: Missing parameters


        :param short_name          : The short name by which the metric is identified.

        :param evaluated_on_batches: Determines whether the metric is evaluated for minibatches. Used for plotting.

        :param max_num_batches     : The maximum number of batches in an epoch. Used to aggregate epoch information. See
                                     specific uses in child classes.

        :param lower_is_better     : Identifies whether a lower value of the metric indicates that the model performs
                                     better.

        :param load_path           : The path that leads to a saved state of an instance of this metric.

        :param load_epoch_or_best  : This parameter identifies whether to load the state associated with a specific
                                     epoch or the state of the epoch in which the model performed the best.

        :param starting_batch_num  : Some metrics can be added to the MetricsManager after the model has already been
                                     trained for a few epochs. This parameter identifies at which batch num the metric
                                     was started.
        """

        # Names.
        self._name = self.__class__.__name__
        self._short_name = short_name

        # Load previously saved metric state.
        self._load_path          = load_path
        self._load_epoch_or_best = load_epoch_or_best

        extra_name = ''
        if (load_path is not None):
            if (load_path[-1] != os.sep):
                path_components = get_path_components(load_path)
                extra_name = path_components[-1]
                load_path  = join_path(path_components[:-1]) + os.sep
        self._metric_state = trainer_helpers.load_checkpoint_state(load_path, extra_name + self._name, load_epoch_or_best)

        # Evaluation history.
        self._batch_history   = [] if self._metric_state is None else self._metric_state['batch_history']
        self._batch_history_x = [] if self._metric_state is None else self._metric_state['batch_history_x']
        self._epoch_history   = [] if self._metric_state is None else self._metric_state['epoch_history']
        self._epoch_history_x = [] if self._metric_state is None else self._metric_state['epoch_history_x']

        # Used to (only) evaluate the model's performance, so that no state of the singular evaluation is kept.
        self._eval_only_history = None

        # Batch information.
        self._evaluated_on_batches     = evaluated_on_batches
        self._max_num_batches          = max_num_batches
        self._accumulate_n_batch_grads = accumulate_n_batch_grads
        self._training_batch_sizes     = training_batch_sizes
        if (training_batch_sizes[0] is None):
            self._last_acc_batch_start_num = None
            self._normal_acc_batch_size    = None
            self._last_acc_batch_size      = None
        else:
            max_b = max_num_batches
            acc = accumulate_n_batch_grads
            self._last_acc_batch_start_num = max_b - max_b%acc + (1 if max_b%acc >= 1 else -acc + 1)
            self._normal_acc_batch_size    = acc*training_batch_sizes[0]
            self._last_acc_batch_size      = (max_b - self._last_acc_batch_start_num)*training_batch_sizes[0]
            self._last_acc_batch_size     += training_batch_sizes[1]
        self._starting_batch_num       = starting_batch_num #TODO: Should load from memory for existing metrics

        # Early stopping and save best parameters
        self._lower_is_better      = lower_is_better
        self._early_stop_tolerance = None
        self._patience             = None

        # Variables used to determine the length of the information to be printed.
        self._print_info_max_length = 0

        # Class information parameters. Used in metrics such as F1, MacroF1 and HarmonicMacroF1.
        self._classes_in_split_to_idx_map = {}
        self._idx_to_classes_in_split_map = []
        self._is_seen_class_indicator     = {}




    @abstractmethod
    def evaluate(self, batch_num, epoch_num, predictions, labels, training, evaluate_only=False):
        """
        This is the super method of the actual evaluation function. Here the batch/epoch number is accounted for.


        :param batch_num    : The number of the current minibatch, since the start of the training procedure.

        :param epoch_num    : The number of the current epoch, since the start of the training procedure.

        :param predictions  : Not used in the super method (this method). Refer to child classes.

        :param labels       : Not used in the super method (this method). Refer to child classes.

        :param training     : Determines if this evaluation is performed for training purposes, which means batch
                              information will be kept.

        :param evaluate_only: Not used in the super method (this method). Refer to child classes.


        :return: Nothing
        """

        # The model is training, so we increase the batch number.
        if (training):
            if (not bool(self._batch_history_x)):
                self._batch_history_x = [self._starting_batch_num]
            else:
                # This allows to identify the first batch of an accumulation.
                if ((batch_num - 1) % self._accumulate_n_batch_grads == 0):
                    self._batch_history_x.append(self._batch_history_x[-1] + 1)

        # Start of new epoch. Increase epoch number. Works for validation and test too, since they always have 1 batch.
        if (batch_num == 1):
            self._epoch_history_x.append(epoch_num)




    def set_convergence_parameters(self, parameters):# TODO: can probably move this to the construction phase
        """
        Sets the parameters used to determine if early stopping should occur.


        :param tolerance: The value that asserts whether two epochs have (pretty much) the same performance.

        :param patience : Number of successive epochs in which conditions need to be verified before early stopping.


        :return: Nothing
        """

        self._early_stop_tolerance = parameters[0]
        self._patience = parameters[1]




    @abstractmethod
    def update_classes_info(self, classes_in_split_to_idx_map=None, is_seen_class_indicator=None):
        """
        Sets the parameters used to determine if early stopping should occur.


        :param classes_in_split_to_idx_map: A dictionary that maps classes names to their corresponding index, as
                                            defined in the dataset being used.

        :param is_seen_class_indicator    : A dictionary that indicates whether a class belongs to the 'seen' set or
                                            the 'unseen' set.


        :return: Nothing
        """

        pass




    def evaluate_early_stopping(self):
        """
        Checks whether conditions for early stopping have been met or not.


        :return: Nothing
        """

        # Determines if there have been more epochs than the set patience.
        if (len(self._epoch_history) >= self._patience):
            for i in range(-self._patience, -1): # For each of the last 'self._patience' epochs.
                # Conditions for metrics where a lower score represents better performance.
                if (self._lower_is_better):
                    if (self._epoch_history[i] > self._epoch_history[i + 1] and
                            not np.isclose(self._epoch_history[i], self._epoch_history[i + 1],
                                           atol=self._early_stop_tolerance)):
                        return False

                # Conditions for metrics where a higher score represents better performance.
                else:
                    if (self._epoch_history[i] < self._epoch_history[i + 1] and
                            not np.isclose(self._epoch_history[i], self._epoch_history[i + 1],
                                           atol=self._early_stop_tolerance)):
                        return False
            return True
        return False




    def check_if_best(self, current_epoch_num):
        """
        Checks if the last epoch has been the one where the model has performed the best, under this metric.


        :param current_epoch_num: The current epoch number.


        :return: True if last epoch has been the one where the model has performed the best so far. False otherwise.
        """

        # Checks if the metric has been evaluated for the input epoch number.
        if (current_epoch_num in self._epoch_history_x):
            # Conditions for metrics where a lower score represents better performance.
            if (self._lower_is_better):
                # Determines if the index of the lowest score in the epoch history corresponds with the latest epoch.
                # if (np.argmin(self._epoch_history) == len(self._epoch_history) - 1):
                if (np.min(self._epoch_history) == self._epoch_history[-1]):
                    return True
                else:
                    return False

            # Conditions for metrics where a higher score represents better performance.
            else:
                # Determines if the index of the highest score in the epoch history corresponds with the latest epoch.
                # if (np.argmax(self._epoch_history) == len(self._epoch_history) - 1):
                if (np.max(self._epoch_history) == self._epoch_history[-1]):
                    return True
                else:
                    return False

        # The metric has NOT been evaluated for the input epoch number, so it cannot state that the model performed the
        # best for the input epoch number..
        else:
            return False




    def plot_statistics(self):
        """
        plot_statistics() docstring here

        :return:
        """

        # So far let's do only end of training plotting. Can later update to real time plotting
        raise NotImplementedError

    #         style.use('dark_background')
    #         if (self._evaluated_on_batches):
    #             plt.figure(figsize=(15,10))
    #             plt.plot(self._batch_history_x, self._batch_history, 'yo', markersize=2)
    #             plt.xlabel('Batch nº')
    #             plt.ylabel(self._name)
    #             plt.title('Gaussian colored noise')
    #             plt.rcParams.update({'font.size': 30})
    #             plt.savefig('test.png')




    def print_info(self, epoch, val, avg):
        """
        Creates a string with relevant information regarding this metric's short name and whether the values presented
        are a running average over the last epoch or not.


        :param val: Identifies if this specific instance of the metric is evaluated on the validation set.

        :param avg: Identifies whether this specific instance of the metric is evaluated on batches and this print
                    regards a running average of the performance on the training set.


        :return: A string that allows the user to identify the cases above. E.g.: '~A: ' would be returned for a
                 running average of the models accuracy on the train set.
        """

        return ("" if val or not epoch else ('~' if avg else ' ')) + self._short_name + ': '




    # For printing train set info while training
    def epoch_avg(self, step):
        """
        Determines the average of this metric for a specific epoch.


        :param step: Identifies the epoch for which the average is being computed.


        :return: The running average of the epoch or '-' if this metric was not evaluated for the specific epoch.
        """

        # Determines whether the metric has been run at all and if the step is valid for this metric.
        if ((len(self._epoch_history_x) > 0 and step == -1) or step in self._epoch_history_x):
            # Gets the epoch_history correct index for the input step.
            if (step in self._epoch_history_x):
                step = self._epoch_history_x.index(step)

            # If the epoch is not finished yet, the average needs to be computed.
            if (isinstance(self._epoch_history[step], np.ndarray)):
                return self._epoch_history[step][0] / self._epoch_history[step][1]
            # Otherwise it has already been computed and we only need to return the value.
            else:
                return self._epoch_history[step]

        # The metric has not been run for this specific step, so we return a placeholder.
        else:
            return "-"




    def _save_state(self):
        """
        Creates the base state dict for a metric class, which can then be extended by each individual class.


        :return: The metric's base state dict.
        """

        metric_state = {
            "batch_history"  : self._batch_history,
            "batch_history_x": self._batch_history_x,
            "epoch_history"  : self._epoch_history,
            "epoch_history_x": self._epoch_history_x
        }
        return metric_state




    def save(self, dir_path, file_type):
        """
        Saves the metric's state to file.


        :param dir_path : The path to the dir in which to save the metric's state.

        :param file_type: The type of the file to be saved. That is, a regular checkpoint, '.ckp', or the best state,
                          '.bst', so far.


        :return: Nothing
        """

        torch.save(self._save_state(), join_path([dir_path] + [self._name + file_type]))



    #TODO: MISSING SOME PARAMETERS
    @property
    def name(self):
        return self._name

    @property
    def short_name(self):
        return self._short_name

    @property
    def load_path(self):
        return self._load_path

    @property
    def load_epoch_or_best(self):
        return self._load_epoch_or_best

    @property
    def batch_history(self):
        return self._batch_history

    @property
    def batch_history_x(self):
        return self._batch_history_x

    @property
    def epoch_history(self):
        return self._epoch_history

    @property
    def epoch_history_x(self):
        return self._epoch_history_x

    @property
    def eval_only_history(self):
        return self._eval_only_history

    @property
    def evaluated_on_batches(self):
        return self._evaluated_on_batches

    @property
    def max_num_batches(self):
        return self._max_num_batches

    @property
    def accumulate_n_batch_grads(self):
        return self._accumulate_n_batch_grads

    @property
    def training_batch_sizes(self):
        return self._training_batch_sizes

    @property
    def last_acc_batch_start_num(self):
        return self._last_acc_batch_start_num

    @property
    def normal_acc_batch_size(self):
        return self._normal_acc_batch_size

    @property
    def last_acc_batch_size(self):
        return self._last_acc_batch_size

    @property
    def lower_is_better(self):
        return self._lower_is_better

    @property
    def early_stop_tolerance(self):
        return self._early_stop_tolerance

    @property
    def patience(self):
        return self._patience

    @property
    def print_info_max_length(self):
        return self._print_info_max_length

    @property
    def classes_in_split_to_idx_map(self):
        return self._classes_in_split_to_idx_map

    @property
    def idx_to_classes_in_split_map(self):
        return self._idx_to_classes_in_split_map

    @property
    def is_seen_class_indicator(self):
        return self._is_seen_class_indicator