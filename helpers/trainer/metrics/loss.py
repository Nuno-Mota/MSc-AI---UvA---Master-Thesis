"""
##################################################
##################################################
## This file implements the (general) Loss      ##
## metric, which will, for each instance, be    ##
## associated with a specific loss function.    ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import numpy as np
import math

# *** Own modules imports. *** #

from helpers.trainer.metrics.metric import Metric





#################
##### CLASS #####
#################

class Loss(Metric):
    """
    The Loss metric is used to evaluate the loss (for a specified loss function) of a model.
    """

    def __init__(self, loss_function, evaluated_on_batches, max_num_batches, accumulate_n_batch_grads,
                 training_batch_sizes, load_path, load_epoch_or_best, starting_batch_num=1, _DEBUG=False):
        """
        Instantiates a Loss metric object.


        :param loss_function       : The loss function that is called to compute the loss.

        :param evaluated_on_batches: Determines whether this specific instance of a Loss metric is evaluated for
                                     minibatches. Used for plotting.

        :param max_num_batches     : The maximum number of batches in an epoch. Used to aggregate epoch information.

        :param load_path           : The path that leads to a saved state of an instance of the Loss metric.

        :param load_epoch_or_best  : This parameter identifies whether to load the state associated with a specific
                                     epoch or the state of the epoch in which the model performed the best.

        :param starting_batch_num  : Some metrics can be added to the MetricsManager after the model has already been
                                     trained for a few epochs. In this case the Loss is always evaluated, so it
                                     defaults to 1.
        """

        short_name = 'L'
        if (loss_function.__class__.__name__ in ['RE_BoW_LOSS']): #TODO: create dict with all the losses that are ELBOs
            short_name = 'ELBO'

        super().__init__(short_name, evaluated_on_batches, max_num_batches, accumulate_n_batch_grads,
                         training_batch_sizes, True, load_path, load_epoch_or_best, starting_batch_num)

        # Loss function specifics
        self._loss_function = loss_function
        self._loss = None # The actual loss that is used to backpropagate with autograd (Only when the specific
                          # instance of this metric is associated with the training set).

        # Variables used to determine the length of the information to be printed.
        self._max_loss_len = 0 if self._metric_state is None else self._metric_state['max_loss_len']
        self._print_info_max_length = 7 + self._max_loss_len + len(short_name)

        # Variables concerning special loss elements to be tracked when debugging.
        self._DEBUG = _DEBUG
        self._annealing_batch_x = [] if self._metric_state is None else self._metric_state['annealing_batch_x']
        self._annealing_batch   = [] if self._metric_state is None else self._metric_state['annealing_batch']
        self._max_kl_len        = 0  if self._metric_state is None else self._metric_state['max_kl_len']
        self._kl_batch_x        = [] if self._metric_state is None else self._metric_state['kl_batch_x']
        self._kl_batch          = [] if self._metric_state is None else self._metric_state['kl_batch']
        self._kl_epoch_x        = [] if self._metric_state is None else self._metric_state['kl_epoch_x']
        self._kl_epoch          = [] if self._metric_state is None else self._metric_state['kl_epoch']
        self._nlue_batch_x      = [] if self._metric_state is None else self._metric_state['nlue_batch_x']
        self._nlue_batch        = [] if self._metric_state is None else self._metric_state['nlue_batch']
        self._nlue_epoch_x      = [] if self._metric_state is None else self._metric_state['nlue_epoch_x']
        self._nlue_epoch        = [] if self._metric_state is None else self._metric_state['nlue_epoch']
        if (self._DEBUG):
            self._print_info_max_length += 24 + self._max_kl_len + self._max_loss_len




    def evaluate(self, batch_num, epoch_num, predictions, labels, training, evaluate_only=False):
        """
        This method computes de loss for the given predictions/labels and registers the results.


        :param batch_num    : The number of the current minibatch, since the start of the training procedure.

        :param epoch_num    : The number of the current epoch, since the start of the training procedure.

        :param predictions  : The model's predictions.

        :param labels       : The correct output for the corresponding data instances.

        :param training     : Determines if this evaluation is performed for training purposes, which means batch
                              information will be kept.

        :param evaluate_only: Used to determine whether to register the computed values or not. If True, then it is
                              assumed this was a singular evaluation, not within the training procedure.


        :return: Nothing
        """

        # TODO: Some losses have annealing components. Should those be set to the full value when evaluating?
        # TODO: Can use 'if (not training or evaluate_only)' and toggle the value in the Loss object.

        # Compute the tracked loss elements. THESE ARE NOT AVERAGED, THEY ARE SUMMED
        if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
            # ELBO, Kullback–Leibler divergence, Negative Log Uniform Evidence
            self._loss, self._KL, self._NLUE = self._loss_function(predictions, labels)
        else:
            # NLL
            self._loss = self._loss_function(predictions, labels)[0]

        # Update the batch size.
        batch_size = predictions[0].shape[0]


        # If this IS NOT a singular evaluation, decoupled from the training procedure.
        if (not evaluate_only):
            # Register batch/epoch number information and set self._current_batch_size to 0, if necessary,
            # using the super class' method.
            super().evaluate(batch_num, epoch_num, None, None, training)

            # If this IS the beginning of a new epoch, start registering the epoch's sum of losses and batches' sizes.
            if (batch_num == 1):
                self._epoch_history.append(np.array([self._loss.item(), batch_size]))
                if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
                    self._kl_epoch_x.append(self._epoch_history_x[-1])
                    self._kl_epoch.append(np.array([self._KL.item(), batch_size]))
                    self._nlue_epoch_x.append(self._epoch_history_x[-1])
                    self._nlue_epoch.append(np.array([self._NLUE.item(), batch_size]))
            # If this IS NOT the beginning of a new epoch, update the epoch's sum of losses and batches' sizes.
            else:
                self._epoch_history[-1] += np.array([self._loss.item(), batch_size])
                if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
                    self._kl_epoch[-1] += np.array([self._KL.item(), batch_size])
                    self._nlue_epoch[-1] += np.array([self._NLUE.item(), batch_size])

            # If the epoch has completed.
            if (batch_num == self._max_num_batches):
                # Compute the final running average for this epoch.
                self._epoch_history[-1] = self._epoch_history[-1][0] / self._epoch_history[-1][1]
                if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
                    self._kl_epoch[-1] = self._kl_epoch[-1][0] / self._kl_epoch[-1][1]
                    self._nlue_epoch[-1] = self._nlue_epoch[-1][0] / self._nlue_epoch[-1][1]

                # If this was epoch 0 (pre-evaluation) compute how many digits the loss will have, for printing
                # purposes. Assumes that loss will never be bigger than the pre-evaluation loss.
                # For _DEBUG mode we assume the print length values will be the same as the loss. TODO: Works?
                if (epoch_num == 0):
                    self._max_loss_len = len(str(math.floor(self._epoch_history[-1])))
                    self._print_info_max_length += self._max_loss_len
                    if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
                        self._max_kl_len = len(str(math.floor(self._kl_epoch[-1]))) + 1
                        self._print_info_max_length += self._max_kl_len + self._max_loss_len

            # If the evaluation is performed on the train set (while training, (see evaluate_only)) register batch loss
            # value.
            if (training):
                # Normalise the loss correctly.
                if (self._loss_function.__class__.__name__ in ['MeanSquaredErrorLoss']):
                    # The accumulation factor when pre-training the embedding layer is always 1!
                    self._loss /= batch_size
                else:
                    if (batch_num < self._last_acc_batch_start_num):
                        self._loss /= self._normal_acc_batch_size
                        if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
                            self._KL /= self._normal_acc_batch_size
                            self._NLUE /= self._normal_acc_batch_size
                    else:
                        self._loss /= self._last_acc_batch_size
                        if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
                            self._KL /= self._last_acc_batch_size
                            self._NLUE /= self._last_acc_batch_size

                # This allows to identify the first batch of an accumulation.
                if ((batch_num - 1) % self._accumulate_n_batch_grads == 0):
                    self._batch_history.append(self._loss.item())
                    if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
                        self._kl_batch_x.append(self._batch_history_x[-1])
                        self._kl_batch.append(self._KL.item())
                        self._nlue_batch_x.append(self._batch_history_x[-1])
                        self._nlue_batch.append(self._NLUE.item())
                        self._annealing_batch_x.append(self._batch_history_x[-1])
                        self._annealing_batch.append(self._loss_function.annealing_factor())
                # Everything else is accumulated
                else:
                    self._batch_history[-1] += self._loss.item()
                    if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
                        self._kl_batch[-1] += self._KL.item()
                        self._nlue_batch[-1] += self._NLUE.item()

        # If this IS a singular evaluation, decoupled from the training procedure.
        else:
            # If this IS the beginning of a new evaluation, start registering the
            # evaluation's sum of losses and batches' sizes.
            if (batch_num == 1):
                self._eval_only_history = np.array([self._loss.item(), batch_size])
            # If this IS NOT the beginning of a new evaluation, update the
            # evaluation's sum of losses and batches' sizes.
            else:
                self._eval_only_history += np.array([self._loss.item(), batch_size])

            # If the evaluation has completed.
            if (batch_num == self._max_num_batches):
                # Compute the final loss for this specific instance's dataset split.
                self._eval_only_history = self._eval_only_history[0] / self._eval_only_history[1]

                # As this was a singular evaluation, if the model has never been evaluated using the loss,
                # compute how many digits the loss will have, for printing purposes.
                if (self._max_loss_len == 0):
                    self._max_loss_len = len(str(math.floor(self._eval_only_history)))
                    self._print_info_max_length += self._max_loss_len




    def print_info(self, val=False, epoch=True, avg=True, step=-1, evaluate_only=False):
        """
        This method constructs a string that contains the relevant information for a specific batch/epoch number.


        :param val          : Identifies if this specific instance is evaluated on the validation set.

        :param epoch        : Whether to print information regarding an epoch or a batch.

        :param avg          : Identifies whether this specific instance is evaluated on batches and this print regards
                              a running average of the performance on the training set.

        :param step         : Identifies the batch/epoch whose value is being printed.

        :param evaluate_only: Identifies whether this is a singular evaluation, decoupled from training.


        :return: A string containing the relevant information. E.g.: "~L: 1.87033", for avg=True
        """

        # Normal padding and formatting.
        format_padding = ''
        format_terminology = '.3f}'
        if (self._DEBUG):
            format_padding_debug = ''
            format_terminology_debug = '.3f}'
            format_terminology_debug_ann = '.4f}'

        # Padding and formatting for epochs for which this Loss metric instance was not evaluated.
        if (epoch and step != -1 and step not in self._epoch_history_x and not evaluate_only):
            format_padding = '-^'
            format_terminology = '}'
            if (self._DEBUG and step not in self._kl_epoch_x):
                format_padding_debug = '-^'
                format_terminology_debug = '}'
                format_terminology_debug_ann = '}'

        # Final formatting and string construction.
        format_type = '{:' + format_padding + str(self._max_loss_len + 4) + format_terminology
        if (self._DEBUG):
            format_type_debug_ann  = '{:' + format_padding_debug + str(1 + 5) + format_terminology_debug_ann
            format_type_debug_kl   = '{:' + format_padding_debug + str(self._max_kl_len + 4) + format_terminology_debug
            format_type_debug_nlue = '{:' + format_padding_debug + str(self._max_loss_len + 4) + format_terminology_debug
        avg = super().print_info(epoch, val, avg)
        not_eval_loss = self.epoch_avg(step) if epoch else self._batch_history[step]
        loss = not_eval_loss if not evaluate_only else self._eval_only_history
        print_str = avg + format_type.format(loss)
        if (self._DEBUG and self._loss_function.__class__.__name__ in ['RE_BoW_LOSS']):
            print_str += " ("
            if (not epoch):
                print_str += "An: " + format_type_debug_ann.format(self._annealing_batch[step]) + ","
            not_eval_kl = self.special_element_avg(step, self._kl_epoch_x, self._kl_epoch) if epoch else self._kl_batch[step]
            print_str += (avg[:1] if epoch else " ") + "KL: " + format_type_debug_kl.format(not_eval_kl) + ", "
            not_eval_nlue = self.special_element_avg(step, self._nlue_epoch_x, self._nlue_epoch) if epoch else self._nlue_batch[step]
            print_str += (avg[:1] if epoch else "") + "NLL: " + format_type_debug_nlue.format(not_eval_nlue) + ")"

        return print_str




    # For printing special loss terms info while training.
    def special_element_avg(self, step, special_element_x, special_element):
        """
        Determines the average of this metric for a specific epoch.


        :param step: Identifies the epoch for which the average is being computed.


        :return: The running average of the epoch or '-' if this metric was not evaluated for the specific epoch.
        """

        # Determines whether the metric has been run at all and if the step is valid for this metric.
        if ((len(special_element_x) > 0 and step == -1) or step in special_element_x):
            # Gets the epoch_history correct index for the input step.
            if (step in special_element_x):
                step = special_element_x.index(step)

            # If the epoch is not finished yet, the average needs to be computed.
            if (isinstance(special_element[step], np.ndarray)):
                return special_element[step][0] / special_element[step][1]
            # Otherwise it has already been computed and we only need to return the value.
            else:
                return special_element[step]

        # The metric has not been run for this specific step, so we return a placeholder.
        else:
            return "-"




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




    # OVERRIDE DUE TO extra variables
    def save(self, dir_path, file_type):
        """
        Saves the Loss' state to file.


        :param dir_path : The path to the dir in which to save the Loss' state.

        :param file_type: The type of the file to be saved. That is, a regular checkpoint, '.ckp', or the best state,
                          '.bst', so far.


        :return: Nothing
        """

        # Get base state dict, common to all metrics.
        state = super()._save_state()


        # Extend the base state dict with the extra information relevant for the Loss metric.
        state["max_loss_len"]      = self._max_loss_len
        state["annealing_batch_x"] = self._annealing_batch_x
        state["annealing_batch"]   = self._annealing_batch
        state["max_kl_len"]        = self._max_kl_len
        state["kl_batch_x"]        = self._kl_batch_x
        state["kl_batch"]          = self._kl_batch
        state["kl_epoch_x"]        = self._kl_epoch_x
        state["kl_epoch"]          = self._kl_epoch
        state["nlue_batch_x"]      = self._nlue_batch_x
        state["nlue_batch"]        = self._nlue_batch
        state["nlue_epoch_x"]      = self._nlue_epoch_x
        state["nlue_epoch"]        = self._nlue_epoch

        torch.save(state, dir_path + self._name + file_type)




    @property
    def loss_function(self):
        return self._loss_function

    @property
    def loss(self):
        return self._loss

    @property
    def max_loss_len(self):
        return self._max_loss_len