"""
##################################################
##################################################
## This file implements the F1 metric, which    ##
## computes the Harmonic mean between recall    ##
## and precision, for a single class.           ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import numpy as np
import re


# *** Own modules imports. *** #

from helpers.trainer.metrics.metric import Metric
from helpers.general_helpers import print_warning, join_path





#################
##### CLASS #####
#################

class F1(Metric):
    """
    The F1 score of a class is the harmonic mean between precision and recall.
    """

    def __init__(self, evaluated_on_batches, max_num_batches, accumulate_n_batch_grads, training_batch_sizes,
                 load_path, load_epoch_or_best, starting_batch_num=1, optional_class_name=''):
        """
        Initiates a F1 metric object.


        :param evaluated_on_batches: Determines whether this specific instance of a F1 metric is evaluated for
                                     minibatches. Used for plotting.

        :param max_num_batches     : The maximum number of batches in an epoch. Used to aggregate epoch information.

        :param load_path           : The path that leads to a saved state of an instance of the F1 metric.

        :param load_epoch_or_best  : This parameter identifies whether to load the state associated with a specific
                                     epoch or the state of the epoch in which the model performed the best.

        :param starting_batch_num  : Some metrics can be added to the MetricsManager after the model has already been
                                     trained for a few epochs. This identifies the first batch in which this particular
                                     instance was initiated.
        """

        super().__init__("F1", evaluated_on_batches, max_num_batches, accumulate_n_batch_grads, training_batch_sizes,
                         False, load_path, load_epoch_or_best, starting_batch_num)


        # Variables specific to the F1 metric.
        self._optional_class_name = optional_class_name if self._metric_state is None else self._metric_state['optional_class_name']

        self._precision_batch_x   = [] if self._metric_state is None else self._metric_state['precision_batch_x']
        self._precision_batch     = [] if self._metric_state is None else self._metric_state['precision_batch']
        self._precision_epoch_x   = [] if self._metric_state is None else self._metric_state['precision_epoch_x']
        self._precision_epoch     = [] if self._metric_state is None else self._metric_state['precision_epoch']
        self._recall_batch_x      = [] if self._metric_state is None else self._metric_state['recall_batch_x']
        self._recall_batch        = [] if self._metric_state is None else self._metric_state['recall_batch']
        self._recall_epoch_x      = [] if self._metric_state is None else self._metric_state['recall_epoch_x']
        self._recall_epoch        = [] if self._metric_state is None else self._metric_state['recall_epoch']

        # Used to (only) evaluate the model's performance, so that no state of the singular evaluation is kept.
        self._precision_eval_only_history = None
        self._recall_eval_only_history = None


        # Variables used to determine the length of the information to be printed.
        self._print_info_max_length = 12




    def evaluate(self, batch_num, epoch_num, predictions, labels, training, evaluate_only=False):
        """
        This method computes de accuracy for the given predictions/labels and registers the results.


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

        # Get correct components of both predictions and labels.
        predictions = predictions[0] # This will be a vector of zeros and ones.
        labels = labels[0][0]        # And so will this.

        # Compute the number of correct classifications.
        batch_num_matched  = torch.sum(torch.tensordot(predictions.float(), labels.float(), dims=1))

        # If this IS NOT a singular evaluation, decoupled from the training procedure.
        if (not evaluate_only):
            # Register batch/epoch number information using the super class' method.
            super().evaluate(batch_num, epoch_num, None, None, training)

            # If this IS the beginning of a new epoch, start registering the epoch's sum of number of correct
            # classifications and batches' sizes.
            if (batch_num == 1):
                # In the case of F1 we don't register straight away the epoch history, as it depends on both the
                # precision and recall. When we do need it (or when the epoch is finished) we compute it from those two.
                # Precision.
                self._precision_epoch_x.append(self._epoch_history_x[-1])
                self._precision_epoch.append(np.array([batch_num_matched.item(), predictions.sum().item()]))
                # Recall.
                self._recall_epoch_x.append(self._epoch_history_x[-1])
                self._recall_epoch.append(np.array([batch_num_matched.item(), labels.sum().item()]))

            # If this IS NOT the beginning of a new epoch, update the epoch's precision, recall and corresponding
            # number of elements.
            else:
                self._precision_epoch[-1] += np.array([batch_num_matched.item(), predictions.sum().item()])
                self._recall_epoch[-1]    += np.array([batch_num_matched.item(), labels.sum().item()])

            # If the epoch has completed.
            if (batch_num == self._max_num_batches):
                if (self._recall_epoch[-1][1] == 0):
                    print_warning("F1-score: '" + self._optional_class_name + "' had no instances registered during " +
                                  "this epoch")

                # Compute the final running average for this epoch.
                # Precision.
                if (self._precision_epoch[-1][1] == 0):
                    self._precision_epoch[-1] = 0
                else:
                    self._precision_epoch[-1] = self._precision_epoch[-1][0] / self._precision_epoch[-1][1]

                # Recall.
                if (self._recall_epoch[-1][0] == 0):
                    self._recall_epoch[-1] = 0
                else:
                    self._recall_epoch[-1] = self._recall_epoch[-1][0] / self._recall_epoch[-1][1]

                # F1-score.
                if (self._precision_epoch[-1] == 0 and self._recall_epoch[-1] == 0):
                    self._epoch_history.append(0)
                else:
                    self._epoch_history.append(100*2*self._precision_epoch[-1]*self._recall_epoch[-1])
                    self._epoch_history[-1] /= self._precision_epoch[-1] + self._recall_epoch[-1]

            # If the evaluation is performed on the train set (while training, (see evaluate_only))
            # register batch precision/recall values.
            if (training):
                # This allows to identify the first batch of an accumulation.
                if ((batch_num - 1) % self._accumulate_n_batch_grads == 0):
                    # Precision.
                    self._precision_batch_x.append(self._batch_history_x[-1])
                    self._precision_batch.append(np.array([batch_num_matched.item(), predictions.sum().item()]))
                    # Recall.
                    self._recall_batch_x.append(self._batch_history_x[-1])
                    self._recall_batch.append(np.array([batch_num_matched.item(), labels.sum().item()]))
                # Everything else is accumulated
                else:
                    self._precision_batch[-1] += np.array([batch_num_matched.item(), predictions.sum().item()])
                    self._recall_batch[-1] += np.array([batch_num_matched.item(), labels.sum().item()])


                # If this is the last batch of an accumulation, compute the batch F1 score.
                if (batch_num % self._accumulate_n_batch_grads == 0):
                    if (self._recall_batch[-1][1] != 0):
                        # Precision.
                        if (self._precision_batch[-1][1] == 0):
                            self._precision_batch[-1] = 0
                        else:
                            self._precision_batch[-1] = self._precision_batch[-1][0] / self._precision_batch[-1][1]

                        # Recall.
                        if (self._recall_batch[-1][0] == 0):
                            self._recall_batch[-1] = 0
                        else:
                            self._recall_batch[-1] = self._recall_batch[-1][0] / self._recall_batch[-1][1]

                        # F1-score.
                        if (self._precision_batch[-1] == 0 and self._recall_batch[-1] == 0):
                            self._batch_history.append(0)
                        else:
                            self._batch_history.append(100 * 2 * self._precision_batch[-1] * self._recall_batch[-1])
                            self._batch_history[-1] /= self._precision_batch[-1] + self._recall_batch[-1]
                    else:
                        self._batch_history.append(False)


        # If this IS a singular evaluation, decoupled from the training procedure.
        else:
            # If this IS the beginning of a new evaluation, start registering the
            # evaluation's number of correct classifications and batches' sizes.
            if (batch_num == 1):
                self._precision_eval_only_history = np.array([batch_num_matched.item(), predictions.sum().item()])
                self._recall_eval_only_history    = np.array([batch_num_matched.item(), labels.sum().item()])
            # If this IS NOT the beginning of a new evaluation, update the
            # evaluation's number of correct classifications and batches' sizes.
            else:
                self._precision_eval_only_history += np.array([batch_num_matched.item(), predictions.sum().item()])
                self._recall_eval_only_history    += np.array([batch_num_matched.item(), labels.sum().item()])

            # If the evaluation has completed.
            if (batch_num == self._max_num_batches):
                # Compute the final F1-score for this specific instance's dataset split.
                # Precision.
                if (self._precision_eval_only_history[1] == 0):
                    self._precision_eval_only_history = 0
                else:
                    self._precision_eval_only_history = self._precision_eval_only_history[0] / self._precision_eval_only_history[1]

                # Recall.
                if (self._recall_eval_only_history[0] == 0):
                    self._recall_eval_only_history = 0
                else:
                    self._recall_eval_only_history = self._recall_eval_only_history[0] / self._recall_eval_only_history[1]

                # F1-score.
                if (self._recall_eval_only_history == 0 and self._recall_eval_only_history == 0):
                    self._eval_only_history = 0
                else:
                    self._eval_only_history = 100 * 2 * self._precision_eval_only_history * self._recall_eval_only_history
                    self._eval_only_history /= self._precision_eval_only_history + self._recall_eval_only_history




    def print_info(self, val=False, epoch=True, avg=True, step=-1, evaluate_only=False):
        """
        This method constructs a string that contains the relevant information for a specific batch/epoch number.


        :param val          : Identifies if this specific instance is evaluated on the validation set.

        :param epoch        : Whether to print information regarding an epoch or a batch.

        :param avg          : Identifies whether this specific instance is evaluated on batches and this print regards
                              a running average of the performance on the training set.

        :param step         : Identifies the batch/epoch whose value is being printed.

        :param evaluate_only: Identifies whether this is a singular evaluation, decoupled from training.


        :return: A string containing the relevant information. E.g.: "~A:  58.98%", for avg=True
        """

        # Normal padding and formatting.
        format_padding = ''
        format_terminology = '.2f}'

        # Padding and formatting for epochs for which this F1 metric instance was not evaluated.
        if (epoch and step != -1 and step not in self._epoch_history_x):
            format_padding = '-^'
            format_terminology = '}'

        # Final formatting and string construction.
        format_type = '{:' + format_padding + str(6 if format_terminology == '.2f}' else 7) +\
                      format_terminology + ('%' if format_terminology == '.2f}' else "")
        return super().print_info(epoch, val, avg) +\
               (format_type.format((self.epoch_avg(step) if epoch else self._batch_history[step]) if not
               evaluate_only else self._eval_only_history))




    # OVERRIDE DUE TO % --- For printing train set info while training
    def epoch_avg(self, step):
        """
        Determines the average of the accuracy for a specific epoch.


        :param step: Identifies the epoch for which the average is being computed.


        :return: The running average of the epoch or '-' if this metric was not evaluated for the specific epoch.
        """

        # Determines whether the metric has been run at all and if the step is valid for this metric.
        if ((len(self._epoch_history_x) > 0 and step == -1) or step in self._epoch_history_x):
            # Gets the epoch_history correct index for the input step.
            if (step in self._epoch_history_x):
                step = self._epoch_history_x.index(step)

            # If the epoch is not finished yet, the average needs to be computed.
            if (isinstance(self._precision_epoch[step], np.ndarray)):
                if (self._recall_epoch[step][1] != 0):
                    # Precision.
                    if (self._precision_epoch[step][1] == 0):
                        running_avg_precision = 0
                    else:
                        running_avg_precision = self._precision_epoch[step][0] / self._precision_epoch[step][1]

                    # Recall.
                    if (self._recall_epoch[step][0] == 0):
                        running_avg_recall = 0
                    else:
                        running_avg_recall = self._recall_epoch[-1][0] / self._recall_epoch[step][1]

                    # F1-score.
                    if (running_avg_precision == 0 and running_avg_recall == 0):
                        running_avg_F1 = 0
                    else:
                        running_avg_F1 = 100 * 2 * running_avg_precision * running_avg_recall
                        running_avg_F1 /= running_avg_precision + running_avg_recall

                    return running_avg_F1
                else:
                    return "-"
            # Otherwise it has already been computed and we only need to return the value.
            else:
                return self._epoch_history[step]

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
        Saves the F1's state to file.


        :param dir_path : The path to the dir in which to save the F1's state.

        :param file_type: The type of the file to be saved. That is, a regular checkpoint, '.ckp', or the best state,
                          '.bst', so far.


        :return: Nothing
        """

        # Get base state dict, common to all metrics.
        state = super()._save_state()


        # Extend the base state dict with the extra information relevant for the Loss metric.
        state["optional_class_name"] = self._optional_class_name
        state["precision_batch_x"]   = self._precision_batch_x
        state["precision_batch"]     = self._precision_batch
        state["precision_epoch_x"]   = self._precision_epoch_x
        state["precision_epoch"]     = self._precision_epoch
        state["recall_batch_x"]      = self._recall_batch_x
        state["recall_batch"]        = self._recall_batch
        state["recall_epoch_x"]      = self._recall_epoch_x
        state["recall_epoch"]        = self._recall_epoch

        optional_class_name_part = re.sub('/', '-', re.sub(' ', '_', self._optional_class_name))
        optional_class_name_part += ('_' if self._optional_class_name != '' else '')
        torch.save(state, join_path([dir_path] + [optional_class_name_part + self._name + file_type]))




    @property
    def precision_batch_x(self):
        return self._precision_batch_x

    @property
    def precision_batch(self):
        return self._precision_batch

    @property
    def precision_epoch_x(self):
        return self._precision_epoch_x

    @property
    def precision_epoch(self):
        return self._precision_epoch

    @property
    def recall_batch_x(self):
        return self._recall_batch_x

    @property
    def recall_batch(self):
        return self._recall_batch

    @property
    def recall_epoch_x(self):
        return self._recall_epoch_x

    @property
    def recall_epoch(self):
        return self._recall_epoch