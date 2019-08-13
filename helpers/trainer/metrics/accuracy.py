"""
##################################################
##################################################
## This file implements the Accuracy metric,    ##
## which measures the correct number of         ##
## predictions in a classification setting.     ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import numpy as np
import pickle

# *** Own modules imports. *** #

from helpers.trainer.metrics.metric import Metric





#################
##### CLASS #####
#################

class Accuracy(Metric):
    """
    The Accuracy metric is used to evaluate the percentage of correct answers that a model gets in a classification
    task.
    """

    def __init__(self, evaluated_on_batches, max_num_batches, accumulate_n_batch_grads, training_batch_sizes,
                 load_path, load_epoch_or_best, starting_batch_num=1):
        """
        Initiates an Accuracy metric object.


        :param evaluated_on_batches: Determines whether this specific instance of an Accuracy metric is evaluated for
                                     minibatches. Used for plotting.

        :param max_num_batches     : The maximum number of batches in an epoch. Used to aggregate epoch information.

        :param load_path           : The path that leads to a saved state of an instance of the Accuracy metric.

        :param load_epoch_or_best  : This parameter identifies whether to load the state associated with a specific
                                     epoch or the state of the epoch in which the model performed the best.

        :param starting_batch_num  : Some metrics can be added to the MetricsManager after the model has already been
                                     trained for a few epochs. This identifies the first batch in which this particular
                                     instance was initiated.
        """

        super().__init__("A", evaluated_on_batches, max_num_batches, accumulate_n_batch_grads, training_batch_sizes,
                         False, load_path, load_epoch_or_best, starting_batch_num)

        # Variables used to determine the length of the information to be printed.
        self._print_info_max_length = 11
        # if (not self._evaluated_on_batches):
        #     self._correlation_per_epoch = {}




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
        predictions = predictions[0]
        labels = labels[0][0]
        # if (batch_num == 1):
        #     if (not self._evaluated_on_batches):
        #         self._correlation_per_epoch[str(epoch_num)] = torch.zeros(predictions.shape[1],
        #                                                                   predictions.shape[1],
        #                                                                   device=predictions.device)
        #
        # if (not self._evaluated_on_batches and batch_num == self._max_num_batches):
        #     with open("correlation_matrices_cpu_esim_enc_long.crm", 'wb') as f:
        #         pickle.dump({epoch: self._correlation_per_epoch[epoch].cpu() for epoch in self._correlation_per_epoch}, f)


        # Update the batch size.
        batch_size = predictions.shape[0]

        # Compute the number of correct classifications.
        num_correct = (labels.eq(torch.argmax(predictions, dim=-1).long())).sum().item()

        # If this IS NOT a singular evaluation, decoupled from the training procedure.
        if (not evaluate_only):
            # Register batch/epoch number information using the super class' method.
            super().evaluate(batch_num, epoch_num, None, None, training)

            # If this IS the beginning of a new epoch, start registering the epoch's sum of number of correct
            # classifications and batches' sizes.
            if (batch_num == 1):
                self._epoch_history.append(np.array([num_correct, batch_size]))
            # If this IS NOT the beginning of a new epoch, update the epoch's sum of number of correct
            # classifications and batches' sizes.
            else:
                self._epoch_history[-1] += np.array([num_correct, batch_size])

            # If the epoch has completed.
            if (batch_num == self._max_num_batches):
                # Compute the final running average for this epoch.
                self._epoch_history[-1] = 100 * self._epoch_history[-1][0] / self._epoch_history[-1][1]

            # If the evaluation is performed on the train set (while training, (see evaluate_only))
            # register batch accuracy value.
            if (training):
                # Normalise the accuracy correctly.
                if (batch_num < self._last_acc_batch_start_num):
                    accuracy = 100*num_correct/self._normal_acc_batch_size
                else:
                    accuracy = 100*num_correct/self._last_acc_batch_size

                # This allows to identify the first batch of an accumulation.
                if ((batch_num - 1) % self._accumulate_n_batch_grads == 0):
                    self._batch_history.append(accuracy)
                # Everything else is accumulated
                else:
                    self._batch_history[-1] += accuracy

        # If this IS a singular evaluation, decoupled from the training procedure.
        else:
            # If this IS the beginning of a new evaluation, start registering the
            # evaluation's number of correct classifications and batches' sizes.
            if (batch_num == 1):
                self._eval_only_history = np.array([num_correct, batch_size])
            # If this IS NOT the beginning of a new evaluation, update the
            # evaluation's number of correct classifications and batches' sizes.
            else:
                self._eval_only_history += np.array([num_correct, batch_size])

            # If the evaluation has completed.
            if (batch_num == self._max_num_batches):
                # Compute the final accuracy for this specific instance's dataset split.
                self._eval_only_history = 100 * self._eval_only_history[0] / self._eval_only_history[1]




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

        # Padding and formatting for epochs for which this Accuracy metric instance was not evaluated.
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
            if (isinstance(self._epoch_history[step], np.ndarray)):
                return 100 * self._epoch_history[step][0] / self._epoch_history[step][1]
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