"""
##################################################
##################################################
## This file implements the harmonicMacro F1    ##
## metric, which measures harmonic mean between ##
## the MacroF1-score of 'seen' and 'unseen'     ##
## classes. For more information on the MacroF1 ##
## metric, consult the corresponding class.     ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import os


# *** Own modules imports. *** #

from helpers.trainer.metrics.metric import Metric
from helpers.trainer.metrics.macroF1 import MacroF1
from helpers.general_helpers import print_warning, join_path





#################
##### CLASS #####
#################

class HarmonicMacroF1(Metric):
    """
    The HarmonicMacroF1 metric is the harmonic mean of the Macro F1 scores of both 'seen' and 'unseen' classes.
    See MacroF1's corresponding class for more info.
    """

    def __init__(self, evaluated_on_batches, max_num_batches, accumulate_n_batch_grads, training_batch_sizes,
                 load_path, load_epoch_or_best, starting_batch_num=1):
        """
        Initiates a HarmonicMacroF1 metric object.


        :param evaluated_on_batches: Determines whether this specific instance of a HarmonicMacroF1 metric is evaluated
                                    for minibatches. Used for plotting.

        :param max_num_batches     : The maximum number of batches in an epoch. Used to aggregate epoch information.

        :param load_path           : The path that leads to a saved state of an instance of the HarmonicMacroF1 metric.

        :param load_epoch_or_best  : This parameter identifies whether to load the state associated with a specific
                                     epoch or the state of the epoch in which the model performed the best.

        :param starting_batch_num  : Some metrics can be added to the MetricsManager after the model has already been
                                     trained for a few epochs. This identifies the first batch in which this particular
                                     instance was initiated.
        """

        super().__init__("HMF1", evaluated_on_batches, max_num_batches, accumulate_n_batch_grads, training_batch_sizes,
                         False, load_path, load_epoch_or_best, starting_batch_num)

        # Variables specific to the HarmonicMacroF1 metric.
        load_path_seen   = None if load_path is None else (join_path([load_path] + [self._name + '_sub_metrics'] + ['seen']))
        load_path_unseen = None if load_path is None else (join_path([load_path] + [self._name + '_sub_metrics'] + ['unseen']))
        self._seen_unseen_MacroF1 = {'seen':   MacroF1(evaluated_on_batches, max_num_batches, accumulate_n_batch_grads,
                                                       training_batch_sizes, load_path_seen, load_epoch_or_best,
                                                       starting_batch_num, 'seen'),
                                     'unseen': MacroF1(evaluated_on_batches, max_num_batches, accumulate_n_batch_grads,
                                                       training_batch_sizes, load_path_unseen, load_epoch_or_best,
                                                       starting_batch_num, 'unseen'),
                                    }

        self._seen_classes_to_idx   = None if self._metric_state is None else self._metric_state['seen_classes_to_idx']
        self._unseen_classes_to_idx = None if self._metric_state is None else self._metric_state['unseen_classes_to_idx']

        # Variables used to determine the length of the information to be printed.
        self._print_info_max_length = 14 if self._metric_state is None else self._metric_state['print_info_max_length']




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

        # Increment size of printed string, since the metric is evaluated.
        if (epoch_num == 0 and batch_num == 1):
            self._print_info_max_length += 19

        # Compute the MacroF1-score for each 'seen'/'unseen' set.
        for _, MF1 in self.seen_unseen_MacroF1.items():
            MF1.evaluate(batch_num, epoch_num, predictions, labels, training, evaluate_only)


        # If this IS NOT a singular evaluation, decoupled from the training procedure.
        if (not evaluate_only):
            # Register batch/epoch number information using the super class' method.
            super().evaluate(batch_num, epoch_num, None, None, training)

            # HarmonicMacroF1 is not like most of the other metrics. It just calls the necessary components computed by
            # the lower-level MacroF1 metrics. As such, it only registers epoch information (on itself) at the end of
            # the epoch.

            # If the epoch has completed.
            if (batch_num == self._max_num_batches):
                # Compute the final running average for this epoch.
                seen_MF1   = self._seen_unseen_MacroF1['seen'].epoch_history[-1]/100.0
                unseen_MF1 = self._seen_unseen_MacroF1['unseen'].epoch_history[-1]/100.0
                if (seen_MF1 == 0 or unseen_MF1 == 0):
                    self._epoch_history.append(0.)
                else:
                    self._epoch_history.append(100*2*seen_MF1*unseen_MF1/(seen_MF1+unseen_MF1))

            # If the evaluation is performed on the train set (while training, (see evaluate_only))
            # register accuracy value, if it is the last element of a batch accumulation.
            if (training):
                # If this is the last batch of an accumulation, compute the batch F1 score.
                if (batch_num % self._accumulate_n_batch_grads == 0):
                    # Compute the final running average for this epoch.
                    seen_MF1   = self._seen_unseen_MacroF1['seen'].batch_history[-1]
                    if (not isinstance(seen_MF1, bool)):
                        seen_MF1 /= 100.0
                    unseen_MF1 = self._seen_unseen_MacroF1['unseen'].batch_history[-1]
                    if (not isinstance(unseen_MF1, bool)):
                        unseen_MF1 /= 100.0

                    if (isinstance(seen_MF1, bool) or isinstance(unseen_MF1, bool)):
                        self._batch_history.append(False)
                    else:
                        if (seen_MF1 == 0 or unseen_MF1 == 0):
                            self._batch_history.append(0.)
                        else:
                            self._batch_history.append(100 * 2 * seen_MF1 * unseen_MF1 / (seen_MF1 + unseen_MF1))

        # If this IS a singular evaluation, decoupled from the training procedure.
        else:
            # If the evaluation has completed.
            if (batch_num == self._max_num_batches):
                # Compute the final MacroF1 for this specific instance's dataset split.
                seen_MF1 = self._seen_unseen_MacroF1['seen'].eval_only_history / 100.0
                unseen_MF1 = self._seen_unseen_MacroF1['unseen'].eval_only_history / 100.0
                if (seen_MF1 == 0 or unseen_MF1 == 0):
                    self._eval_only_history = 0.
                else:
                    self._eval_only_history = 100 * 2 * seen_MF1 * unseen_MF1 / (seen_MF1 + unseen_MF1)




    def print_info(self, val=False, epoch=True, avg=True, step=-1, evaluate_only=False):
        """
        This method constructs a string that contains the relevant information for a specific batch/epoch number.


        :param val          : Identifies if this specific instance is evaluated on the validation set.

        :param epoch        : Whether to print information regarding an epoch or a batch.

        :param avg          : Identifies whether this specific instance is evaluated on batches and this print regards
                              a running average of the performance on the training set.

        :param step         : Identifies the batch/epoch whose value is being printed.

        :param evaluate_only: Identifies whether this is a singular evaluation, decoupled from training.


        :return: A string containing the relevant information. E.g.: "~HMF1:  58.98%", for avg=True
        """

        # Normal padding and formatting.
        format_padding = ''
        format_terminology = '.2f}'
        format_padding_error = '-^'
        format_terminology_error = '}'

        no_data_all    = False
        no_data_seen   = False
        no_data_unseen = False
        if (epoch):
            if (len(self._epoch_history) == 0 or (step != -1 and step not in self._epoch_history_x)):
                no_data_all    = True
                no_data_seen   = True
                no_data_unseen = True
        else:
            if (len(self._batch_history) == 0 or (step != -1 and step not in self._batch_history_x)):
                batch_info = '-'
                no_data_all    = True
            else:
                seen_MF1 = self._seen_unseen_MacroF1['seen'].batch_history[step]
                if (isinstance(seen_MF1, bool)):
                    no_data_seen = True
                    seen_MF1 = '-'

                unseen_MF1 = self._seen_unseen_MacroF1['unseen'].batch_history[step]
                if (isinstance(unseen_MF1, bool)):
                    no_data_unseen = True
                    unseen_MF1 = '-'

                batch_info = self._batch_history[step]
                if (isinstance(batch_info, bool)):
                    batch_info  = '-'
                    no_data_all = True

        format_padding_HMF1     = format_padding_error if no_data_all else format_padding
        format_terminology_HMF1 = format_terminology_error if no_data_all else format_terminology

        format_padding_SMF1     = format_padding_error if no_data_seen else format_padding
        format_terminology_SMF1 = format_terminology_error if no_data_seen else format_terminology

        format_padding_UMF1     = format_padding_error if no_data_unseen else format_padding
        format_terminology_UMF1 = format_terminology_error if no_data_unseen else format_terminology

        # Final formatting and string construction.
        avg = super().print_info(epoch, val, avg)

        format_type_HMF1 = '{:' + format_padding_HMF1 + str(6 if not no_data_all else 7) + \
                           format_terminology_HMF1 + ('%' if not no_data_all else "")

        format_type_SMF1 = '{:' + format_padding_SMF1 + str(6 if not no_data_seen else 7) + \
                           format_terminology_SMF1 + ('%' if not no_data_seen else "")

        format_type_UMF1 = '{:' + format_padding_UMF1 + str(6 if not no_data_unseen else 7) + \
                           format_terminology_UMF1 + ('%' if not no_data_unseen else "")

        print_str = avg + (format_type_HMF1.format((self.epoch_avg(step) if epoch else batch_info)
                                              if not evaluate_only else self._eval_only_history))

        if (not (epoch and len(self._epoch_history) == 0) and not (not epoch and len(self._batch_history) == 0)):
            print_str += ' (' + (format_type_SMF1.format((self.seen_unseen_MacroF1['seen'].epoch_avg(step)
                                                          if epoch else seen_MF1)
                                                         if not evaluate_only
                                                         else self.seen_unseen_MacroF1['seen'].eval_only_history))

            print_str += ', ' + (format_type_UMF1.format((self.seen_unseen_MacroF1['unseen'].epoch_avg(step)
                                                          if epoch else unseen_MF1)
                                                         if not evaluate_only
                                                         else self.seen_unseen_MacroF1['unseen'].eval_only_history))
            print_str += ')'

        return  print_str




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
            if (len(self._epoch_history) < len(self._epoch_history_x)):
                seen_MF1 = self._seen_unseen_MacroF1['seen'].epoch_avg(step) / 100.0
                unseen_MF1 = self._seen_unseen_MacroF1['unseen'].epoch_avg(step) / 100.0

                if (isinstance(seen_MF1, bool) or isinstance(unseen_MF1, bool)):
                    return False
                else:
                    if (seen_MF1 == 0 or unseen_MF1 == 0):
                        return 0.
                    else:
                        return 100 * 2 * seen_MF1 * unseen_MF1 / (seen_MF1 + unseen_MF1)
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

        # HarmonicMacroF1 needs to distinguish between seen and unseen classes.
        if (bool(is_seen_class_indicator)):
            err_msg = "HarmonicMacroF1: There's a distinction between 'seen' and 'unseen' classes, yet the dictionary "
            err_msg += "'classes_in_split_to_idx_map' does not have the same classes has the dictionary "
            err_msg += "'is_seen_class_indicator'."
            assert set(classes_in_split_to_idx_map.keys()) == set(is_seen_class_indicator.keys()), err_msg

        class_overlap = False
        for new_class in classes_in_split_to_idx_map:
            if (new_class not in self._classes_in_split_to_idx_map):
                self._classes_in_split_to_idx_map[new_class] = classes_in_split_to_idx_map[new_class]
                self._is_seen_class_indicator[new_class] = is_seen_class_indicator[new_class]
            else:
                class_overlap = True

        self._seen_classes_to_idx   = {_class: idx for _class, idx in self._classes_in_split_to_idx_map.items()
                                       if self._is_seen_class_indicator[_class]}
        self._unseen_classes_to_idx = {_class: idx for _class, idx in self._classes_in_split_to_idx_map.items()
                                       if not self._is_seen_class_indicator[_class]}


        # Update the classes of the sub-elements of MacroF1.
        for seen_unseen, MF1 in self._seen_unseen_MacroF1.items():
            MF1.update_classes_info(self._seen_classes_to_idx if seen_unseen == 'seen' else self._unseen_classes_to_idx,
                                    None)

        self._idx_to_classes_in_split_map = [_class for _class, idx in sorted(self._classes_in_split_to_idx_map.items(),
                                                                              key=lambda x: x[1])]

        if (class_overlap):
            print_warning("When adding new classes to the 'MacroF1' metric, some of these new classes were already " +
                          "being tracked. The new setting WILL BE IGNORED.")




    # OVERRIDE DUE TO extra variables
    def save(self, dir_path, file_type):
        """
        Saves the MacroF1's state to file.


        :param dir_path : The path to the dir in which to save the Loss' state.

        :param file_type: The type of the file to be saved. That is, a regular checkpoint, '.ckp', or the best state,
                          '.bst', so far.


        :return: Nothing
        """

        # Get base state dict, common to all metrics.
        state = super()._save_state()

        # Extend the base state dict with the extra information relevant for the Loss metric.
        state["seen_classes_to_idx"]   = self._seen_classes_to_idx
        state["unseen_classes_to_idx"] = self._unseen_classes_to_idx
        state['print_info_max_length'] = self._print_info_max_length

        torch.save(state, join_path([dir_path] + [self._name + file_type]))

        # Now we save each sub-metric. First we guarantee that the correct dir exists.
        sub_metrics_dir_path = join_path([dir_path] + [self._name + '_sub_metrics' + os.sep])
        directory = os.path.dirname(sub_metrics_dir_path)
        if (not os.path.exists(directory)):
            os.makedirs(directory)

        for _, mf1 in self.seen_unseen_MacroF1.items():
            mf1.save(sub_metrics_dir_path, file_type)




    @property
    def seen_unseen_MacroF1(self):
        return self._seen_unseen_MacroF1

    @property
    def seen_classes_to_idx(self):
        return self._seen_classes_to_idx

    @property
    def unseen_classes_to_idx(self):
        return self._unseen_classes_to_idx