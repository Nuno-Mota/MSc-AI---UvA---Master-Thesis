"""
##################################################
##################################################
## This file implements the Macro F1 metric,    ##
## which measures average of F1-scores of each  ##
## of the relevant classes. For more            ##
## information on the F1-score, please consult  ##
## the corresponding class.                     ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import os
import re


# *** Own modules imports. *** #

from helpers.trainer.metrics.metric import Metric
from helpers.trainer.metrics.f1 import F1
from helpers.general_helpers import print_warning, join_path, get_path_components





#################
##### CLASS #####
#################

class MacroF1(Metric):
    """
    The MacroF1 metric is the average of the F1 scores of each individual class. The F1 score of each class is the
    harmonic mean between precision and recall.
    """

    def __init__(self, evaluated_on_batches, max_num_batches, accumulate_n_batch_grads, training_batch_sizes,
                 load_path, load_epoch_or_best, starting_batch_num=1, optional_seen_unseen=''):
        """
        Initiates a MacroF1 metric object.


        :param evaluated_on_batches: Determines whether this specific instance of a MacroF1 metric is evaluated for
                                     minibatches. Used for plotting.

        :param max_num_batches     : The maximum number of batches in an epoch. Used to aggregate epoch information.

        :param load_path           : The path that leads to a saved state of an instance of the MacroF1 metric.

        :param load_epoch_or_best  : This parameter identifies whether to load the state associated with a specific
                                     epoch or the state of the epoch in which the model performed the best.

        :param starting_batch_num  : Some metrics can be added to the MetricsManager after the model has already been
                                     trained for a few epochs. This identifies the first batch in which this particular
                                     instance was initiated.
        """

        if (load_path is not None):
            load_path = load_path + ('_' if load_path is not None and optional_seen_unseen != '' else '')
        super().__init__("MF1", evaluated_on_batches, max_num_batches, accumulate_n_batch_grads, training_batch_sizes,
                         False, load_path, load_epoch_or_best, starting_batch_num)


        # Variables specific to the MacroF1 metric.
        self._optional_seen_unseen = optional_seen_unseen if self._metric_state is None else self._metric_state['optional_seen_unseen']

        self._classes_individual_F1_scores = {}

        # Variables used to determine the length of the information to be printed.
        self._print_info_max_length = 13 if self._metric_state is None else self._metric_state['print_info_max_length']




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

        # Compute the F1-score for each class.
        argmax_preds = torch.argmax(predictions, dim=-1).long()
        for _class in self._classes_in_split_to_idx_map:
            class_predictions = (argmax_preds == self._classes_in_split_to_idx_map[_class])
            class_labels      = (labels == self._classes_in_split_to_idx_map[_class])
            self._classes_individual_F1_scores[_class].evaluate(batch_num, epoch_num, (class_predictions, ),
                                                                ((class_labels, ), ), training, evaluate_only)


        # If this IS NOT a singular evaluation, decoupled from the training procedure.
        if (not evaluate_only):
            # Register batch/epoch number information using the super class' method.
            super().evaluate(batch_num, epoch_num, None, None, training)

            # MacroF1 is not like most of the other metrics. It just calls the necessary components computed by the
            # lower-level F1 metrics. As such, it only registers epoch information (on itself) at the end of the epoch.

            # If the epoch has completed.
            if (batch_num == self._max_num_batches):
                # Compute the final running average for this epoch.
                numerator = 0
                denominator = 0
                # print('\n' if self._evaluated_on_batches else '')
                for class_num, _class in enumerate(self._classes_individual_F1_scores):
                    class_f1_score = self._classes_individual_F1_scores[_class].epoch_history[-1]
                    # print(_class, class_f1_score,
                    #       end=' ' if class_num + 1 != len(self._classes_individual_F1_scores) else '\n',
                    #       flush=True)
                    numerator   += class_f1_score
                    denominator += 1 if not isinstance(class_f1_score, bool) and class_f1_score == 0 else bool(class_f1_score)
                self._epoch_history.append(numerator / denominator)

            # If the evaluation is performed on the train set (while training, (see evaluate_only))
            # register accuracy value, if it is the last element of a batch accumulation.
            if (training):
                # If this is the last batch of an accumulation, compute the batch MacroF1 score.
                if (batch_num % self._accumulate_n_batch_grads == 0):
                    # Compute the final running average for this epoch.
                    numerator = 0
                    denominator = 0
                    for _class in self._classes_individual_F1_scores:
                        class_f1_score = self._classes_individual_F1_scores[_class].batch_history[-1]
                        numerator += class_f1_score
                        denominator += 1 if not isinstance(class_f1_score, bool) and class_f1_score == 0 else bool(class_f1_score)

                    if (denominator == 0): # This should only be possible to happen when using HarmonicMacroF1.
                        self._batch_history.append(False)
                    else:
                        self._batch_history.append(numerator / denominator)

        # If this IS a singular evaluation, decoupled from the training procedure.
        else:
            # If the evaluation has completed.
            if (batch_num == self._max_num_batches):
                # Compute the final MacroF1 for this specific instance's dataset split.
                numerator = 0
                denominator = 0
                for _class in self._classes_individual_F1_scores:
                    class_f1_score = self._classes_individual_F1_scores[_class].eval_only_history
                    numerator += class_f1_score
                    denominator += 1 if class_f1_score == 0 else bool(class_f1_score)
                self._eval_only_history = numerator / denominator




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

        no_data = False
        if (epoch):
            if (len(self._epoch_history) == 0 or (step != -1 and step not in self._epoch_history_x)):
                no_data = True
        else:
            if (len(self._batch_history) == 0 or (step != -1 and step not in self._batch_history_x)):
                batch_info = '-'
                no_data    = True
            else:
                batch_info = self._batch_history[step]

        if (no_data):
            format_padding = '-^'
            format_terminology = '}'

        # Final formatting and string construction.
        format_type = '{:' + format_padding + str(6 if not no_data else 7) + \
                      format_terminology + ('%' if not no_data else "")
        return super().print_info(epoch, val, avg) +\
               (format_type.format((self.epoch_avg(step) if epoch else batch_info) if not
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
            if (len(self._epoch_history) < len(self._epoch_history_x)):
                numerator = 0
                denominator = 0
                for _class in self._classes_individual_F1_scores:
                    class_f1_epoch_avg = self._classes_individual_F1_scores[_class].epoch_avg(step)
                    numerator += 0 if class_f1_epoch_avg == '-' else class_f1_epoch_avg
                    denominator += 0 if class_f1_epoch_avg == '-' else 1

                if (denominator == 0): # This should only be possible to happen when using HarmonicMacroF1.
                    return False
                else:
                    return numerator / denominator
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

        # Get the correct load_path for the sub-F1 metrics.
        if (self._optional_seen_unseen != ''):
            load_path = None if self._load_path is None else join_path(get_path_components(self._load_path)[:-1])
            load_path = None if self._load_path is None else join_path([load_path] +
                                                                       [self._optional_seen_unseen + '_' + self._name +
                                                                        '_sub_metrics' + os.sep])
        else:
            load_path = None if self._load_path is None else join_path([self._load_path] +
                                                                       [self._name + '_sub_metrics' + os.sep])

        class_overlap = False
        for new_class in classes_in_split_to_idx_map:
            if (new_class not in self._classes_in_split_to_idx_map):
                class_file_name = re.sub('/', '-', re.sub(' ', '_', new_class))
                class_load_path = None if self._load_path is None else join_path([load_path] + [class_file_name + '_'])

                self._classes_in_split_to_idx_map[new_class]  = classes_in_split_to_idx_map[new_class]
                self._classes_individual_F1_scores[new_class] = F1(self._evaluated_on_batches, self._max_num_batches,
                                                                   self._accumulate_n_batch_grads,
                                                                   self._training_batch_sizes, class_load_path,
                                                                   self._load_epoch_or_best,
                                                                   len(self._batch_history_x) + 1, new_class)


                # self._is_seen_class_indicator[new_class] = is_seen_class_indicator[new_class]

            else:
                class_overlap = True

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
        state["optional_seen_unseen"]        = self._optional_seen_unseen
        state["classes_in_split_to_idx_map"] = self._classes_in_split_to_idx_map
        state["idx_to_classes_in_split_map"] = self._idx_to_classes_in_split_map
        state["is_seen_class_indicator"]     = self._is_seen_class_indicator
        state['print_info_max_length']       = self._print_info_max_length

        optional_seen_unseen = self._optional_seen_unseen + '_' if self._optional_seen_unseen != '' else ''
        torch.save(state, join_path([dir_path] + [optional_seen_unseen + self._name + file_type]))

        # Now we save each sub-metric. First we guarantee that the correct dir exists.
        sub_metrics_dir_path = join_path([dir_path] + [optional_seen_unseen + self._name +
                                                       '_sub_metrics' + os.sep])
        directory = os.path.dirname(sub_metrics_dir_path)
        if (not os.path.exists(directory)):
            os.makedirs(directory)

        for _, f1 in self._classes_individual_F1_scores.items():
            f1.save(sub_metrics_dir_path, file_type)




    @property
    def classes_individual_F1_scores(self):
        return self._classes_individual_F1_scores