"""
##################################################
##################################################
## This file implements a Metrics Manager,      ##
## whose job is to keep track of all the        ##
## metrics being evaluated, actually having     ##
## them evaluate, print train information,      ##
## check for convergence and determine the best ##
## model.                                       ##
## It can be instantiated by itself to check    ##
## the metrics values at specific checkpoints.  ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
import os
import sys


# *** Own modules imports. *** #

import helpers.names                as _alt_names
import helpers.classes_instantiator as _classes
import helpers.trainer.helpers      as trainer_helpers
from   helpers.trainer.metrics.loss import Loss





#################
##### CLASS #####
#################

class MetricsManager(object):
    """
    The MetricsManager (MMa) keeps track of all issues related to the evaluation of metrics.
    """

    def __init__(self, input_metrics, output_to_metrics,
                 max_num_epochs, max_num_batches, accumulate_n_batch_grads, training_batch_sizes, loss_function_name,
                 evaluate_validation, evaluate_test,
                 load_path, load_epoch_or_best, loss_params_dict={}, just_MMa=False):
        """
        Instantiates a MetricsManager object.


        :param input_metrics      : A tuple that contains the names of the metrics that are meant to be instantiated,
                                    the metrics, if any, and the corresponding parameters, that will determine if a
                                    model has converged, the names of the metrics for which information should be
                                    displayed and the metric which will be used to check if the latest evaluated epoch
                                    has been the one where the model has performed the best.

        :param max_num_epochs     : The maximum number of epochs for which the model will be trained in this current
                                    instantiation. Used for printing purposes.

        :param max_num_batches    : The maximum number of batches in an epoch, for this instantiation.

        :param loss_function_name : The name of the loss function with which the model will be trained.

        :param evaluate_validation: Whether the model will be evaluated on the validation set or not. If True this will
                                    instantiate separate metrics' instances specifically for the validation set.

        :param evaluate_test      : Whether the model will be evaluated on the test set or not. If True this will
                                    instantiate separate metrics' instances specifically for the test set.

        :param load_path          : The path that leads to a saved state of an instance of a MetricsManager.

        :param load_epoch_or_best : This parameter identifies whether to load the state associated with a specific
                                    epoch or the state of the epoch in which the model performed the best.

        :param just_MMa           : TODO
        """

        # If the model is to be loaded from memory, append the directory that contains the saved states of instances
        # related to metrics.
        if (load_path is not None):
            load_path = load_path + "MMa" + os.sep

        # Load the state of a previously instantiated MetricsManager. Will be None if no path is provided.
        self._mma_state = trainer_helpers.load_checkpoint_state(load_path, "state", load_epoch_or_best)


        #         if (just_MMa):
        #             raise NotImplementedError("The possibility to instantiate only the Metrics Manager has not been implemented yet.")
        #             # TODO: CHECK for metrics validity (in case we are just instantiating a Metrics Manager
        #             # TODO: (to look at results, for example))
        #             # metrics_names will determine which metrics to load into the MMa and print_metrics should match it?

        # The names of the metrics to be evaluated.
        self._metrics_names = input_metrics[0] if not just_MMa else self._mma_state['metrics_names']

        # The correspondence between elements of the model's output and the existing metrics.
        self._output_to_metrics = output_to_metrics if not just_MMa else None

        # The names and corresponding parameters of the metrics that will be used to test for model convergence.
        self._convergence_metrics_and_criteria = input_metrics[1] if not just_MMa else self._mma_state['convergence_metrics_and_criteria']

        # The names of the metrics for which information will be displayed.
        self._print_metrics = input_metrics[2] if not just_MMa else self._mma_state['metrics_names']

        # Saves the previous print for the case where only the MMa is instantiated and the information about the
        # training, up to that point, is meant to be displayed.
        self._previous_print = "" if not just_MMa else self._mma_state['previous_print']

        # The name of the metric which will be used to determine if the latest epoch has been the one where the model
        # has performed the best.
        self._save_best_metric = input_metrics[3] if not just_MMa else self._mma_state['save_best_metric']

        # Variables regarding whether to evaluate on the validation/test set and in which epochs the validation set has
        # been evaluated.
        self._evaluate_validation = evaluate_validation if not just_MMa else self._mma_state['evaluate_validation']
        self._validation_evaluations = [] if self._mma_state is None else self._mma_state['validation_evaluations']
        self._evaluate_test = evaluate_test if not just_MMa else self._mma_state['evaluate_test']

        # The name of the loss function that will be used to train the mode.
        self._loss_function_name = loss_function_name if not just_MMa else self._mma_state['loss_function_name']
        self._loss_params_dict   = loss_params_dict if not just_MMa else {}

        # Variables used to determine information about the batch/epoch number.
        self._max_num_epochs           = max_num_epochs
        self._max_num_batches          = max_num_batches if not just_MMa else {'train': 0, 'val': 0, 'test': 0}
        self._accumulate_n_batch_grads = accumulate_n_batch_grads
        self._training_batch_sizes     = training_batch_sizes if not just_MMa else (None, )
        # TODO: Check if current_batch_num should start at 1 for metrics that are added after model has already been partially trained.
        self._current_batch_num        = 1 if self._mma_state is None else self._mma_state['current_batch_num']


        # Creates metrics for all dataset partitions
        self._metrics = self._load_create_metrics(load_path, load_epoch_or_best)


        # Sets early stopping conditions
        # If a validation set exists, early stopping will be checked on it.
        if (not just_MMa):
            for metric in self._convergence_metrics_and_criteria:
                split = self._convergence_metrics_and_criteria[metric][0]
                if (split in self._metrics):
                    metric_instance = self._metrics[split][_alt_names.METRICS[metric]]
                    metric_instance.set_convergence_parameters(self._convergence_metrics_and_criteria[metric][1:])




    def _load_create_metrics(self, load_path, load_epoch_or_best):
        """
        This method is used to initiate the MetricsManager arguments. They can either be loaded from memory or given as
        an input during instantiation.


        :param load_path           : The path that leads to a saved state of an instance of this metric.

        :param load_epoch_or_best  : This parameter identifies whether to load the state associated with a specific
                                     epoch or the state of the epoch in which the model performed the best.


        :return: A dictionary that for each dataset split has a dictionary that contains each metric meant to be
                 evaluated.
        """

        metrics = {}

        # If a saved state was loaded from memory, get which metrics had been in use at that point, as they will be
        # instantiated again. TODO: Possibly redundant with a MEM, but might be useful when instantiating only the MMa.
        previous_metrics = None
        if (self._mma_state is not None):
            previous_metrics = self._mma_state['metrics']

        # Determine the dataset splits that are meant to be evaluated, also considering what has already been evaluated
        # in past epochs.
        dataset_splits = list(
            set(['train'] + (['val'] if self._evaluate_validation else []) + (['test'] if self._evaluate_test else []) +
                (list(previous_metrics) if previous_metrics is not None else [])))

        # Instantiate each metric for each dataset split
        for dataset_split in dataset_splits:
            metrics[dataset_split] = {}

            acc_n = self._accumulate_n_batch_grads if dataset_split == 'train' else 1
            training_batch_sizes = self._training_batch_sizes if dataset_split == 'train' else (None, None)
            for metric in self._metrics_names:
                # Determine the loss parameters dictionary. Used for special loss uses, like KL annealing and so on.
                loss_params_dict = self._loss_params_dict if dataset_split == 'train' and metric == 'Loss' else {}
                if (dataset_split == 'train' and metric == 'Loss' and '_DEBUG' in loss_params_dict):
                    loss_debug = loss_params_dict['_DEBUG']
                else:
                    loss_debug = False

                # Load past metrics, if necessary
                if (previous_metrics is not None and dataset_split in previous_metrics and
                        _alt_names.METRICS[metric] in previous_metrics[dataset_split]):

                    # TODO: Test might be unnecessary as "previous_metrics is not None"
                    # Redefine the directory where the metrics will be loaded from.
                    load_path_temp = load_path if load_path is None else load_path + dataset_split + os.sep

                    # Load the Loss metric
                    if (_alt_names.METRICS[metric] == 'Loss'):
                        loaded = Loss(_classes.LOSSES[_alt_names.LOSSES[self._loss_function_name]](**loss_params_dict),
                                      dataset_split == 'train', self._max_num_batches[dataset_split], acc_n,
                                      training_batch_sizes, load_path_temp, load_epoch_or_best, _DEBUG=loss_debug)
                    # Load any other metric
                    else:
                        loaded = _classes.METRICS[_alt_names.METRICS[metric]](dataset_split == 'train',
                                                                              self._max_num_batches[dataset_split],
                                                                              acc_n, training_batch_sizes,
                                                                              load_path_temp, load_epoch_or_best)

                    metrics[dataset_split][_alt_names.METRICS[metric]] = loaded

                # Instantiate inexistent metrics
                else:
                    # Create a new Loss metric instance
                    if (_alt_names.METRICS[metric] == 'Loss'):
                        new = Loss(_classes.LOSSES[_alt_names.LOSSES[self._loss_function_name]](**loss_params_dict),
                                   dataset_split == 'train', self._max_num_batches[dataset_split], acc_n,
                                   training_batch_sizes, None, None, self._current_batch_num, _DEBUG=loss_debug)

                    # Create a new instance of any other metric
                    else:
                        new = _classes.METRICS[_alt_names.METRICS[metric]](dataset_split == 'train',
                                                                           self._max_num_batches[dataset_split],
                                                                           acc_n, training_batch_sizes,
                                                                           None, None, self._current_batch_num)

                    metrics[dataset_split][_alt_names.METRICS[metric]] = new

        return metrics




    def evaluate(self, data_loader_type, batch_num, epoch_num, predictions, labels, training, evaluate_only=False):
        """
        This method makes each metric, associated with a specific dataset split, evaluate the performance of the model
        on the current batch.

        :param data_loader_type: Identifies the dataset split that is going to be evaluated.

        :param batch_num       : The number of the current minibatch, since the start of the training procedure.

        :param epoch_num       : The number of the current epoch, since the start of the training procedure.

        :param predictions     : The model's predictions.

        :param labels          : The correct output for the corresponding data instances.

        :param training        : Determines if this evaluation is performed for training purposes, which means batch
                                 information will be kept.

        :param evaluate_only   : Used to determine whether to register the computed values or not. If True, then it is
                                 assumed this was a singular evaluation, not within the training procedure.


        :return: Nothing
        """

        for metric in self._metrics[data_loader_type]:
            if (metric in self._output_to_metrics[data_loader_type]):

                # TODO: Need to make this into a nice pattern. Right now it's really hacky.
                output_elements = self._output_to_metrics[data_loader_type][metric]
                if (isinstance(predictions, tuple)):
                    metric_predictions = tuple(predictions[ele] for ele in output_elements)
                else:
                    raise RuntimeError("Predictions should always be a tuple. Something went wrong.")

                if (isinstance(labels, tuple)):
                    metric_labels = tuple(labels[ele] for ele in output_elements)
                else:
                    # Pytorch Tensors have a 'tuple' method that converts them to tuples. So we write it this way
                    # instead.
                    metric_labels = (labels, )

                self._metrics[data_loader_type][metric].evaluate(batch_num, epoch_num, metric_predictions,
                                                                 metric_labels, training, evaluate_only=evaluate_only)




    def check_for_early_stopping(self):
        """
        Checks whether the early stopping conditions have been met, for all metrics defined as convergence metrics.


        :return: Nothing
        """

        # Determines if there are any convergence criteria (i.e., if any metrics have been specified on which to test
        # for convergence).
        if (bool(self._convergence_metrics_and_criteria)):
            # Check if the convergence criteria has been met for all convergence metrics.
            return all(self._metrics[self._convergence_metrics_and_criteria[metric][0]][_alt_names.METRICS[metric]].evaluate_early_stopping()
                       for metric in self._convergence_metrics_and_criteria)

        # There are no convergence criteria, so return that the model has not converged.
        else:
            return False




    def check_if_best(self, current_epoch_num):
        """
        Checks if the last epoch has been the one where the model has performed the best, under metric defined to test
        for the best performance.


        :param current_epoch_num: The current epoch number.


        :return: True if last epoch has been the one where the model has performed the best so far. False otherwise.
        """

        best = False

        # If a a metric was specified as the one to be used to test if the current epoch is the one where the model has
        # performed the best.
        if (bool(self._save_best_metric)):
            # Determine if the test is to be performed on the train or validation metrics.
            train_or_val = "train" if not self._evaluate_validation else "val"
            best = self._metrics[train_or_val][self._save_best_metric].check_if_best(current_epoch_num)
        return best




    def backpropagate_loss(self, batch_num):
        """
        This method is used to have the Loss metric, evaluated on the train set, backpropagate it's value to the model
        parameters.


        :return: Nothing
        """

        # Backpropagate.
        self._metrics["train"]["Loss"].loss.backward()

        # If this is the first batch of an accumulation, update the current batch number.
        if ((batch_num - 1) % self._accumulate_n_batch_grads == 0):
            self._current_batch_num += 1

        # If the next batch is the first one of an accumulation, update the annealing factor.
        if (batch_num % self._accumulate_n_batch_grads == 0 or batch_num == self._max_num_batches['train']):
            self._metrics["train"]["Loss"].loss_function.annealing_step()






    def get_specific_metric(self, data_loader_type, metric_name):
        """
        This method allows the retrieval of the instance corresponding to a specific metric associated with a specific
        dataset split


        :param data_loader_type: The dataset split associated with the desired metric.
        :param metric_name     : The metric that is intended to be returned.


        :return: The instance of the 'metric_name' metric, associated with the dataset split 'data_loader_type'.
        """

        # Makes sure the dataset split associated with the desired metric has been evaluated at all.
        if (data_loader_type in self._metrics):
            # Makes sure that the specified metric has been implemented and can be instantiated.
            if (metric_name in _alt_names.METRICS):
                # Makes sure that the specified metric has been instantiated.
                if (_alt_names.METRICS[metric_name] in self._metrics[data_loader_type]):
                    return self._metrics[data_loader_type][_alt_names.METRICS[metric_name]]
                else:
                    raise KeyError(
                        "Untested metric: \'" + metric_name + "\' (\'" + _alt_names.METRICS[metric_name] + "\').")
            else:
                raise KeyError("Unknown metric: \'" + metric_name + "\'.")
        else:
            raise KeyError("Unknown dataset type: \'" + data_loader_type + "\'.")




    def _print_header(self, evaluate_only):
        """
        This method is used to print the header that identifies the information being printed during training or
        singular evaluations.


        :param evaluate_only: Parameter that identifies whether this is a singular evaluation, decoupled from training,
        or not as, if it is, then there is no need to write the part of the header associated with the epoch number.


        :return: A string that is the header itself.
        """

        header = ""

        # Determine if we are not performing a singular evaluation, decoupled from the training procedure.
        if (not evaluate_only):
            # Print the header related to the epoch number and percentage of training that has been completed.
            format_epoch_num_info = "{:^" + str(max(2 * len(str(self._max_num_epochs)) + 1, 5)) + "s}"

            header += format_epoch_num_info.format('Epoch') + "     Trained" +\
                      ( " |||" if len(self._print_metrics) > 0 else "")

        # Determine whether there are actually any metrics whose information is meant to be printed.
        if (len(self._print_metrics) > 0):
            # To account for space between "|||" (which does not exist when evaluate_only is True) and metric symbol.
            length_train_info = (0 if not evaluate_only else -1)

            for metric in self._print_metrics:
                # Determines the length of the train set header separator.----v TODO: shouldn't the names of print_metrics already be valid?
                length_train_info += self._metrics["train"][_alt_names.METRICS[metric]].print_info_max_length
                # +(0 or -1) for no average (~) when evaluate_only.
                # +2 for comma and space between metrics.
                length_train_info += (0 if not evaluate_only else -1) + 2

            format_train_info = "{:^" + str(length_train_info) + "s}"

            header += format_train_info.format('Train Set') + ("|||" if self._evaluate_validation else "")

            # If there is a validation set being used
            if (self._evaluate_validation):
                # Compute the length that the validation info will take
                length_val_info = 0
                for metric in self._print_metrics:
                    # Explanation of the +1: we first subtract 1 from each 'print_info_max_length' as the '~' symbol
                    # used to represent the average will never be present. Then we add 2: 1 for the space before and
                    # another 1 for either the comma or last space.
                    length_val_info += self._metrics["val"][_alt_names.METRICS[metric]].print_info_max_length + 1
                format_val_info = "{:^" + str(length_val_info) + "s}"

                header += format_val_info.format('Val Set') + "|"

        return header




    def print_info(self, epoch_num, batch_num, new_val=False, batch=False, previous_print=False,
                   step_is_epoch_num=False, evaluate_only=False):
        """
        This method constructs a string that contains the relevant training information for a specific batch/epoch
        number.


        :param epoch_num        : The epoch number for which the information is meant to be printed.
        :param batch_num        : When printing the information regarding a batch, this identifies the batch number.
        :param new_val          : Identifies whether a new evaluation on the validation set was performed this epoch.
        :param batch            : Identifies whether the information pertains a batch or an epoch.
        :param previous_print   : Identifies whether to print the previously printed information or not.
        :param step_is_epoch_num: Identifies whether the number that is used to select which time step of each metric is
                                  meant to be printed is equal to the epoch number or not.
        :param evaluate_only    : Identifies whether the information to be printed pertains a singular evaluation,
                                  decoupled from training, or not.


        :return: Nothing
        """

        # Determines whether to print the previously printed information or not (i.e. the information of past epochs).
        if (previous_print):
            print(self._previous_print, end="", flush=True)
            return

        # This allows the MMa to know in which epochs the performance on the validation set was evaluated. It's useful
        # for reprinting the information pertaining past epochs, when reloading a model from memory.
        if (batch_num == self._max_num_batches['train'] and not batch and not step_is_epoch_num and not evaluate_only):
            self._validation_evaluations.append(new_val)

        # TODO: See if this is actually required or not. If it is, write a comment
        step_is_epoch_num = epoch_num if step_is_epoch_num else -1

        print_string = "\r"

        # If this is the pre-evaluation, or a singular evaluation, decoupled from the training procedure, print the
        # header.
        if (epoch_num == 0):
            print_string = self._print_header(evaluate_only) + '\n'

        # If this is NOT a singular evaluation, decouple from the training procedure, write the information pertaining
        # the epoch number and training completion.
        if (not evaluate_only):
            # Formats the length required to identify the epoch number.
            num_digits_epoch = len(str(self._max_num_epochs))
            format_epoch_number = "{:" + str(num_digits_epoch) + "d}"

            # Adds the epoch number information to the string.
            print_string += ("  " if num_digits_epoch == 1 else "") + format_epoch_number.format(epoch_num)
            print_string += '/' + str(self._max_num_epochs) + " --> "

            # Adds the information regarding the percentage of training that has been completed.
            training_percent = 100.0*(epoch_num - 1 + batch_num/self._max_num_batches['train'])
            training_percent /= max(self._max_num_epochs, 1)
            print_string += '{:6.2f}%'.format(training_percent)

            # Adds a delimiter, if required, that separates the epoch number and completed training percentage from the
            # metrics information.
            print_string += (" |||" if len(self._print_metrics) > 0 else "")

        # If this is a batch, print how much, in percentage, this epoch has been completed.
        if (batch):
            print_string += (" |||" if len(self._print_metrics) == 0 else "") + ' %Epoch: ' +\
                            '{:6.2f}%'.format(100.0 * batch_num / self._max_num_batches['train'])

        # Print metrics if necessary
        if (len(self._print_metrics) > 0):
            # End of epoch
            if (not batch):
                # Train set
                sys.stdout.write('\x1b[2K')# Clear previous output.
                for i, metric in enumerate(self._print_metrics):
                    # Accounts for extra print space when not performing an evaluation decoupled from the training
                    # procedure.
                    if (not evaluate_only):
                        print_string += " "

                    # TODO: need to see how much of a hack this is, just to get the correct printing spacing.
                    # Get the metric info and allow for space not required when performing an evaluation decoupled from
                    # the training procedure.
                    print_from = 1 if i == 0 and evaluate_only else 0
                    metric_to_print = self._metrics["train"][_alt_names.METRICS[metric]]
                    print_string += metric_to_print.print_info(val=False, avg=False if epoch_num == 0 else True,
                                                               step=epoch_num, evaluate_only=evaluate_only)[print_from:]

                    # Write a comma separating different metrics, or a delimiter between the metrics' information
                    # regarding the train set and the metrics' information regarding the validation set, if evaluated.
                    not_last_metric = i + 1 < len(self._print_metrics)
                    print_string += ',' if not_last_metric else (" |||" if self._evaluate_validation else "")

                # Validation set
                if (self._evaluate_validation):
                    for i, metric in enumerate(self._print_metrics):
                        # Get the metric info.
                        metric_to_print = self._metrics["val"][_alt_names.METRICS[metric]]
                        print_string += " " + metric_to_print.print_info(val=True, avg=False, step=epoch_num,
                                                                         evaluate_only=evaluate_only)

                        # Write a comma separating different metrics, or a delimiter and an indication if the model has
                        # been evaluated on the validation set on this epoch.
                        not_last_metric = i + 1 < len(self._print_metrics)
                        print_val_eval = self._validation_evaluations[epoch_num] and not evaluate_only
                        print_string += ',' if not_last_metric else ( " | (Val eval)" if print_val_eval else " |")

            # During epoch
            else:
                # Batch info
                print_string += ' | Batch:'
                for i, metric in enumerate(self._print_metrics):
                    # Get the metric info.
                    metric_to_print = self._metrics["train"][_alt_names.METRICS[metric]]
                    print_string += " " + metric_to_print.print_info(epoch=False, avg=False)

                    # Write a comma separating different metrics, or a delimiter between the metrics' batch information
                    # and the epoch's train set running average information.
                    print_string += ',' if i + 1 < len(self._print_metrics) else " |"

                # Train set info
                print_string += ' Train Set:'
                for i, metric in enumerate(self._print_metrics):
                    # Get the metric info.
                    print_string += " " + self._metrics["train"][_alt_names.METRICS[metric]].print_info(avg=True)

                    # Write a comma separating different metrics, or end the string.
                    print_string += ',' if i + 1 < len(self._print_metrics) else ""

        # If the information to be written pertains the end of epoch information, write a new line.
        print_string += "\n" if not batch else ""

        # If the information to be written pertains the end of epoch information (and is not a singular evaluation,
        # decoupled from the training procedure), add it to the previous printed information.
        if (not batch and not evaluate_only):
            self._previous_print += print_string

        print(print_string, end="", flush=True)




    def save(self, dir_path, file_type, test_only_best=False):
        """
        Saves the MetricsManager's state to file.


        :param current_epoch: Identifies the epoch number of the state being saved.

        :param dir_path     : Path where the states will be saved.

        :param file_type    : Determines the file type to be saved. Either a regular checkpoint or the state associated
                              with the model that has performed the best, so far.


        :return: Nothing
        """

        # Creates the state dict for the relevant variables.
        state_dict = {
            "metrics_names"                   : self._metrics_names,
            "metrics"                         : {data_split: [metric for metric in self._metrics[data_split]]
                                                 for data_split in self._metrics},
            "convergence_metrics_and_criteria": self._convergence_metrics_and_criteria,
            "print_metrics"                   : self._print_metrics,
            "previous_print"                  : self._previous_print,
            "save_best_metric"                : self._save_best_metric,
            "evaluate_validation"             : self._evaluate_validation,
            "validation_evaluations"          : self._validation_evaluations,
            "evaluate_test"                   : self._evaluate_test,
            "loss_function_name"              : self._loss_function_name,
            "current_batch_num"               : self._current_batch_num,
        }

        # Determine if the correct directory related to metrics already exists
        dir_path += "MMa" + os.sep
        directory = os.path.dirname(dir_path)
        if (not os.path.exists(directory)):
            os.makedirs(directory)

        # Save the MMa state
        mma_state_filename = dir_path + "state" + file_type
        torch.save(state_dict, mma_state_filename)

        # Creates the data_split directories, in which the associated metrics' states will be saved.
        for dataset_type in self._metrics:
            # Makes sure the directory exists
            directory = os.path.dirname(dir_path + dataset_type + os.sep)
            if (not os.path.exists(directory)):
                os.makedirs(directory)

            # Saves each metric's state.
            for metric in self._metrics[dataset_type]:
                self._metrics[dataset_type][metric].save(dir_path + dataset_type + os.sep, file_type)




    @property
    def metrics_names(self):
        return self._metrics_names

    @property
    def convergence_metrics_and_criteria(self):
        return self._convergence_metrics_and_criteria

    @property
    def print_metrics(self):
        return self._print_metrics

    @property
    def previous_print(self):
        return self._previous_print

    @property
    def save_best_metric(self):
        return self._save_best_metric

    @property
    def evaluate_validation(self):
        return self._evaluate_validation

    @property
    def validation_evaluations(self):
        return self._validation_evaluations

    @property
    def evaluate_test(self):
        return self._evaluate_test

    @property
    def loss_function_name(self):
        return self._loss_function_name

    @property
    def max_num_epochs(self):
        return self._max_num_epochs

    @property
    def max_num_batches(self):
        return self._max_num_batches

    @property
    def current_batch_num(self):
        return self._current_batch_num

    @property
    def metrics(self):
        return self._metrics