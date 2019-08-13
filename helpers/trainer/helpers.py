"""
##################################################
##################################################
## This module contains helper functions used   ##
## by the different elements of the trainer.    ##
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

import helpers.names as _names
from   helpers.general_helpers import get_path_components





#####################
##### FUNCTIONS #####
#####################

def load_checkpoint_state(model_to_load_path, file_type, checkpoint_or_best, device=None):
    """
    load_check_point_state() allows loading a previously saved state of any element of the trainer, be it the
    ModelEvaluationModule, the MetricsManager or any Metric. The element to be loaded is defined by the input
    parameters.


    :param model_to_load_path  : This parameter is the path that leads to the directory of the element to be loaded.

    :param element_to_load_type: This parameter identifies the type of element to be loaded.

    :param checkpoint_or_best  : This parameter identifies the saved state's specific epoch to load.


    :return: Returns the corresponding state_dict or None, if no valid model_to_load_path and checkpoint_or_best was
             provided.
    """

    loaded_state = None

    if (model_to_load_path is not None and checkpoint_or_best is not None and isinstance(checkpoint_or_best, int)):

        # Concatenates the path of the specific element to load with the general path of where models are saved.
        load_path = model_to_load_path

        # Returns the names of the existing files in the load_path directory
        existing_files = os.listdir(load_path)

        # Loads the state associated with the best model
        if (checkpoint_or_best == -1 and ".bst" in {existing_file[-4:] for existing_file in existing_files}):
            path = load_path
            path += [file for file in existing_files if (file.endswith(".bst") and file.startswith(file_type))][0]
            loaded_state = torch.load(path, map_location=device)

        # Loads the state associated with a specific checkpoint
        elif (checkpoint_or_best >= 0 and os.path.exists(load_path+file_type + "." + str(checkpoint_or_best) + ".ckp")):
            loaded_state = torch.load(load_path + file_type + "." + str(checkpoint_or_best) + ".ckp",
                                      map_location=device)

        # The desired epoch has not yet been computed. Issue and error.
        else:
            split_path = get_path_components(load_path)
            if (split_path[-1] in list(_names.METRICS) + ["Loss"]):
                # This means that the state of a metric was meant to be loaded.
                raise ValueError("Tried to load " + split_path[-1] + "'s' (" + split_path[-2] + ") state dict for " +
                                 "invalid or inexisting epoch (epoch num: " + str(checkpoint_or_best) + ").")
            else:
                # This means that the state of either the MEM or the MMa was meant to be loaded.
                raise ValueError("Tried to load " + split_path[-1] + "'s state dict for invalid or inexisting epoch " +
                                 "(epoch num: " + str(checkpoint_or_best) + ").")

    return loaded_state




#######################################################################################################################

def verify_valid_metrics(metrics_names, convergence_metrics_and_criteria, print_metrics, save_best_metric):
    """
    verify_valid_metrics() makes sure that all input metrics exist and that they are valid. This is done by
    specifying in metrics_names all the metrics that are meant to be evaluated and then the validity of those is
    checked (this is done because the corresponding metric class needs to have been implemented). All other metrics
    parameters are then checked against the valid_metrics_names.


    :param metrics_names                   : This parameter is a list that contains the names of all metrics that are
                                             meant to be evaluated.

    :param convergence_metrics_and_criteria: This parameter specifies which of the valid metrics in metrics_names will
                                             be used for early stopping.

    :param print_metrics                   : This parameters specifies which metrics will have their information
                                             displayed during training and evaluation. This is so that one can compute
                                             the value of certain metrics without necessarily wanting to know their
                                             values at train time.

    :param save_best_metric                : This identifies the metric on which the best model is identified. This is
                                             used during training. By default, identifying the best model is done on the
                                             validation set. If it does not exist, the check is done on the trains set.
                                             If it is invalid it defaults to the 'Loss'.


    :return: Returns each of the components above, in the same order, but, from those, only the valid metrics.
    """


    # Determine which metrics are valid to be used, by verifying that they exist in the general ALTERNATIVE NAMES dict.
    valid_metrics_names   = []
    unknown_metrics_names = [] # For error feedback
    for metric in metrics_names:
        if (metric in _names.METRICS):
            if (_names.METRICS[metric] not in valid_metrics_names): # Avoids repeated metrics names
                valid_metrics_names.append(_names.METRICS[metric])
        else:
            unknown_metrics_names.append(metric)


    # Evaluate and determine which selected convergence_metrics are valid, given the valid metrics_names.
    valid_convergence_metrics = {} # TODO: Check that convergence parameters make sense
    invalid_convergence_metrics = [] # For error feedback
    unknown_convergence_metrics = [] # For error feedback
    for metric in convergence_metrics_and_criteria:
        if (metric in _names.METRICS):
            if (_names.METRICS[metric] in valid_metrics_names and
                    _names.METRICS[metric] not in valid_convergence_metrics): # Avoids repeated metrics names
                valid_convergence_metrics[_names.METRICS[metric]] = convergence_metrics_and_criteria[metric]
            else:
                invalid_convergence_metrics.append(_names.METRICS[metric])
        else:
            unknown_convergence_metrics.append(metric)


    # Evaluate and determine which print_metrics are valid, given the valid metrics_names.
    invalid_print_metrics = [] # For error feedback
    unknown_print_metrics = [] # For error feedback
    if (print_metrics == 'All'):
        valid_print_metrics = valid_metrics_names
    else:
        valid_print_metrics = []
        for metric in print_metrics:
            if (metric in _names.METRICS):
                if (_names.METRICS[metric] in valid_metrics_names and
                        _names.METRICS[metric] not in valid_print_metrics): # Avoids repeated metrics names
                    valid_print_metrics.append(_names.METRICS[metric])
                else:
                    invalid_print_metrics.append(_names.METRICS[metric])
            else:
                unknown_print_metrics.append(metric)


    # Evaluate and determine if save_best_metric is valid, given the valid metrics_names.
    valid_save_best_metric = False
    invalid_save_best_metric = False # For error feedback
    unknown_save_best_metric = False # For error feedback
    if (save_best_metric != False):
        if (save_best_metric in _names.METRICS):
            if (_names.METRICS[save_best_metric] in valid_metrics_names):
                valid_save_best_metric = _names.METRICS[save_best_metric]
            else:
                invalid_save_best_metric = True
                valid_save_best_metric = "Loss" # save_best_metric defaults to the "Loss" metric
        else:
            unknown_save_best_metric = True
            valid_save_best_metric = "Loss" # save_best_metric defaults to the "Loss" metric



    # This section prints warnings regarding any inconsistencies detected with the chosen metrics. #
    need_to_warn = any((bool(unknown_metrics_names), bool(invalid_convergence_metrics),
                        bool(unknown_convergence_metrics), bool(invalid_print_metrics),
                        bool(unknown_print_metrics), invalid_save_best_metric, unknown_save_best_metric))
    if (need_to_warn):
        print("The following irregularities with metrics have been detected " +
              "and the corresponding settings will be ignored:", end="")


    # metrics_names irregularities
    if (bool(unknown_metrics_names)):
        print("\n\n##################\n# METRICS' NAMES #\n##################")
        print_string = "Unknown metrics to be evaluated:"
        unknown_metrics_names = set(unknown_metrics_names)
        for i, unknown_metric in enumerate(unknown_metrics_names):
            print_string += " " + unknown_metric + ("," if i < len(unknown_metrics_names) - 1 else "")
        print(print_string)


    # convergence_metrics irregularities
    if (bool(invalid_convergence_metrics) or bool(unknown_convergence_metrics)):
        print("\n\n#######################\n# CONVERGENCE METRICS #\n#######################")

    if (bool(invalid_convergence_metrics)):
        print_string = "Invalid convergence metrics to be evaluated:"
        invalid_convergence_metrics = set(invalid_convergence_metrics)
        for i, invalid_convergence_metric in enumerate(set(invalid_convergence_metrics)):
            print_string += " " + invalid_convergence_metric + (
                "," if i < len(invalid_convergence_metrics) - 1 else "")
        print(print_string)

    if (bool(unknown_convergence_metrics)):
        print_string = "Unknown convergence metrics to be evaluated:"
        unknown_convergence_metrics = set(unknown_convergence_metrics)
        for i, unknown_convergence_metric in enumerate(unknown_convergence_metrics):
            print_string += " " + unknown_convergence_metric + (
                "," if i < len(unknown_convergence_metrics) - 1 else "")
        print(print_string)


    # print_metrics irregularities
    if (bool(invalid_print_metrics) or bool(unknown_print_metrics)):
        print("\n\n#################\n# PRINT METRICS #\n#################")

    if (bool(invalid_print_metrics)):
        print_string = "Invalid print metrics to be evaluated:"
        invalid_print_metrics = set(invalid_print_metrics)
        for i, invalid_print_metric in enumerate(invalid_print_metrics):
            print_string += " " + invalid_print_metric + ("," if i < len(invalid_print_metrics) - 1 else "")
        print(print_string)

    if (bool(unknown_print_metrics)):
        print_string = "Unknown print metrics to be evaluated:"
        unknown_print_metrics = set(unknown_print_metrics)
        for i, unknown_print_metric in enumerate(unknown_print_metrics):
            print_string += " " + unknown_print_metric + ("," if i < len(unknown_print_metrics) - 1 else "")
        print(print_string)


    # save_best_metric potential irregularities
    if (save_best_metric == False):
        if (need_to_warn): print("\n")
        print("##############################\n# WARNING - SAVE BEST METRIC #\n##############################")
        print("THE BEST MODEL WILL NOT BE SAVED, since save_best_metric=" +
              str(save_best_metric) + ", by user's design.")
    else:
        if (save_best_metric in _names.METRICS):
            if (_names.METRICS[save_best_metric] not in valid_metrics_names):
                print("\n\n####################\n# SAVE BEST METRIC #\n####################")
                print("The best model will be saved on the Loss metric, since save_best_metric (" +
                      str(save_best_metric) + ") is invalid.")
        else:
            print("\n\n####################\n# SAVE BEST METRIC #\n####################")
            print("The best model will be saved on the Loss metric, since save_best_metric (" +
                  str(save_best_metric) + ") is unknown.")


    return valid_metrics_names, valid_convergence_metrics, valid_print_metrics, valid_save_best_metric