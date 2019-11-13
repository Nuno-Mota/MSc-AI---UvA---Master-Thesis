"""
##################################################
##################################################
## This file contains the main script to be     ##
## run.                                         ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch.autograd as autograd


# *** Import own functions. *** #

from   settings import Settings, validate_argparse_args
import helpers.classes_instantiator as _classes
from   helpers.trainer.model_evaluation_module import ModelEvaluationModule
from   datasets.data_loaders.into_trainer import select_dataset
from   helpers.general_helpers import print_warning





################
##### MAIN #####
################

def main(valid_args, aux):
    """
    The main function, that receives its arguments from argparse, and allows for the training procedure to happen.


    :param valid_args: A dictionary with he valid arguments that are to be used by the model, dataset and trainer.

    :param aux       : A dictionary with auxiliary arguments, such as whether to create the model from scratch or not.


    :return: Nothing
    """

    # Auxiliary variables used to dynamically adapt the mini-batch sizes. This is useful as sometimes the selected
    # batch sizes can be on the edge of the GPU's memory capability and if a mini-batch has a set of longer sentences
    # it can cause an out of memory error. This way the program recovers automatically and allows the job to finish.
    if ('batch_size_xs' in valid_args['dataset_params']):
        original_batch_size      = valid_args['dataset_params']['batch_size_xs']
        original_batch_size_eval = valid_args['dataset_params']['batch_size_xs_eval']
    else:
        original_batch_size      = valid_args['mem_params']['batch_size']
        if (valid_args["dataset"] == 'UW-RE-UVA'):
            original_batch_size_eval = valid_args['dataset_params']['batch_size_eval']
        else:
            original_batch_size_eval = valid_args['mem_params']['batch_size_eval']

    current_accumulate_n_batch_grads = valid_args['mem_params']['accumulate_n_batch_grads']
    current_batch_size      = original_batch_size
    current_batch_size_eval = original_batch_size_eval


    # Determines whether the training procedure has converged.
    converged = False

    while (not converged):
        try:
            # Creates the model, if necessary.
            model = _classes.MODELS[valid_args["model_name"]](**valid_args['model_params']) if aux['create_model'] else None

            # Loads the dataset splits, along with the additional information regarding the mini-batch sizes.
            train_dataset, val_dataset, test_dataset = select_dataset(valid_args['dataset'], valid_args['dataset_params'])
            training_batch_sizes = train_dataset.normal_last_batch_sizes(valid_args['mem_params']["batch_size"])

            # Creates the trainer.
            mem = ModelEvaluationModule(model, train_dataset, training_batch_sizes,
                                        val_dataset=val_dataset, test_dataset=test_dataset, **valid_args['mem_params'])

            # Trains the model.
            with autograd.detect_anomaly():
                mem.train()
            converged = True

        # The out of memory error, due to variable mini-batch size, is handled here.
        except RuntimeError as err:
            # If the error occurred during training, then the mini-batch size is halved and twice as many gradients are
            # accumulated.
            if (mem.current_dataloader_type == 'train'):
                current_batch_size = int(current_batch_size/2)
                current_accumulate_n_batch_grads *= 2
                valid_args['mem_params']['accumulate_n_batch_grads'] = current_accumulate_n_batch_grads
            # If the error occurred on an evaluation split, the mini-batch size for evaluation is simply decreased by 1.
            else:
                current_batch_size_eval -= 1

            if (current_batch_size == 0 or current_batch_size_eval == 0):
                raise

            # Inform the user of the new batch sizes.
            print_warning(str(err) + ".\nAttempting to reduce physical batch size. New batch_size=" +
                          str(current_batch_size) + ', accumulate_n_batch_grads=' +
                          str(current_accumulate_n_batch_grads) + ', batch_size_eval=' +
                          str(current_batch_size_eval) + '.')

            # Update the auxiliary variables.
            if ('batch_size_xs' in valid_args['dataset_params']):
                valid_args['dataset_params']['batch_size_xs']      = current_batch_size
                valid_args['dataset_params']['batch_size_xs_eval'] = current_batch_size_eval
            else:
                valid_args['mem_params']['batch_size']          = current_batch_size
                valid_args['dataset_params']['batch_size_eval'] = current_batch_size_eval

            valid_args['mem_params']['load_epoch_or_best'] = mem.current_epoch_num - 1



if __name__ == "__main__":
    args = Settings.args
    valid_args, aux = validate_argparse_args(args)
    main(valid_args, aux)