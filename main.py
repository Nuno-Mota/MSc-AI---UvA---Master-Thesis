"""
##################################################
##################################################
## This file contains classes designed to       ##
## easily implement the VerbOcean graph KB.     ##
##                                              ##
## Each class/function will have a small        ##
## description of its use cases.                ##
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
# TODO: remove line below when it's done properly
from   datasets.data_loaders.into_trainer import PadCollateDecoderPreTraining
import helpers.paths as _paths
from   helpers.general_helpers import pretty_dict_json_load, inspect_class_parameters





################
##### MAIN #####
################

def main(valid_args, aux):
    """
    TODO
    :param args:
    :return:
    """

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


    converged = False

    while (not converged):
        try:
            model = _classes.MODELS[valid_args["model_name"]](**valid_args['model_params']) if aux['create_model'] else None

            train_dataset, val_dataset, test_dataset = select_dataset(valid_args['dataset'], valid_args['dataset_params'])
            training_batch_sizes = train_dataset.normal_last_batch_sizes(valid_args['mem_params']["batch_size"])

            mem = ModelEvaluationModule(model, train_dataset, training_batch_sizes,
                                        val_dataset=val_dataset, test_dataset=test_dataset, **valid_args['mem_params'])

            with autograd.detect_anomaly():
                mem.train()
            converged = True

        except RuntimeError as err:
            # raise
            if (mem.current_dataloader_type == 'train'):
                current_batch_size = int(current_batch_size/2)
                current_accumulate_n_batch_grads *= 2
                valid_args['mem_params']['accumulate_n_batch_grads'] = current_accumulate_n_batch_grads
            else:
                current_batch_size_eval -= 1

            if (current_batch_size == 0 or current_batch_size_eval == 0):
                raise

            print_warning(str(err) + ".\nAttempting to reduce physical batch size. New batch_size=" +
                          str(current_batch_size) + ', accumulate_n_batch_grads=' +
                          str(current_accumulate_n_batch_grads) + ', batch_size_eval=' +
                          str(current_batch_size_eval) + '.')

            if ('batch_size_xs' in valid_args['dataset_params']):
                valid_args['dataset_params']['batch_size_xs']      = current_batch_size
                valid_args['dataset_params']['batch_size_xs_eval'] = current_batch_size_eval
            else:
                valid_args['mem_params']['batch_size']          = current_batch_size
                valid_args['dataset_params']['batch_size_eval'] = current_batch_size_eval

            valid_args['mem_params']['load_epoch_or_best'] = mem.current_epoch_num - 1


    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    #
    # num_decoder_runs_per_step = 1
    # concurrent_training_num_epochs = 50
    #
    # decoder_mem_params = {key: valid_args['mem_params'][key] for key in valid_args['mem_params']}
    # decoder_mem_params['max_num_epochs'] = concurrent_training_num_epochs * num_decoder_runs_per_step
    # decoder_mem_params['learning_rate'] = 0.0075
    # decoder_mem_params['accumulate_n_batch_grads'] = 1
    # decoder_mem_params['batch_size'] = 64
    # decoder_mem_params['loss_function_name'] = 'Bag of Words Log Loss'
    # decoder_mem_params['special_loss_params'] = {}
    # decoder_mem_params['collate_fn'] = PadCollateDecoderPreTraining(model.decoder.rels_descs_embed_type)
    # decoder_mem_params['num_workers'] = 8
    # with open(_paths.params_dicts + 're_bow_decoder_metrics.txt', 'r') as f:
    #     decoder_mem_params.update(pretty_dict_json_load(f))
    #
    #
    # train_dataset, val_dataset, test_dataset = select_dataset(valid_args['dataset'], valid_args['dataset_params'])
    # training_batch_sizes = train_dataset.normal_last_batch_sizes(valid_args['mem_params']["batch_size"])
    #
    # decoder_dataset_params = {}
    # decoder_dataset_params['path_to_embeddings'] = valid_args["relations_embeds_file"]
    # decoder_dataset_params['experiment_test_dataset'] = valid_args['dataset_params']['experiment_test_dataset']
    # decoder_dataset_params['experiment_type'] = valid_args['dataset_params']['experiment_type']
    # decoder_dataset_params['ulbs_type'] = 'basic'
    # train_dataset_decoder, _, _ = select_dataset('UW-RE-UVA-DECODER-PRE-TRAIN', decoder_dataset_params)
    # training_batch_sizes_decoder = train_dataset.normal_last_batch_sizes(decoder_mem_params['batch_size'])
    #
    # mem_re_bow_decoder = ModelEvaluationModule(model.decoder, train_dataset_decoder, training_batch_sizes_decoder,
    #                                            val_dataset=None, test_dataset=None,
    #                                            **decoder_mem_params)
    # mem_re_bow = ModelEvaluationModule(model, train_dataset, training_batch_sizes,
    #                                    val_dataset=val_dataset, test_dataset=test_dataset, **valid_args['mem_params'])
    #
    # # inspect_class_parameters(mem_re_bow)
    # # print("\n\n\n")
    # # inspect_class_parameters(mem_re_bow_decoder)
    # # TODO: Add Debug Toggle
    # with autograd.detect_anomaly():
    #     for epoch_num in range(concurrent_training_num_epochs):
    #         print("RE_BoW_DECODER")
    #         if (mem_re_bow_decoder.current_epoch_num == 0):
    #             print("Performing pre-evaluation")
    #             mem_re_bow_decoder.run_epoch_train(mem_re_bow_decoder.current_epoch_num)
    #
    #         # Print past epochs, if resuming training
    #         if (mem_re_bow_decoder.current_epoch_num > 1):
    #             for epoch_num_print in range(0, mem_re_bow_decoder.current_epoch_num):
    #                 mem_re_bow_decoder.metrics_manager.print_info(epoch_num_print,
    #                                                               mem_re_bow_decoder.max_num_batches['train'],
    #                                                               step_is_epoch_num=True)
    #
    #         for _ in range(num_decoder_runs_per_step):
    #             mem_re_bow_decoder.run_epoch_train(mem_re_bow_decoder.current_epoch_num)
    #
    #         print("\n\n\n")
    #         print("RE_BoW")
    #         if (mem_re_bow.current_epoch_num == 0):
    #             print("Performing pre-evaluation")
    #             mem_re_bow.run_epoch_train(mem_re_bow.current_epoch_num)
    #         if (mem_re_bow.current_epoch_num == 3):
    #             for param in model.encoder.parameters():
    #                 param.requires_grad = True
    #
    #         # Print past epochs, if resuming training
    #         if (mem_re_bow.current_epoch_num > 1):
    #             for epoch_num_print in range(0, mem_re_bow.current_epoch_num):
    #                 mem_re_bow.metrics_manager.print_info(epoch_num_print,
    #                                                       mem_re_bow.max_num_batches['train'],
    #                                                       step_is_epoch_num=True)
    #         mem_re_bow.run_epoch_train(mem_re_bow.current_epoch_num)
    #         print("\n\n\n")
    #
    #
    #     print("STOPPING DECODER INDIVIDUAL TRAINING.\n\n")
    #     mem_re_bow.train()
    #     # mem.train()



if __name__ == "__main__":
    args = Settings.args
    valid_args, aux = validate_argparse_args(args)
    main(valid_args, aux)





###################
##### HELPERS #####
###################

def check_argument_validity():
    pass