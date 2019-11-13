"""
##################################################
##################################################
## This module defines the main argument        ##
## parser, for ease of running the trainer and  ##
## models using the terminal.                   ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

from   argparse import ArgumentParser
import pickle
import torch
import warnings
import os


# *** Import own functions. *** #

import helpers.names as _names
import helpers.classes_instantiator as _classes
import helpers.paths as _paths
import datasets.data_paths as _data_paths
from   datasets.data_loaders.into_trainer import PadCollateMetaDataloader, PadCollateDecoderPreTraining
from   datasets.data_loaders.into_trainer import PadCollateEmbeddingLayerPreTraining
from   helpers.general_helpers import pretty_dict_json_load, file_len, isfloat, print_warning, join_path





###########################
##### ARGUMENT PARSER #####
###########################

class Settings:
    """
    Wrapper for argument parser.
    """

    parser = ArgumentParser('Main script settings.', fromfile_prefix_chars='@')



    # Training and evaluation related arguments.
    parser.add_argument('--max_num_epochs', type=int, default=0,
                        help="Specifies the maximum number of epochs for which the model will be trained.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specifies the number of instances in a minibatch (for training).")
    parser.add_argument('--accumulate_n_batch_grads', type=int, default=1,
                        help="Identifies how many mini-batches will be computed before actually backpropagating. The " +
                             "loss will be summed for all those mini-batches (and averaged accordingly), before " +
                             "backpropagating.")
    parser.add_argument('--batch_size_eval', type=int, default=-1,
                        help="Specifies the number of instances in a minibatch (for evaluation splits). -1 means " +
                             "the entire validation sets will be computed in one go.")
    parser.add_argument('--optimiser', default=None,
                        help="Specifies the optimiser that will be used to train the model.")
    parser.add_argument('--learning_rate', type=float, default=None,
                        help="Specifies the learning rate to be used with the corresponding optimiser.")
    parser.add_argument('--weight_decay', type=float, default=None,
                        help="Specifies the weight decay parameter to be used with L2 regularisation.")
    parser.add_argument('--loss_function_name', default=None,
                        help="Specifies the loss function that will be used to train the model.")
    parser.add_argument('--loss_debug', default=False,
                        help="Specifies whether special components of the loss, like KL values, are meant to be " +
                             "displayed, in order to help theoretically debug a model.")


    # Dataset related arguments.
    parser.add_argument('--dataset', default=None,
                        help="The name of the dataset that is to be loaded into memory. Existing datasets: MNIST, " +
                             "UW-RE-UVA")
    parser.add_argument('--validation', default=False,
                        help="Specifies whether a validation split is meant to be used or not.")
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help="Specifies on which epochs to evaluate the model's performance on the validation split.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Specifies the number of independent sub-processes that will be used to load data.")
    # Dataset parameters regarding the experiment type.
    parser.add_argument('--masking_type', default=None,
                        help="Determines what kind of masking experiment is to be conducted.")
    parser.add_argument('--dataset_type', default=None,
                        help="Determines whether the dataset to be used is meant to be a very small subset of the " +
                             "entire data, with the purpose of testing an implementation (DEBUG), a small subset of " +
                             "the entire data, with the purpose of tuning hyperprameters (hyperparameter_tuning), or " +
                             "a bigger version, with the purpose fo performing a final evaluation of the model's " +
                             "performance (final_evaluation).")
    parser.add_argument('--setting', default=None,
                        help="Determines the kind of setting (e.g. 'normal, (G)zero-shot(O/C)', '(G)few-shot-n', etc.) on " +
                             "which the model is to be evaluated.")
    parser.add_argument('--fold', default=None,
                        help="Determines the fold, of the respective dataset_type and setting, on " +
                             "which the model is to be evaluated.")
    parser.add_argument('--ulbs_type', default=None,
                        help="Determines the kind of unsupervised labels to be used.")
    parser.add_argument('--classes_descs_embed_file', default=None,
                        help='Indicates the file that contains the embeddings of the relation descriptions associated' +
                             'with a UW_RE_UVA dataset')


    # Specific state related arguments.
    parser.add_argument('--epoch_or_best', type=int, default=None,
                        help="An integer identifying the epoch of the states to be loaded. -1 indicates the states" +
                             "associated with the best performance, so far.")
    parser.add_argument('--keep_last_n_chekpoints', type=int, default=-1,
                        help="An integer identifying how many of the last checkpoint saved epochs are mean to be " +
                             "kept. '-1' means all, '0' means none, and every other positive integer means keep the " +
                             "corresponding many last")
    parser.add_argument('--save_path', default=None,
                        help="A path to where checkpoints should be saved.")


    # Model related arguments.
    parser.add_argument('--model_name', default=None,
                        help="Identifies the type of model to be trained/evaluated.")
    parser.add_argument('--load_model_arch', default=None,
                        help="A path that leads to a specific model architecture that is meant to be loaded.")
    parser.add_argument('--load_model_params_from_dict_file', default=None,
                        help="The name of the file containing a dict with at least some of the model parameters.")
    parser.add_argument('--model_eval_metrics_dict_file', default=None,
                        help="The name of the file containing a dict with the metrics that will be used to evaluate " +
                             "the corresponding model.")


    # Other arguments.
    parser.add_argument('--cuda_device', default=None,
                        help="Identifies the cuda device that is meant to be used. If 'None' it is assumed that the" +
                             "program is meant to be run entirely on CPU.")
    parser.add_argument('--server_scratch', default=None,
                        help="If specified indicates the name of the remote server in which the code is running and"
                             "identifies whether the scratch partition is to be used.")


    # Parse known arguments.
    args, unknown = parser.parse_known_args()


    # Show the user which command line arguments are actually in use.
    # TODO: Make this, when checking for argument validity

    # TODO: Make this a warning
    # This identifies arguments that might have been mis-written.
    unknown_and_not_commented = [arg for arg in unknown if (arg != "" and arg[0] != "#")]
    if (bool(unknown_and_not_commented)):
        print("\nThe following command line arguments are unknown:" + str(unknown_and_not_commented))
        print("\n############################################\n")





#############################################
##### ARGUMENT VALIDITY CHECK FUNCTIONS #####
#############################################

def validate_argparse_args(args):
    """
    Function that validates the given input console arguments (and some of the arguments in the correspondingly
    specified dictionaries).


    :param args: The arguments as interpreted by the argparse library.


    :return: Two dictionaries: the first contains the valid arguments and the second contains auxiliar arguments.
    """

    valid_args = {}
    aux_args   = {}

    mem_params = {}

    # We validate the model's name first, as some of the other arguments' validity depends on it.
    # Validate 'model_name'.
    if (args.model_name is not None):
        if (args.model_name in _names.MODELS):
            valid_args["model_name"] = _names.MODELS[args.model_name]
        else:
            raise ValueError("Invalid command line argument value for --model_name=" + str(args.model_name) + " as " +
                             "it does not correspond to a known model.")


    # ******************************** #
    # *** Validate other arguments *** #
    # ******************************** #

    # Validate 'cuda_device'.
    if (args.cuda_device is not None):
        if (torch.cuda.is_available()):
            n_cuda_devices = torch.cuda.device_count()
            if (args.cuda_device.isdigit() and int(args.cuda_device) >= 0 and int(args.cuda_device) < n_cuda_devices):
                valid_args["device"] = torch.device('cuda:' + args.cuda_device)
                torch.cuda.set_device(valid_args["device"])
            else:
                raise ValueError("Selected cuda device should be represented by an integer between 0 and " +
                                 "(" + str(n_cuda_devices) + "- 1). Current value: " + str(args.cuda_device) + ".")
        else:
            warnings.warn("A cuda device was specified, but CUDA is not available. Using CPU instead.")
            valid_args["device"] = torch.device('cpu')
    else:
        valid_args["device"] = torch.device('cpu')


    # Validate 'server_scratch', for the LISA cluster.
    if (args.server_scratch is not None):
        if (args.server_scratch in _names.REMOTE_SERVERS):
            if (_names.REMOTE_SERVERS[args.server_scratch] == 'Lisa'):
                # Model related paths
                tmpdir = os.popen("echo $TMPDIR").read().strip()

                # Data paths
                _data_paths.MNIST                     = join_path([tmpdir, _data_paths.MNIST])
                _data_paths.UW_RE_UVA                 = join_path([tmpdir, _data_paths.UW_RE_UVA])
                _data_paths.UW_RE_UVA_PROPOSED_SPLITS = join_path([tmpdir, _data_paths.UW_RE_UVA_PROPOSED_SPLITS])
        else:
            raise ValueError("A remote server name was specified, but it corresponds to no known remote server.")




    # ********************************************************** #
    # *** Validate training and evaluation related arguments *** #
    # ********************************************************** #

    # Validate 'max_num_epochs' setting.
    if (args.max_num_epochs < 0):
        raise ValueError("Maximum number of epochs (max_num_epochs) size needs to be a non-negative integer.")
    else:
        mem_params["max_num_epochs"] = args.max_num_epochs


    # Validate training 'batch_size' setting.
    if (args.batch_size < 1):
        raise ValueError("Training batch size needs to be a positive integer.")
    else:
        mem_params["batch_size"] = args.batch_size


    # Validate 'accumulate_n_batch_grads'.
    if (args.accumulate_n_batch_grads < 1):
        err_msg = "Command line parameter '--accumulate_n_batch_grads' needs to be an integer greater or equal to '1'. "
        err_msg += "This is currently not the case. '--accumulate_n_batch_grads'="
        err_msg += str(args.accumulate_n_batch_grads) + "."
        raise ValueError("err_msg")
    else:
        mem_params['accumulate_n_batch_grads'] = args.accumulate_n_batch_grads


    # Validate evaluation 'batch_size' setting.
    if (args.batch_size_eval < -1 or args.batch_size_eval == 0):
        raise ValueError("Evaluation batch size needs to be a positive integer or '-1' if the entire split is to be " +
                         "evaluated in one go.")
    else:
        mem_params["batch_size_eval"] = args.batch_size_eval


    # Validate 'optimiser' setting.
    if (args.optimiser is not None):
        if (args.optimiser in _names.OPTIMISERS):
            mem_params['optimiser'] = _names.OPTIMISERS[args.optimiser]
        else:
            raise ValueError("Invalid command line argument value for --optimiser=" + str(args.optimiser) + " as " +
                             "it does not correspond to a known optimiser.")


    # Validate 'learning_rate'.
    if (args.learning_rate is not None):
        if (args.learning_rate <= 0):
            err_msg = "Command line parameter '--learning_rate' needs to be a float greater than '0.' (usually smaller"
            err_msg += "than '1.'). This is currently not the case. '--learning_rate'="
            err_msg += str(args.learning_rate) + "."
            raise ValueError("err_msg")
        else:
            mem_params['learning_rate'] = args.learning_rate
    else:
        raise RuntimeError(" It is necessary to specify a valid learning rate in order to train a model when using a " +
                           "stochastic optimiser.")


    # Validate 'weight_decay'.
    if (args.weight_decay is not None):
        if (args.weight_decay < 0):
            err_msg = "Command line parameter '--weight_decay' needs to be a float greater than or equal to '0.' "
            err_msg += "(usually smaller than '1.'). This is currently not the case. '--weight_decay'="
            err_msg += str(args.weight_decay) + "."
            raise ValueError("err_msg")
        else:
            mem_params['weight_decay'] = args.weight_decay


    # Validate 'loss_function_name' setting.
    if (args.loss_function_name is not None):
        if (args.loss_function_name in _names.LOSSES):
            mem_params['loss_function_name'] = _names.LOSSES[args.loss_function_name]
        else:
            raise ValueError("Invalid command line argument value for --loss_function_name=" +
                             str(args.loss_function_name) + " as it does not correspond to a known loss function.")


    # Validate 'loss_debug' setting.
    if (any(args.loss_debug == value for value in [True, "True", "true", False, "False", "false"])):
        valid_args["loss_debug"] = any(args.loss_debug == value for value in [True, "True", "true"])
    else:
        raise ValueError("Invalid command line argument value for '--loss_debug': " +
                         str(args.loss_debug))



    # ****************************************** #
    # *** Validate dataset related arguments *** #
    # ****************************************** #
    # We verify most dataset parameters here. Some of these will be used by the model, in specifying parts of itself
    # that might depend on the dataset at hand. On the other hand, some elements concerning the dataset, specifically
    # the DataLoader, will depend on the actual configuration of the model, so we verify those later.

    # Validate choice of 'dataset'.
    if (args.dataset is not None):
        if (args.dataset in _names.DATASETS):
            valid_args["dataset"] = _names.DATASETS[args.dataset]
        else:
            raise ValueError("Invalid command line argument value for --dataset=" + str(args.dataset) + " as " +
                             "it cannot be found in _names.DATASETS.")
    else:
        raise RuntimeError("Missing command line argument '--dataset', necessary to identify the dataset that will " +
                           "be used to train the model.")


    # Load dataset parameters.
    path_dataset_params = _paths.params_dicts + _names.DATASETS[args.dataset] + "_params.txt"
    with open(path_dataset_params, 'r') as f:
        dataset_params = pretty_dict_json_load(f)


    # Validate 'validation' setting.
    if (any(args.validation == value for value in [True, "True", "true", False, "False", "false"])):
        valid_args["validation"] = any(args.validation == value for value in [True, "True", "true"])
    else:
        raise ValueError("Invalid command line argument value for '--validation': " +
                             str(args.validation))


    # Validate 'check_val_every_n_epoch' setting.
    if (args.check_val_every_n_epoch < 1):
        raise ValueError("The parameter 'check_val_every_n_epoch' needs to be a positive integer.")
    else:
        mem_params["check_val_every_n_epoch"] = args.check_val_every_n_epoch


    # Validate 'num_workers' setting.
    if (args.num_workers < 0):
        raise ValueError("The command line argument '--num_workers' is invalid as it needs to be either 0 (load the " +
                         "data in the same process as the one running main) or a positive integer (which indicates " +
                         "the number of sub-processes responsible for loading data).")
    else:
        mem_params["num_workers"] = args.num_workers


    # Validate the rest of the parameters concerning the dataset.
    # TODO: properly verify all parameters.
    if (valid_args["dataset"] == 'MNIST'):
        if (not valid_args["validation"]):
            dataset_params['validation_size'] = 0

    elif (valid_args["dataset"] == 'UW-RE-UVA-DECODER-PRE-TRAIN' or
          valid_args["dataset"] == 'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN' or
          valid_args["dataset"] == 'UW-RE-UVA'):

        # Validate 'masking_type' setting.
        if (args.masking_type in ['unmasked', 'sub_obj_masking', 'NER_masking']):
            dataset_params["masking_type"] = args.masking_type
        else:
            raise ValueError("Invalid command line argument value for '--masking_type': " +
                             str(args.masking_type))


        if (valid_args["dataset"] == 'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN'):
            # Validate 'dataset_type' setting.
            if (args.dataset_type.lower() in ['debug', 'non_debug']):
                if (args.dataset_type.lower() == 'debug'):
                    dataset_params["dataset_type"] = args.dataset_type.upper()
                else:
                    dataset_params["dataset_type"] = args.dataset_type.lower()
            else:
                raise ValueError("Invalid command line argument value for '--dataset_type': " +
                                 str(args.dataset_type))
        else:
            # Validate 'dataset_type' setting.
            if (args.dataset_type.lower() in ['debug', 'hyperparameter_tuning', 'final_evaluation']):
                if (args.dataset_type.lower() == 'debug'):
                    dataset_params["dataset_type"] = args.dataset_type.upper()
                else:
                    dataset_params["dataset_type"] = args.dataset_type.lower()
            else:
                raise ValueError("Invalid command line argument value for '--dataset_type': " +
                                 str(args.dataset_type))


            # Validate 'setting' setting.
            if (args.setting.upper() in _names.SETTING_TYPES[dataset_params["dataset_type"]]):
                dataset_params["setting"] = args.setting.upper()
            else:
                raise ValueError("Invalid command line argument value for '--setting': " +
                                 str(args.setting))

            # Validate 'fold' setting.
            if (int(args.fold) >= 0 and int(args.fold) <= _names.FOLD_NUMS[dataset_params["dataset_type"]]):
                dataset_params["fold"] = args.fold
            else:
                raise ValueError("Invalid command line argument value for '--fold': " +
                                 str(args.fold))

            # Validate 'ulbs_type' setting.
            if (args.ulbs_type.lower() in ['basic']):
                valid_args["ulbs_type"] = args.ulbs_type.lower()
            else:
                raise ValueError("Invalid command line argument value for '--ulbs_type': " +
                                 str(args.ulbs_type))

        if (valid_args["dataset"] == 'UW-RE-UVA-DECODER-PRE-TRAIN'):
            dataset_params['ulbs_type'] = valid_args['ulbs_type']

        elif (valid_args["dataset"] == 'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN'):
            mem_params["collate_fn"] = PadCollateEmbeddingLayerPreTraining()

        else:
            dataset_params['validation'] = valid_args["validation"]

            # For UW-RE-UVA the dataloader always has a batch size of 1, and the actual mini-batch size is handled
            # internally by the dataset.
            dataset_params['batch_size_xs'] = mem_params["batch_size"]
            mem_params["batch_size"] = 1
            dataset_params['batch_size_xs_eval'] = mem_params["batch_size_eval"]
            mem_params["batch_size_eval"] = 1
            dataset_params['num_workers_xs'] = mem_params["num_workers"]
            mem_params["num_workers"] = 0
            mem_params["collate_fn"] = PadCollateMetaDataloader()

            if (valid_args["model_name"] == 'RE_BoW'):
                dataset_params['labels_to_load'] = ['supervised_lbls', 'u_sentence_lbls']

    if (valid_args["dataset"] != 'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN'):
        del valid_args['validation']
        del valid_args["ulbs_type"]



    # ********************************************** #
    # *** Validate load specific state arguments *** #
    # ********************************************** #

    # Validate 'epoch_or_best' setting
    if (args.epoch_or_best is not None):
        err_msg = "Command line parameter '--epoch_or_best' needs to be an integer greater or equal to '-1'. This is "
        err_msg += "currently not the case. '--epoch_or_best'=" + str(args.epoch_or_best) + "."

        try:
            epoch_or_best = int(args.epoch_or_best)
        except ValueError:
            raise ValueError(err_msg) from None

        if (args.epoch_or_best < -1):
            raise ValueError(err_msg)


    # Validate 'keep_last_n_chekpoints' setting
    if (args.keep_last_n_chekpoints < -1):
        err_msg = "Command line parameter '--keep_last_n_chekpoints' needs to be an integer greater or equal to '-1'. "
        err_msg += "This is currently not the case. '--keep_last_n_chekpoints'=" + str(args.keep_last_n_chekpoints) + "."
        raise ValueError("err_msg")
    else:
        mem_params['keep_last_n_chekpoints'] = args.keep_last_n_chekpoints


    # Validate 'save_path' setting
    if (args.save_path is not None):
        directory = os.path.dirname(args.save_path)
        if (not os.path.exists(directory)):
            os.makedirs(directory)
        mem_params['save_path'] = args.save_path



    # ******************************************************* #
    # *** Validate model related arguments and parameters *** #
    # ******************************************************* #

    # Check if it is at all possible to instantiate the model.
    if (args.load_model_arch is None and args.load_model_params_from_dict_file is None):
        raise RuntimeError("There is no way to create the specified model, as no previous architecture " +
                           "('--load_model_arch') or a parameters' dictionary file ('--load_params_from_dict_file') " +
                           "was provided.")

    if (args.load_model_arch is not None and args.load_model_params_from_dict_file is not None):
        raise RuntimeWarning("Both a previous architecture and a parameter dict file were provided as command line" +
                             "arguments. At this point the extension of a previous architecture has not been " +
                             "implemented, so the '--load_params_from_dict_file' will be ignored.")

    # Validate 'relations_embeddings_file' setting. TODO: Verify from the start that the file exists?
    if (args.classes_descs_embed_file is not None):
        valid_args["classes_descs_embed_file"] = join_path([_data_paths.UW_RE_UVA, args.classes_descs_embed_file])
    else:
        raise ValueError("The command line argument '--classes_descs_embed_file' needs to be provided when working " +
                         "with the UW-RE-UVA dataset.")

    aux_args['create_model'] = True
    # Here we check whether some training has already been performed for this model (by checking if the 'save_path' dir
    # already has some saved states).
    if (os.path.isdir(join_path([mem_params['save_path']] + ['Model']))):
        # If it is already specified that we should load a state from model, then we do not need to worry. Otherwise:
        if ('epoch_or_best' not in mem_params):
            # The model is the last component to be saved, so we select the state to load as being the last model state
            # that has been saved.
            model_files = os.listdir(join_path([mem_params['save_path']] + ['Model']))
            last_ckp_state = max([int(file.split('.')[-2]) for file in model_files if file.endswith('.ckp')])
            if (last_ckp_state != 0):
                print_warning(
                    'Reloading state \'' + str(last_ckp_state) + '\' from memory, as some training has already ' +
                    'happened.')
                aux_args['create_model'] = False

                mem_params['load_path'] = mem_params['save_path']
                mem_params['load_epoch_or_best'] = last_ckp_state


    if (aux_args['create_model']):
        if (args.load_model_arch is None):
            with open(_paths.params_dicts + args.load_model_params_from_dict_file, 'r') as f:
                model_params = pretty_dict_json_load(f)
                validate_model_parameters(valid_args, dataset_params, model_params)
                valid_args['model_params'] = model_params
        else:
            if (args.epoch_or_best is None):
                model_params = {'load_path': args.load_model_arch}
                if (valid_args['model_name'] == 'RE_BoW'):
                    model_params['encoder_params_dict']['classes_descs_embed_file'] = valid_args['classes_descs_embed_file']
                valid_args['model_params'] = model_params
            else:
                aux_args['create_model'] = False
                mem_params['load_path'] = args.load_model_arch
                mem_params['load_epoch_or_best'] = epoch_or_best

    # Add special model parameters that are necessary.
    mem_params['special_model_params'] = {}
    if (valid_args['model_name'] == 'ESIM_StS'):
        mem_params['special_model_params']['classes_descs_embed_file'] = valid_args['classes_descs_embed_file']
    elif (valid_args['model_name'] == 'Baseline_BiLSTM'):
        mem_params['special_model_params']['classes_descs_embed_file'] = valid_args['classes_descs_embed_file']
    elif (valid_args['model_name'] == 'RE_BoW'):
        classes_embeds = valid_args['classes_descs_embed_file']
        mem_params['special_model_params']['encoder_params_dict']['classes_descs_embed_file'] = classes_embeds


    # Validate metrics that will be used to evaluate the model.
    if (args.model_eval_metrics_dict_file is not None):
        # TODO: properly validate the metrics.
        with open(_paths.params_dicts + args.model_eval_metrics_dict_file, 'r') as f:
            mem_params.update(pretty_dict_json_load(f))
    else:
        raise RuntimeError("There is no way to evaluate the specified model, as no metrics dict file " +
                           "(--model_eval_metrics_dict_file) was provided.")


    # Validate special loss function parameters.
    special_loss_params = {}
    if (valid_args['model_name'] == 'RE_BoW'):
        special_loss_params['_DEBUG'] = valid_args['loss_debug']

        if ('special_loss_params' in model_params):
            if ('full_kl_step' in model_params['special_loss_params']):
                full_kl_step = model_params['special_loss_params']['full_kl_step']
                if (not isinstance(full_kl_step, int) or  full_kl_step <= 0):
                    raise ValueError("The step at which the KL Annealing terminates should be a positive integer. " +
                                     "Current value: (" + str(full_kl_step) + ").")
                special_loss_params['full_kl_step'] = full_kl_step

            if ('alpha_prior_val' in model_params['special_loss_params']):
                alpha_prior_val = model_params['special_loss_params']['alpha_prior_val']
                if (alpha_prior_val is not None and (not isinstance(alpha_prior_val, float) or alpha_prior_val <= 0)):
                    raise ValueError("The parameter of the Dirichlet prior should be a positive float. " +
                                     "Current value: (" + str(alpha_prior_val) + ").")
                special_loss_params['alpha_prior_val'] = alpha_prior_val

            if ('instance_prior' in model_params['special_loss_params']):
                instance_prior = model_params['special_loss_params']['instance_prior']
                if (any(instance_prior == value for value in [True, "True", "true", False, "False", "false"])):
                    special_loss_params["instance_prior"] = any(instance_prior == value for value in [True, "True", "true"])
                else:
                    raise ValueError("Loss' special parameter 'instance_prior' should be either 'True' or 'False', " +
                                     "indicating that a different prior will be sampled for each instance or the same" +
                                     " prior for the entire batch, respectively. Current value: " +
                                     str(instance_prior) + ").")
                special_loss_params['instance_prior'] = instance_prior
    mem_params['special_loss_params'] = special_loss_params



    # ********************************************* #
    # *** Validate Dataloader related arguments *** #
    # ********************************************* #

    # If we are pre-training the encoder, specify the path to the relation embeddings and the way of get relation
    # description encodings, for the Dataloader.
    if (valid_args['dataset'] == 'UW-RE-UVA-DECODER-PRE-TRAIN'):
        dataset_params['path_to_embeddings'] = valid_args["classes_descs_embed_file"]
        mem_params["collate_fn"] = PadCollateDecoderPreTraining(model_params['rels_descs_embed_type'])

    mem_params['device'] = valid_args['device']
    if (aux_args['create_model']):
        model_params['device'] = valid_args["device"]
    else:
        if ('special_model_params') not in mem_params:
            mem_params['special_model_params'] = {}
        mem_params['special_model_params']['device'] = valid_args["device"]



    valid_args['dataset_params'] = dataset_params
    valid_args['mem_params'] = mem_params


    print("The following command line arguments have been specified:\n")
    for arg in valid_args:
        print(arg + "=" + str(valid_args[arg]))

    print("\n\n\n############################################\n\n")
    return valid_args, aux_args




def validate_model_parameters(valid_args, dataset_params, model_params):
    """
    This function helps validate the parameters of the selected model.


    :param valid_args    : The valid arguments that have been processed so far.

    :param dataset_params: The valid dataset parameters that have been processed so far.

    :param model_params  : The valid model parameters that have been processed so far.


    :return: Nothing
    """

    # If the model is a Multi Layer Perceptron.
    if (valid_args["model_name"] == 'MLP'):
        verify_mlp_model_parameters(model_params)


    # If the model is our ESIM Set to Set Relation Classification model.
    elif (valid_args["model_name"] == 'ESIM_StS'):
        model_params['classes_descs_embed_file'] = valid_args['classes_descs_embed_file']
        verify_ESIM_StS_encoder_params(model_params)


    # If the model is our simple Baseline BiLSTM Relation Classification model.
    elif (valid_args["model_name"] == 'Baseline_BiLSTM'):
        model_params['classes_descs_embed_file'] = valid_args['classes_descs_embed_file']

        # Verify model's word embedding size.
        if ('word_embedding_size' not in model_params):
            raise KeyError("The Baseline_BiLSTM model requires a specified 'word_embedding_size' parameter.")
        else:
            if (model_params['word_embedding_size'] != 1024 and model_params['word_embedding_size'] != 3 * 1024):
                raise ValueError("The Baseline_BiLSTM model expects ELMO embeddings, which have a size of either " +
                                 "1024 or 3*1024. Current value: " + str(model_params['word_embedding_size']) + ".")

        # Verify Baseline_BiLSTM' first LSTM's hidden size.
        if ('first_bilstm_hidden_size' not in model_params):
            raise KeyError("Baseline_BiLSTM requires a specified 'first_bilstm_hidden_size' parameter.")
        else:
            try:
                model_params['first_bilstm_hidden_size'] = verify_layer_size(model_params['first_bilstm_hidden_size'],
                                                                             0)
            except ValueError:
                raise ValueError("Parameter 'first_bilstm_hidden_size' is not a valid layer size. Should " +
                                 "be a positive integer. Current value: " +
                                 str(model_params['first_bilstm_hidden_size'])) from None

        # Verify Baseline_BiLSTM' single embedding bilstm parameter.
        if ('single_embedding_bilstm' not in model_params):
            raise KeyError("Baseline_BiLSTM encoder requires a specified 'single_embedding_bilstm' parameter.")
        else:
            s_e_b = model_params['single_embedding_bilstm']
            if (any(s_e_b == value for value in [True, "True", "true", False, "False", "false"])):
                model_params["single_embedding_bilstm"] = any(s_e_b == value for value in [True, "True", "true"])
            else:
                raise ValueError("Baseline_BiLSTM' parameter 'single_embedding_bilstm' should be either 'True' or " +
                                 "'False', indicating whether to use a common (single) BiLSTM to encode both sentences " +
                                 "and classes descriptions or whether to use separate BiLSTMs. Current value: "
                                 + str(s_e_b) + ").")


    # If the model is our Bag of Words Unsupervised Relation Classification Decoder.
    elif (valid_args["model_name"] == 'RE_BoW_DECODER'):
        verify_RE_BoW_DECODER_params(dataset_params, model_params)


    # If the model is our ESIM Embedding layer, used for pre-training purposes.
    elif (valid_args["model_name"] == 'ESIM_Embed_Layer_AE'):
        # Verify model's word embedding size.
        if ('word_embedding_size' not in model_params):
            raise KeyError("The RE_BoW model requires a specified 'word_embedding_size' parameter.")
        else:
            if (model_params['word_embedding_size'] != 1024 and model_params['word_embedding_size'] != 3 * 1024):
                raise ValueError("The RE_BoW model expects ELMO embeddings, which have a size of either 1024 or " +
                                 "3*1024. Current value: " + str(model_params['word_embedding_size']) + ".")

        # Verify ESIM_StS' first LSTM's hidden size.
        if ('first_bilstm_hidden_size' not in model_params):
            raise KeyError("ESIM_Embed_Layer_AE requires a specified 'first_bilstm_hidden_size' parameter.")
        else:
            try:
                model_params['first_bilstm_hidden_size'] = verify_layer_size(model_params['first_bilstm_hidden_size'],
                                                                             0)
            except ValueError:
                raise ValueError("Parameter 'first_bilstm_hidden_size' is not a valid layer size. Should " +
                                 "be a positive integer. Current value: " +
                                 str(model_params['first_bilstm_hidden_size'])) from None


    # If the model is our VAE-like Bag of Words Unsupervised Relation Classification model.
    elif (valid_args["model_name"] == 'RE_BoW'):

        # *** Verify parameters common to both the encoder and the decoder. *** #
        # Verify model's word embedding size.
        if ('word_embedding_size' not in model_params):
            raise KeyError("The RE_BoW model requires a specified 'word_embedding_size' parameter.")
        else:
            if (model_params['word_embedding_size'] != 1024 and model_params['word_embedding_size'] != 3 * 1024):
                raise ValueError("The RE_BoW model expects ELMO embeddings, which have a size of either 1024 or " +
                                 "3*1024. Current value: " + str(model_params['word_embedding_size']) + ".")


        # *** Verify the encoder's parameters. *** #
        if ('encoder_params_dict' not in model_params):
            raise KeyError("Cannot build (using a parameters' dictionary) a RE-BoW model if the parameter" +
                            "'encoder_params_dict' has not been specified.")
        model_params['encoder_params_dict']['word_embedding_size'] = model_params['word_embedding_size']
        model_params['encoder_params_dict']['classes_descs_embed_file'] = valid_args['classes_descs_embed_file']
        verify_ESIM_StS_encoder_params(model_params['encoder_params_dict'])



        # *** Verify the decoder's parameters. *** #
        if ('decoder_params_dict' not in model_params):
            raise KeyError("Cannot build (using a parameters' dictionary) a RE-BoW model if the parameter" +
                            "'decoder_params_dict' has not been specified.")
        model_params['decoder_params_dict']['word_embedding_size'] = model_params['word_embedding_size']
        efbhs = model_params['encoder_params_dict']['first_bilstm_hidden_size']
        model_params['decoder_params_dict']['first_bilstm_hidden_size'] = efbhs
        verify_RE_BoW_DECODER_params(dataset_params, model_params['decoder_params_dict'])
        del model_params['decoder_params_dict']['first_bilstm_hidden_size']


    else:
        raise NotImplementedError




def adapt_model_params_for_experiment(model_type, dataset_params, model_params):
    """
    This function automatically adapts the model's parameters to match the experiment (Any-Shot Learning,
    Unsupervised Relation Classification, ...) being performed.


    :param model_type    : The model in question, from the existing model types in 'Code/models/'

    :param dataset_params: The specified parameters that identify the dataset and its properties.

    :param model_params  : The specific parameters that define most of the model's architecture.


    :return: Nothing
    """

    if (model_type == 'MLP'):
        pass

    elif (model_type == 'ESIM_StS'):
        pass

    elif (model_type == 'RE_BoW_DECODER'):
        if (dataset_params['setting'] in ['ZS-O', 'ZS-C', 'GZS-O', 'GZS-C']):
            setting = dataset_params['setting'].split('-')[0]
        else:
            setting = dataset_params['setting']
        path = join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS, dataset_params['masking_type'],
                          dataset_params['dataset_type'], setting, dataset_params['fold']])

        with open(join_path([path, 'train.ulbs']), 'rb') as f:
            pickle.load(f)
            vocab_size = len(pickle.load(f))

        if ('vector_relation_encoding' in model_params and model_params['vector_relation_encoding']):
            num_train_relations = file_len(join_path([path, 'train_relations.txt']))
            model_params['num_relations'] = num_train_relations
            model_params['vocab_size']    = vocab_size

        else:
            if (model_params['rels_descs_embed_type'] == 'bilstm'):
                decoder_first_layer_size = 2 * model_params['rels_embedder_hidden_size']
            elif (model_params['rels_descs_embed_type'] == 'avg'):
                decoder_first_layer_size = model_params['word_embedding_size']
            elif (model_params['rels_descs_embed_type'] == 'esim'):
                decoder_first_layer_size = 2 * model_params['first_bilstm_hidden_size']
            else:
                raise NotImplementedError

            inner_layers = model_params['mlp_params_dict']['layers_sizes']
            all_layers = [decoder_first_layer_size] + inner_layers + [vocab_size]
            model_params['mlp_params_dict']['layers_sizes'] = all_layers

    elif (model_type == 'RE_BoW'):
        pass

    else:
        raise NotImplementedError




def verify_ESIM_StS_encoder_params(params_dict):
    """
    This function verifies the parameters of our ESIM Set to Set model.


    :param params_dict: The defined model parameters.


    :return: Nothing
    """

    # Verify ESIM_StS' word embedding size.
    if ('word_embedding_size' not in params_dict):
        raise KeyError("ESIM_StS encoder requires a specified 'word_embedding_size' parameter.")
    else:
        if (params_dict['word_embedding_size'] != 1024 and params_dict['word_embedding_size'] != 3*1024):
            raise ValueError("ESIM encoder expects ELMO embeddings, which have a size of either 1024 or 3*1024. " +
                             "Current value: " + str(params_dict['word_embedding_size']) + ".")

    # Verify ESIM_StS' single embedding bilstm parameter.
    if ('single_embedding_bilstm' not in params_dict):
        raise KeyError("ESIM_StS encoder requires a specified 'single_embedding_bilstm' parameter.")
    else:
        s_e_b = params_dict['single_embedding_bilstm']
        if (any(s_e_b == value for value in [True, "True", "true", False, "False", "false"])):
            params_dict["single_embedding_bilstm"] = any(s_e_b == value for value in [True, "True", "true"])
        else:
            raise ValueError("ESIM_StS' parameter 'single_embedding_bilstm' should be either 'True' or " +
                             "'False', indicating whether to use a common (single) BiLSTM to encode both sentences " +
                             "and classes descriptions or whether to use separate BiLSTMs. Current value: "
                             + str(s_e_b) + ").")

    # Verify ESIM_StS' first LSTM's hidden size.
    if ('first_bilstm_hidden_size' not in params_dict):
        raise KeyError("ESIM_StS encoder requires a specified 'first_bilstm_hidden_size' parameter.")
    else:
        try:
            params_dict['first_bilstm_hidden_size'] = verify_layer_size(params_dict['first_bilstm_hidden_size'], 0)
        except ValueError:
            raise ValueError("Parameter 'first_bilstm_hidden_size' is not a valid layer size. Should " +
                             "be a positive integer. Current value: " +
                             str(params_dict['first_bilstm_hidden_size'])) from None

    # Verify ESIM_StS' post attention size.
    if ('post_attention_size' not in params_dict):
        raise KeyError("ESIM_StS encoder requires a specified 'post_attention_size' parameter.")
    else:
        try:
            params_dict['post_attention_size'] = verify_layer_size(params_dict['post_attention_size'], 0)
        except ValueError:
            raise ValueError("Parameter 'post_attention_size' is not a valid layer size. Should " +
                             "be a positive integer. Current value: " +
                             str(params_dict['post_attention_size'])) from None

    # Verify ESIM_StS' second LSTM's hidden size.
    if ('second_bilstm_hidden_size' not in params_dict):
        raise KeyError("ESIM_StS encoder requires a specified 'second_bilstm_hidden_size' parameter.")
    else:
        try:
            params_dict['second_bilstm_hidden_size'] = verify_layer_size(params_dict['second_bilstm_hidden_size'], 0)
        except ValueError:
            raise ValueError("Parameter 'second_bilstm_hidden_size' is not a valid layer size. Should " +
                             "be a positive integer. Current value: " +
                             str(params_dict['second_bilstm_hidden_size'])) from None

    # Verify ESIM_StS' mlp score estimator layers. TODO: Should add a try catch around 'verify_mlp_model_parameters'.
    if ('score_mlp_post_first_layer' not in params_dict):
        raise KeyError("Cannot build (using a parameters dictionary) a ESIM_StS model if the parameter " +
                        "'score_mlp_post_first_layer' has not been specified.")
    else:
        score_mlp_params = params_dict['score_mlp_post_first_layer']
        verify_mlp_model_parameters(score_mlp_params)
        params_dict['score_mlp_post_first_layer_sizes'] = score_mlp_params['layers_sizes']
        params_dict['score_mlp_post_first_layer_activations'] = score_mlp_params['activation_functions']
        # TODO: This will break as ESIM is not ready to have its MLP dropout values be specified. So far I have had no
        # TODO: need for them, so I'll address this problem later.
        if ('dropout_values' in score_mlp_params):
            params_dict['score_mlp_post_first_layer_dropout_values'] = score_mlp_params['dropout_values']
        params_dict.pop('score_mlp_post_first_layer', None)




def verify_RE_BoW_DECODER_params(dataset_params, params_dict):
    """
    This function verifies the validity of our Unsupervised Relation Classification model's Bag of Words decoder's
    parameters.


    :param dataset_params: The parameters concerning the dataset to be used, which determines the experiment type.

    :param params_dict   : The specified model parameters.


    :return: Nothing
    """

    if ('load_path' not in params_dict):
        if ('vector_relation_encoding' in params_dict):
            v_r_e = params_dict['vector_relation_encoding']
            if (any(v_r_e == value for value in [True, "True", "true", False, "False", "false"])):
                params_dict["vector_relation_encoding"] = any(v_r_e == value for value in [True, "True", "true"])
            else:
                raise ValueError("RE_BoW_DECODER parameter 'vector_relation_encoding' should be either 'True' or " +
                                 "'False', indicating whether a simple relation embedding matrix is to be used as " +
                                 "the decoder or not. Current value: " + str(v_r_e) + ").")

        if ('vector_relation_encoding' in params_dict and params_dict['vector_relation_encoding']):
            if 'word_embedding_size' in params_dict: del params_dict['word_embedding_size']
            if 'rels_descs_embed_type' in params_dict: del params_dict['rels_descs_embed_type']
            if 'rels_embedder_hidden_size' in params_dict: del params_dict['rels_embedder_hidden_size']
            if 'rels_embedder_hidden_size' in params_dict: del params_dict['rels_embedder_hidden_size']
            if 'mlp_params_dict' in params_dict: del params_dict['mlp_params_dict']

        else:
            # Verify word embedding size.
            if ('word_embedding_size' not in params_dict):
                raise KeyError("RE_BoW_DECODER requires a specified 'word_embedding_size' parameter.")
            else:
                if (params_dict['word_embedding_size'] != 1024 and params_dict['word_embedding_size'] != 3 * 1024):
                    raise ValueError("RE_BoW_DECODER expects ELMO embeddings, which have a size of either 1024 or " +
                                     "3*1024. Current value: " + str(params_dict['word_embedding_size']) + ".")

            # Verify relations' descriptions' embedding type.
            if ('rels_descs_embed_type' not in params_dict):
                raise KeyError("RE_BoW_DECODER requires a specified 'rels_descs_embed_type' parameter.")
            else:
                if (params_dict['rels_descs_embed_type'] not in ['bilstm', 'avg', 'esim']):
                    raise ValueError("Parameter 'rels_embedder_hidden_size' is not a valid layer size. Should " +
                                     "be a positive integer. Current value: " +
                                     str(params_dict['rels_embedder_hidden_size'])) from None

            # Verify relations' descriptions' embedder hidden size.
            if (params_dict['rels_descs_embed_type'] == 'bilstm' and 'rels_embedder_hidden_size' not in params_dict):
                raise KeyError("RE_BoW_DECODER requires a specified 'rels_embedder_hidden_size' parameter, when using a" +
                               "bilstm relations' descriptions' embedder.")
            else:
                try:
                    params_dict['rels_embedder_hidden_size'] = verify_layer_size(
                        params_dict['rels_embedder_hidden_size'],
                        0)
                except ValueError:
                    raise ValueError("Parameter 'rels_embedder_hidden_size' is not a valid layer size. Should " +
                                     "be a positive integer. Current value: " +
                                     str(params_dict['rels_embedder_hidden_size'])) from None
            if (params_dict['rels_descs_embed_type'] != 'bilstm' and 'rels_embedder_hidden_size' in params_dict):
                del params_dict['rels_embedder_hidden_size']

        # RE_BoW_DECODER's structure depends on the experiment/dataset being performed/used. Here we account for that.
        adapt_model_params_for_experiment("RE_BoW_DECODER", dataset_params, params_dict)

        if ('vector_relation_encoding' not in params_dict or not params_dict['vector_relation_encoding']):
            verify_mlp_model_parameters(params_dict['mlp_params_dict'])




def verify_mlp_model_parameters(params_dict):
    """
    This function verifies the models of a Multi Layer Perceptron.


    :param params_dict: The specified model parameters.


    :return: Nothing
    """

    # Check if the minimum necessary parameters (layers' sizes and activation functions) has been specified.
    if ('layers_sizes' not in params_dict or 'activation_functions' not in params_dict):
        raise ValueError("It is impossible to create a MLP model (using a parameters' dictionary) without having " +
                         "specified both the layers sizes and the corresponding activation functions.")
        # raise ValueError("Attempted to create a MLP model, but no " +
        #                  ("--layers_sizes" if args.layers_sizes is None else "--activation_functions") +
        #                  " was specified.")

    # Check if the layers' sizes are valid (as in, non-negative integers).
    verify_mlp_layers_sizes(params_dict)

    # Check if the activation functions are valid.
    verify_mlp_activation_functions(params_dict)

    # Check that the number of provided activation functions matches the number of layers.
    if (len(params_dict["layers_sizes"]) - 1 != len(params_dict["activation_functions"])):
        raise ValueError("Number of layers is different from number of activation functions. " +
                         "Number of layers: " + str(len(params_dict["layers_sizes"]) - 1) + ", " +
                         "Number of activation functions: " +
                         str(len(params_dict["activation_functions"])) + ".")


    # Check if dropout values were provided. If so, check if they are correct. If not, set them to 0 and equal to the
    # number of layers.
    if ('dropout_values' in params_dict):

        # Check if the dropout values are valid.
        verify_mlp_dropout_values(params_dict)

        # Check that the number of provided dropout values matches the number of layers.
        if (len(params_dict["layers_sizes"]) - 1 != len(params_dict["dropout_values"])):
            raise ValueError("Number of layers is different from number of dropout values. " +
                             "Number of layers: " + str(len(params_dict["layers_sizes"]) - 1) + ", " +
                             "Number of dropout values: " + str(len(params_dict["dropout_values"])) + ".")




def verify_mlp_layers_sizes(params_dict):
    """
    This function verifies if the layer sizes of an MLP are valid.


    :param params_dict: The specified MLP parameters.


    :return: Nothing
    """

    spacing = " " * len('ValueError: ')
    err_msg = "Incorrect formatting of mlp's layers_sizes. Should be a python\n"
    err_msg += spacing + "list of non-negative integers, where each integer represents the layer \n"
    err_msg += spacing + "size of the layer with corresponding index.\n"
    err_msg += "\nE.g.: 3 layers, with sizes -> 784, 200, 10:\n"
    err_msg += "layers_sizes = [784, 200, 10]\n"
    err_msg += "\nCurrent layers_sizes = " + str(params_dict['layers_sizes'])

    try:
        layers_sizes = [verify_layer_size(l_size, l_num) for l_num, l_size in enumerate(params_dict['layers_sizes'])]
        params_dict['layers_sizes'] = layers_sizes
    except ValueError as error:
        raise ValueError(err_msg + ". " + str(error) + " is not a valid layer size.") from None





def verify_layer_size(layer_size, l_num):
    """
    This function verifies the validity of the specified size of a single layer.


    :param layer_size: The layer size.

    :param l_num     : The layer depth, for identification when an error occurs.


    :return: Nothing
    """

    if (isinstance(layer_size, int) and layer_size > 0):
        return layer_size
    else:
        raise ValueError("'" + str(layer_size) + "' (layer " + str(l_num + 1) + ")")




def verify_mlp_activation_functions(params_dict):
    """
    This function verifies if the activation functions of an MLP are valid.


    :param params_dict: The specified MLP parameters.


    :return: Nothing
    """

    spacing = " " * len('ValueError: ')
    err_msg = "Incorrect formatting of mlp's activation functions. Should be a python\n"
    err_msg += spacing + "list of: activation functions names OR list of the form:\n"
    err_msg += spacing + "[activation function name, +str(kwargs)]\n"
    err_msg += "\nE.g.: 2 relu layers followed by a softmax layer computed over dimension 1:\n"
    err_msg += " activation_functions = ['relu']*2 + [['softmax', 'dim=1']]\n"
    err_msg += "\nCurrent activation_functions = " + str(params_dict['activation_functions'])

    try:
        activation_functions = [verify_activation_function(af, af_num) for af_num, af in
                                enumerate(params_dict['activation_functions'])]
        params_dict["activation_functions"] = activation_functions
    except ValueError as error:
        raise ValueError(err_msg + ". " + str(error) + " is not a valid activation function.") from None




def verify_activation_function(activation_function, af_num):
    """
    This function verifies a specific layer's activation function.


    :param activation_function: The activation function name, as in the model parameters dictionary file.

    :param af_num             : The layer depth, for identification when an error occurs.


    :return: Nothing
    """

    if (isinstance(activation_function, str)):
        if (activation_function in _names.AFS):
            return _classes.AFS[_names.AFS[activation_function]]()
        else:
            raise ValueError("'" + activation_function + "'")
    elif (isinstance(activation_function, list) and all(isinstance(ele, str) for ele in activation_function)):
        if (activation_function[0] in _names.AFS):
            if (_names.AFS[activation_function[0]] == 'softmax'):
                if(len(activation_function) == 2 and activation_function[1].startswith("dim=") and
                        (activation_function[1][4:].isdigit())):
                    return _classes.AFS['softmax'](dim=int(activation_function[1][4:]))
                else:
                    raise ValueError(activation_function)
            elif (_names.AFS[activation_function[0]] == 'leaky_relu'):
                if(len(activation_function) == 2 and activation_function[1].startswith("negative_slope=") and
                        isfloat(activation_function[1][len("negative_slope="):])):
                    return _classes.AFS['leaky_relu'](negative_slope=float(activation_function[1][len("negative_slope="):]))
                else:
                    raise ValueError(activation_function)
            else:
                raise ValueError("'" + activation_function[0] + "'")
        else:
            raise ValueError("'" + activation_function[0] + "'")
    else:
        invalid_activation_function = True

    if (invalid_activation_function):
        raise ValueError("'" + str(activation_function) + "' (layer " + str(af_num + 1) + ")")




def verify_mlp_dropout_values(params_dict):
    """
    This function verifies if the dropout values of an MLP are valid.


    :param params_dict: The specified MLP parameters.


    :return: Nothing
    """

    spacing = " " * len('ValueError: ')
    err_msg = "Incorrect formatting of mlp's dropout values. Should be a python\n"
    err_msg += spacing + "list of: floats between 0. and 1. (including).\n"
    err_msg += "\nE.g.: 2 layers of dropout 0.3 and 0.7, respectively:\n"
    err_msg += " dropout_values = [0.3, 0.7]\n"
    err_msg += "\nCurrent dropout_values = " + str(params_dict['dropout_values'])

    try:
        dropout_values = [verify_dropout_value(dropout_value, l_num) for l_num, dropout_value in
                          enumerate(params_dict['dropout_values'])]
        params_dict['dropout_values'] = dropout_values
    except ValueError as error:
        raise ValueError("\n" + err_msg + ". " + str(error) + " is not a valid layer size.") from None




def verify_dropout_value(dropout_value, l_num):
    """
    This function verifies a specific layer's dropout value.


    :param dropout_value: The dropout value, as in the model parameters dictionary file.

    :param l_num        : The layer depth, for identification when an error occurs.


    :return: Nothing
    """

    if (isinstance(dropout_value, float) and dropout_value >= 0 and dropout_value < 1):
        return dropout_value
    else:
        raise ValueError("'" + str(dropout_value) + "' (layer " + str(l_num + 1) + ")")