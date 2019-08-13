"""
##################################################
##################################################
## This file implements a Model Evaluation      ##
## Module, which facilitates model training,    ##
## such as resuming training, evaluating        ##
## metrics, displaying relevant information,    ##
## evaluate the chosen model, etc...            ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
from   torch.utils.data import DataLoader
from   torch.utils.data.dataloader import default_collate
import os
import math


# *** Own modules imports. *** #

import helpers.names as _alt_names
import helpers.classes_instantiator as _classes
import helpers.paths as _paths
import helpers.trainer.helpers as trainer_helpers
from   helpers.trainer.metrics_manager import MetricsManager
from   helpers.general_helpers import remove_files, print_warning, join_path





#################
##### CLASS #####
#################

class ModelEvaluationModule(object):
    """
    The ModelEvaluationManager class aims to simplify and generalise the training procedure of models.
    """

    def __init__(self, model, train_dataset, training_batch_sizes,
                 # MOST 'None' values default to a certain value if not loaded from memory. See '_load_var()'
                 metrics_names=None, output_to_metric=None, convergence_metrics_and_criteria=None, print_metrics=None,
                 print_every_n_batch=None, check_val_every_n_epoch=None, val_dataset=None, test_dataset=None,
                 collate_fn=None, max_num_epochs=None, batch_size=None, batch_size_eval=None, num_workers=None,
                 loss_function_name=None, optimiser=None, learning_rate=None, weight_decay=None,
                 accumulate_n_batch_grads=None, save_path=None, save_best_metric=None, keep_last_n_chekpoints=None,
                 load_path=None, load_epoch_or_best=None, special_model_params={}, special_loss_params={},
                 device=torch.device('cpu'), **mem_params):
        """
        Instantiates a MetricsManager object. TODO: Added new parameters. Need to account for them.


        :param model                           : The model, if any (as it can be loaded from memory) to be evaluated.

        :param train_dataset                   : The dataset split that will be used for training the model.

        :param metrics_names                   : The names of the metrics with which the model will be evaluated.

        :param output_to_metric                : For models with multiple outputs, this identifies which metrics should
                                                 evaluate which output.

        :param convergence_metrics_and_criteria: The  names of the metrics, if any, and the corresponding parameters,
                                                 that will determine if a model has converged.

        :param print_metrics                   : The names of the metrics for which information should be displayed.

        :param print_every_n_batch             : Determines on which batch numbers the training information should be
                                                 displayed.

        :param check_val_every_n_epoch         : Determines on which epochs the model's performance on the validation
                                                 set (if it exists) should be evaluated.

        :param val_dataset                     : The dataset split, if any, that is to be used as validation, to test
                                                 the model's generalisation capabilities during training.

        :param test_dataset                    : The dataset split, if any, that is to be used to test the model's final
                                                 performance.

        :param collate_fn                      : The collate function to be sued with this datasets.

        :param max_num_epochs                  : The maximum number of epochs for which the model will be trained in
                                                 this current instantiation.

        :param batch_size                      : The number of instances that will be in each training minibatch.

        :param num_workers                     : TODO: Find out

        :param loss_function_name              : The name of the loss function that will be used to train the model.

        :param optimiser                       : The optimiser that will be used the update the model's weights during
                                                 training.

        :param save_path                       : The path to the directory on which the state of the model, the MEM, the
                                                 MMa and each metric is meant to be saved.

        :param save_best_metric                : The metric that will be used to determine whether the last epoch as
                                                 been the one where the model has performed the best.

        :param load_path                       : The path that leads to the directory where the states pertaining a
                                                 previously trained model, MEM, etc are located.

        :param load_epoch_or_best              : Determines whether to load a specific checkpoint or whether to load the
                                                 epoch where the model performed the best.
        """

        # Save the device on which the model is to be evaluated.
        self._device = device

        # Compose the correct path necessary to load the MEM's state.
        load_path = None if load_path is None else (os.path.normpath(load_path) + os.sep)
        mem_load_path = None if load_path is None else load_path + "MEM" + os.sep

        # Load, if necessary, the settings associated with a previously instantiated MEM.
        mem_settings_path = None if load_path is None else mem_load_path + "settings.stg"
        mem_settings = None if load_path is None else torch.load(mem_settings_path)

        # Load, if necessary, the state associated with a previously instantiated MEM.
        self._mem_state = trainer_helpers.load_checkpoint_state(mem_load_path, "state", load_epoch_or_best,
                                                                device=self._device)

        # Either sets self._model as the input model or loads a previous one from memory.
        self._special_model_params = special_model_params
        self._model = self._load_var(model, 'model', mem_settings, load_path, load_epoch_or_best)

        # Variables used to determine information about the batch/epoch number.
        self._current_epoch_num        = 0 if self._mem_state is None else self._mem_state['current_epoch_num']
        self._max_num_epochs           = self._load_var(max_num_epochs, 'max_num_epochs', mem_settings)
        self._batch_size               = self._load_var(batch_size, 'batch_size', mem_settings)
        # TODO: do this properly
        self._accumulate_n_batch_grads = accumulate_n_batch_grads
        self._effective_batch_size     = self._batch_size*self._accumulate_n_batch_grads


        # Variables related to the loss function and the optimiser used to train the model.
        self._loss_function_name = self._load_var(loss_function_name, 'loss_function_name', mem_settings)
        # TODO: validate and load properly the learning_rate parameter.
        self._learning_rate      = learning_rate
        self._weight_decay       = self._load_var(weight_decay, 'weight_decay', mem_settings)
        self._optimiser          = self._load_var(optimiser, 'optimiser', mem_settings, load_path, load_epoch_or_best)


        # Datasets' variables
        self._datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        self._num_workers = self._load_var(num_workers, 'num_workers', mem_settings)
        self._collate_fn  = self._load_var(collate_fn, 'collate_fn', mem_settings)
        pin_memory = False if str(device)[:3] == 'cpu' else True

        #TODO: REMOVE THE SWAP. Just used to test if (G)FS test settings are inherently harder than validation ones.

        # Train split.
        self._train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True,
                                            pin_memory=pin_memory, collate_fn=self._collate_fn,
                                            num_workers=self._num_workers)
        self._max_num_batches = {'train': math.ceil(len(train_dataset) / self._batch_size)}

        # Validation split.
        if (val_dataset is not None):
            batch_size_val = len(val_dataset) if batch_size_eval == -1 else batch_size_eval
            self._val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False,
                                              pin_memory=pin_memory, collate_fn=self._collate_fn,
                                              num_workers=self._num_workers)
            self._max_num_batches['val'] = math.ceil(len(val_dataset) / batch_size_val)
        else:
            self._val_dataloader = None

        # Test split.
        if (test_dataset is not None):
            batch_size_test = len(test_dataset) if batch_size_eval == -1 else batch_size_eval
            self._test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                                               pin_memory=pin_memory, collate_fn=self._collate_fn,
                                               num_workers=self._num_workers)
            self._max_num_batches['test'] = math.ceil(len(test_dataset) / batch_size_test)
        else:
            self._test_dataloader = None


        # Determines and creates an easy way to access the dataset splits on which the model will be evaluated.
        self._training_dataloders_types = ['train'] + (['val'] if val_dataset else [])
        self._training_dataloders = {}
        self._training_dataloders['train'] = self._train_dataloader
        if (val_dataset):
            self._training_dataloders['val'] = self._val_dataloader

        # Variables related to the metrics that will be used to evaluate the model.
        self._metrics_names    = self._load_var(metrics_names, 'metrics_names', mem_settings)
        # TODO: add the 'output to metric' variable to load_var.
        self._output_to_metric = output_to_metric
        self._convergence_metrics_and_criteria = self._load_var(convergence_metrics_and_criteria,
                                                                'convergence_metrics_and_criteria', mem_settings)
        self._print_metrics    = self._load_var(print_metrics, 'print_metrics', mem_settings)
        self._save_best_metric = self._load_var(save_best_metric, 'save_best_metric', mem_settings)

        # TODO: validate the 'output to metric' variable.
        valid_metrics = trainer_helpers.verify_valid_metrics(self._metrics_names,
                                                             self._convergence_metrics_and_criteria,
                                                             self._print_metrics, self._save_best_metric)
        self._metrics_manager = MetricsManager(valid_metrics, self._output_to_metric, self._max_num_epochs,
                                               self._max_num_batches, self._accumulate_n_batch_grads,
                                               training_batch_sizes, self._loss_function_name,
                                               bool(self._val_dataloader), bool(self._test_dataloader),
                                               load_path, load_epoch_or_best, loss_params_dict=special_loss_params)

        # We update the metrics that require class information, such as F1, MacroF1, HarmonicF1...
        for split in list(self._training_dataloders) + (['test'] if test_dataset is not None else []):
            for metric in self._metrics_manager.metrics[split]:
                if (metric in self._output_to_metric[split]):
                    classes_info = self._datasets[split].get_classes_info()
                    self._metrics_manager.metrics[split][metric].update_classes_info(*classes_info)

        self._valid_metrics_names = valid_metrics[0]
        self._valid_print_metrics = valid_metrics[2]

        # Miscellaneous
        self._print_every_n_batch = self._load_var(print_every_n_batch, 'print_every_n_batch', mem_settings)
        self._check_val_every_n_epoch = self._load_var(check_val_every_n_epoch, 'check_val_every_n_epoch', mem_settings)

        # Saving parameters
        self._save_path = self._load_var(save_path, 'save_path', mem_settings)
        # TODO: Check if 'keep_last_n_chekpoints' needs to be saved or not.
        self._keep_last_n_chekpoints = keep_last_n_chekpoints
        self._has_converged     = False if self._mem_state is None else self._mem_state['has_converged']
        self._has_computed_test = False if self._mem_state is None else self._mem_state['has_computed_test']
        self._current_dataloader_type = None




    def evaluate(self):
        """
        This method evaluates the model under the specified metrics, in a singular evaluation, decoupled from the
        training procedure.


        :return: Nothing
        """

        # If there are any metrics whose information is actually meant to be displayed.
        if (len(self._valid_print_metrics) > 0):
            self.run_epoch_train(0, evaluate_only=True)
        else:
            print("There are no valid metrics selected for printing. Model will NOT be evaluated.")




    def train(self):
        """
        This method implements the high level training procedure.

        :return: Nothing
        """

        # Print past epochs, if resuming training
        if (self._current_epoch_num > 0):
            for epoch_num in range(0, self._current_epoch_num):
                self._metrics_manager.print_info(epoch_num, self._max_num_batches['train'], step_is_epoch_num=True)

        # # TODO: Remove or implement better.
        # # Freeze encoder's parameters in initial epochs.
        # for param in self._model.encoder.parameters():
        #     param.requires_grad = False

        # Perform pre-evaluation
        if (self._current_epoch_num == 0):
            print("Performing model's pre evaluation.\n")
            self.run_epoch_train(0)
            # self._current_epoch_num += 1
            self.save()  # Saves MEM settings

        # Train
        if (self._has_converged):
            print("\nEarly stopping conditions have been satisfied. Stopping training.")
        else:
            for epoch_num in range(self._current_epoch_num, self._max_num_epochs + 1):
                # # TODO: Remove or implement better.
                # if (epoch_num == 5):
                #     # Unfreeze encoder's parameters after the initial epochs.
                #     for param in self._model.encoder.parameters():
                #         param.requires_grad = True

                # Perform training epoch iteration.
                self.run_epoch_train(epoch_num)

                # Check if early stopping conditions are satisfied
                if (self._metrics_manager.check_for_early_stopping()):
                    self._has_converged = True

                # Save trainer and model state.
                self.save()
                if (self._has_converged):
                    print("\nEarly stopping conditions have been satisfied. Stopping training.")
                    break

        print('\nFinished Training.')


        # Evaluate on the test set.
        if (self._test_dataloader is None):
            print_warning("No test dataset was provided. As such, no evaluation on it can be performed.\n" +
                          "Reload the model, with a provided test dataset, if its evaluation is required.")
        else:
            if (not self._has_computed_test):
                print('\n\nLoading the model that performed the best (given the provided criteria).', end='', flush=True)

                # We start by loading the model that performed the best (or the last one, if best one was the randomly
                # initialised model) on the validation set.
                model_files = os.listdir(self._save_path + os.sep + "Model" + os.sep)
                best_epoch  = [int(file.split('.')[-2]) for file in model_files if file.endswith('.bst')]
                if (not bool(best_epoch)):
                    last_epoch = max([int(file.split('.')[-2]) for file in model_files if file.endswith('.ckp')])
                if ('device' not in self._special_model_params):
                    self._special_model_params['device'] = self._device
                # print("\nTESTING!!!! AHOY", self._device, self._special_model_params['device'])
                self._model.__init__(load_path=self._save_path + os.sep + "Model" + os.sep,
                                     epoch_or_best=-1 if bool(best_epoch) else last_epoch,
                                     **self._special_model_params)

                print('\rEvaluating the validation\'s best performing model (Epoch: ' +
                      str(best_epoch[0] if bool(best_epoch) else last_epoch) + ') on the test set.')

                # Place the model in eval() mode.
                self._model.eval()

                # Perform evaluation for each batch in the dataset split.
                for batch_num, batch in enumerate(self._test_dataloader):
                    self._evaluate_batch('test', batch_num + 1, 0, batch, training=False,
                                         evaluate_only=False)

                self._has_computed_test = True

                # Save again, so as to include the evaluation on the test set.
                self.save(only_metrics_and_state=True)

            # Print the results of the model's performance on the test set.
            print_string = ""
            for i, metric in enumerate(self._valid_print_metrics):
                if (i > 0):
                    print_string += ", "
                print_string += self._metrics_manager.metrics['test'][metric].print_info(val=True, avg=False,
                                                                                         step=0, evaluate_only=False)
            print('\rTest set results (of the model that performed the best on the validation set):')
            print(print_string)





    def run_epoch_train(self, epoch_num, evaluate_only=False):
        """
        This method performs the training procedure for an entire epoch.


        :param epoch_num    : The number of the current epoch.

        :param evaluate_only: Identifies whether this is a singular evaluation, decoupled from the training procedure,
                              or not.


        :return: Nothing
        """

        # Cover all dataset splits being evaluated
        for data_loader_type in self._training_dataloders_types:
            # We define an extra variable that allows us to dynamically change the batch size when necessary.
            self._current_dataloader_type = data_loader_type

            # Determines whether the evaluation should proceed or not. The train split is always evaluated, while the
            # validation split is only evaluated every n_epoch, when self._check_val_every_n_epoch is verified.
            if (data_loader_type == "train" or epoch_num % self._check_val_every_n_epoch == 0):
                # Activate the correct evaluation mode (Disables batch_norm, dropout and such things in eval() mode).
                training = True if data_loader_type == "train" and epoch_num > 0 else False
                self._model.train() if training else self._model.eval()

                # Perform evaluation for each batch in the dataset split.
                for batch_num, batch in enumerate(self._training_dataloders[data_loader_type]):
                    self._evaluate_batch(data_loader_type, batch_num + 1, epoch_num, batch, training=training,
                                         evaluate_only=evaluate_only)

                    # Compute the effective batch number.
                    eff_batch_num = int(math.ceil(batch_num/self._accumulate_n_batch_grads))
                    # Print batch evaluation information regarding the train split, when training. This will actually
                    # print as being at the stage of batch_num level, not the accumulated_batch_num (i.e. it will print
                    # the "EPOCH%" info relative the actual mini-batch size being evaluated, not the accumulated one.).
                    if (training):
                        if ((eff_batch_num % self._print_every_n_batch == 0 and
                             (batch_num + 1) % self._accumulate_n_batch_grads == 0) or
                                      (batch_num + 1) == self._max_num_batches['train']):
                            self._metrics_manager.print_info(epoch_num, batch_num + 1, batch=True)

        # Determine if the validation set was evaluated.
        new_val = True if (epoch_num % self._check_val_every_n_epoch == 0 and bool(self._val_dataloader)) else False
        # Print the information regarding this (finalized) epoch.
        self._metrics_manager.print_info(epoch_num, self._max_num_batches['train'], new_val=new_val,
                                         evaluate_only=evaluate_only)

        # If this was not a singular evaluation, decoupled from the training procedure, increase the current_epoch_num.
        if (not evaluate_only):
            self._current_epoch_num += 1

            # # For debugging gradient flow. Comment when not necessary.
            # print("EPOCH:", epoch_num)
            # for name, param in self._model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data.sum())




    def _try_to_push(self, element):
        """
        This method attempts to push a single element towards the correct computation device. This way we avoid
        having to differentiate between which elements can be pushed and which cannot.
        """

        try:
            return element.to(device=self._device)
        except AttributeError:
            return element




    def _push_to_device(self, batch_ele):
        """
        This method allows easily pushing complex input/label formats into the correct computation device.
        """

        # TODO: This is not ideal as if we get a tuple or a list of ints, for example, we will apply the _try_to_push
        # TODO: method to all of it's elements. The idea was to only have to check tuples, but when the dataloader's
        # TODO: 'pin_memory=True' tuples get converted to lists.
        if ((isinstance(batch_ele, tuple) and batch_ele.__class__.__name__ != 'PackedSequence') or
                isinstance(batch_ele, list)):
            return tuple(self._push_to_device(element) for element in batch_ele)
        else:
            return self._try_to_push(batch_ele)




    def _evaluate_batch(self, data_loader_type, batch_num, epoch_num, batch, training=True, evaluate_only=False):
        """
        This method evaluates a singular batch.


        :param data_loader_type: Identifies the dataset split from where the batch came from. Used to select the
                                 appropriate metrics' instances, in the MetricsManager.

        :param batch_num       : The number of the minibatch, within the epoch.

        :param epoch_num       : The number of the current epoch.

        :param batch           : The actual batch, containing the data instances.

        :param training        : Identifies if this batch belongs to the training set and this is not a singular
                                 evaluation, decoupled from the training procedure.

        :param evaluate_only   : Identifies whether this is a singular evaluation, decoupled from the training procedure,
                                 or not.


        :return: Nothing
        """

        # Get the inputs and labels.
        inputs, labels = self._push_to_device(batch)


        # Perform the forward pass. 'data_loader_type' is provided as some models might behave differently
        # depending on the data split.
        predictions = self._model(inputs, data_loader_type)

        # Compute loss and metrics.
        self._metrics_manager.evaluate(data_loader_type, batch_num, epoch_num, predictions, labels, training,
                                       evaluate_only=evaluate_only)


        if (training):
            # Backpropagate, if training.
            self._metrics_manager.backpropagate_loss(batch_num)

            # # For debugging gradient flow. Comment when not necessary.
            # print("BATCH:", batch_num)
            # for name, param in self._model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad.data.sum())

            # We only want to perform a parameter update once we have accumulated the correct number of mini-batches.
            if (batch_num % self._accumulate_n_batch_grads == 0 or batch_num == self._max_num_batches):
                # Perform a parameters' update step.
                self._optimiser.step()
                # Zero the parameters' gradients.
                self._optimiser.zero_grad()




    def _load_var(self, var, var_name, mem_settings, load_path=None, load_epoch_or_best=None):
        """
        This method is used to determine the value of certain variables of the MEM, when it is instantiated, as some of
        them can be loaded from memory. However, it is possible that only part of a saved state is meant to be loaded
        from memory. If any of the input variables, that instantiated the MEM, is not None, then that value overrules
        the saved state.


        :param var               : The input variable to the MEM __init__() method.

        :param var_name          : A string that identifies which variable is being loaded/set.

        :param mem_settings      : The state dictionary, if any, of a previously instantiated MEM.

        :param load_path         : The path that leads to the directory where the states pertaining a previously
                                   trained model, MEM, etc are located.

        :param load_epoch_or_best: Determines whether to load a specific checkpoint or whether to load the epoch where
                                   the model performed the best.


        :return: Either the initial input variable value, or, if the initial input value was 'None', a value that was
                 stored in memory.
        """

        # A dictionary containing the default input values used when instantiating a MEM object.
        default_values = {
            'metrics_names'                   : [],
            'convergence_metrics_and_criteria': {'loss': ["train", 0.01, 4]},
            'print_metrics'                   : 'All',
            'print_every_n_batch'             : 1,
            'check_val_every_n_epoch'         : 1,
            'collate_fn'                      : default_collate,
            'max_num_epochs'                  : 50,
            'batch_size'                      : 128,
            'batch_size_eval'                 : -1,
            'num_workers'                     : 0,
            'loss_function_name'              : 'cross_entropy',
            'optimiser'                       : 'adam',
            'weight_decay'                    : 0.,
            'accumulate_n_batch_grads'        : 1,
            'save_path'                       : _paths.trained_models_path,
            'save_best_metric'                : 'loss',
            'keep_last_n_chekpoints'          : -1,
            'has_computed_test'               : False,
            'has_converged'                   : False,
        }

        # TODO: A lot of the check in this code will be handled by the argparse.
        # If the var is not None, that is, if the user specified a specific value, the MEM variable will take it.
        if (var is not None):
            # This makes sure that the specified maximum number of epochs for which to train the model is actually
            # valid, considering that the model might have already been previously trained.
            if (var_name == 'max_num_epochs'):
                if (isinstance(var, int)):
                    # Check that the specified max_number_epochs is not smaller than 0.
                    if (var < 0):
                        raise ValueError(
                            "Variable max_num_epochs needs to be an integer bigger or equal to 0 (0 is only valid " +
                            "for pre-evaluation). Current value = " + str(var))
                    # Check if specified max_number of epochs is not lower than what the model has already been trained
                    # for.
                    elif (mem_settings is not None and var_name in mem_settings and
                          var < mem_settings['max_num_epochs']):
                        raise ValueError("Variable max_num_epochs is lower than what the model has already been " +
                                         "trained for. Selected value = " + str(var) + ", while model has been " +
                                         "trained for " + str(mem_settings['max_num_epochs']) + " epochs.")

                # If max_num_epochs is not an integer at all, raise an error.
                else:
                    raise ValueError("Variable max_num_epochs needs to be an integer bigger or equal to 0 (0 is only " +
                                     "valid for pre-evaluation). Current value = " + str(var))
            loaded_var = var

        # If the variable did NOT have a specified value when the MEM was instantiated, attempt to load it from memory.
        else:
            # If the variable is stored in memory, load it.
            if (mem_settings is not None and var_name in mem_settings):
                # The model has its own loading function.
                if (var_name == 'model'):
                    type_of_model = _alt_names.MODELS[mem_settings['model']]
                    path_to_model_data = load_path + "Model/"
                    loaded_var = _classes.MODELS[type_of_model](load_path=path_to_model_data,
                                                                epoch_or_best=load_epoch_or_best,
                                                                **self._special_model_params)
                else:
                    loaded_var = mem_settings[var_name]

            # If the variable is NOT stored in memory, set it to the default value.
            else:
                loaded_var = default_values[var_name]

        # If the variable is the optimiser.
        if (var_name == 'optimiser'):
            optimiser_name = _alt_names.OPTIMISERS[loaded_var]
            # TODO: Allow optimiser parameters configuration.
            loaded_var = _classes.OPTIMS[optimiser_name](self._model.parameters(), lr=self._learning_rate,
                                                         weight_decay=self._weight_decay)

            # If a mem_state was loaded from memory, restore the optimiser state_dict.
            # TODO: OPTIMISER SHOULD ALWAYS STAY THE SAME, OR THEN ITS STATE DICT SHOULD NOT BE RESTORED.
            if (self._mem_state is not None):
                loaded_var.load_state_dict(self._mem_state['optimiser'])

        if (var_name == 'metrics_names'):
            # Make sure the input metrics names are all strings
            if (isinstance(loaded_var, list) and all(isinstance(item, str) for item in loaded_var)):
                loaded_var = ['Loss'] + loaded_var # The Loss metric always exists
            else:
                raise ValueError( "metrics_names should be a list of metrics names (strings) or an empty list. " +
                                  "Received: " + str(loaded_var) + ".")

            # Metrics that were evaluated in the past (associated with a saved state) are always reloaded.
            if (self._mem_state is not None):
                metrics_that_will_be_reloaded = [metric for metric in self._mem_state['valid_metrics_names'] if
                                                 metric not in [_alt_names.METRICS[new_metric] for new_metric in
                                                                loaded_var if new_metric in _alt_names.METRICS]]
                if (bool(metrics_that_will_be_reloaded)):
                    warning_string = "The following metrics will be reloaded from memory since they had already been "
                    warning_string += "used in previous epochs:"
                    for i, metric in enumerate(metrics_that_will_be_reloaded):
                        warning_string += " " + metric + ("," if i != len(metrics_that_will_be_reloaded) - 1 else ".\n")
                    print(warning_string)

                loaded_var += metrics_that_will_be_reloaded

        return loaded_var




    def save(self, only_metrics_and_state=False):
        """
        This method saves the state of the ModelEvaluationModule, and its dependants.


        :return: Nothing
        """

        # Define the correct path where this model's (and associated components) will be saved.
        dir_path = self._save_path
        model_filepath = dir_path + "Model" + os.sep

        # Make sure the directory exists
        directory = os.path.dirname(dir_path)
        if (not os.path.exists(directory)):
            os.makedirs(directory)

        # Make sure the 'Model' Directory exists
        directory = os.path.dirname(dir_path + "Model" + os.sep)
        if (not os.path.exists(directory)):
            os.makedirs(directory)

        # Make sure the 'MEM' Directory exists
        directory = os.path.dirname(dir_path + "MEM" + os.sep)
        if (not os.path.exists(directory)):
            os.makedirs(directory)

        # Create the MEM settings dict
        if (self._current_epoch_num == 1):
            mem_settings_filename = dir_path + "MEM" + os.sep + "settings.stg"
            mem_settings = {
                'model'                           : self._model.__class__.__name__,
                'metrics_names'                   : self._metrics_names,
                'convergence_metrics_and_criteria': self._convergence_metrics_and_criteria,
                'print_metrics'                   : self._print_metrics,
                'print_every_n_batch'             : self._print_every_n_batch,
                'check_val_every_n_epoch'         : self._check_val_every_n_epoch,
                'collate_fn'                      : self._collate_fn,
                'max_num_epochs'                  : self._max_num_epochs,
                'batch_size'                      : self._batch_size,
                'num_workers'                     : self._num_workers,
                'loss_function_name'              : self._loss_function_name,
                'optimiser'                       : self._optimiser.__class__.__name__,
                'weight_decay'                    : self._weight_decay,
                'save_path'                       : self._save_path,
                'save_best_metric'                : self._save_best_metric,
            }

            # This saves the model architecture and the initial parameters
            torch.save(mem_settings, mem_settings_filename)
            self._model.save(self._current_epoch_num - 1, model_filepath,
                             "." + str(self._current_epoch_num - 1) + ".ckp",
                             save_parameters=self._keep_last_n_chekpoints != 0)

        # Save the current state
        else:
            # TODO: When saving the best state, if the best state achieved is epoch 0, then no .bst files will exist.
            # TODO: Correct the above!
            # Determine if the current epoch is the one where the model has performed the best, under the save_best
            # metric.
            best = self._metrics_manager.check_if_best(self._current_epoch_num - 1)

            # Creates the usual checkpoint file termination, but also the termination that identifies the best model,
            # if necessary.
            file_types = ['.' + str(self._current_epoch_num - 1) + save_type for save_type in
                          ((['.ckp'] if self._keep_last_n_chekpoints != 0 else []) + (['.bst'] if best else []))]

            for file_type in file_types:
                # Defines the path for the MEM's state
                mem_state_filename = dir_path + "MEM" + os.sep + "state" + file_type

                # Creates the MEM's state dict
                mem_state = {
                    'current_epoch_num'  : self._current_epoch_num,
                    'optimiser'          : self._optimiser.state_dict(),
                    'valid_metrics_names': self._valid_metrics_names,
                    'has_converged'      : self._has_converged,
                    'has_computed_test'  : self._has_computed_test,
                }


                # Saves the MEM's state
                torch.save(mem_state, mem_state_filename)
                # Tell the MetricsManager to save its (and its dependants) state(s).
                self._metrics_manager.save(dir_path, file_type)
                # Tells the model to save its current state.
                if (not only_metrics_and_state):
                    self._model.save(self._current_epoch_num - 1, model_filepath, file_type)

                # Remove files of previous best model
                if (file_type[-4:] == ".bst"):
                    remove_files(dir_path, '.bst', ('.' + str(self._current_epoch_num - 1) + '.bst',), recursive=True)

                # Remove previous checkpoints beyond 'keep_last_n_chekpoints'.
                if (file_type[-4:] == ".ckp" and self._keep_last_n_chekpoints > 0):
                    min_range = max(self._current_epoch_num - self._keep_last_n_chekpoints, 0)
                    max_range = self._current_epoch_num
                    exceptions = [str(epoch_num_to_keep) for epoch_num_to_keep in range(min_range, max_range)]
                    exceptions = tuple('.' + epoch_num_exception + '.ckp' for epoch_num_exception in exceptions)
                    remove_files(dir_path, '.ckp', exceptions, recursive=True)




    @property
    def model(self):
        return self._model

    @property
    def current_epoch_num(self):
        return self._current_epoch_num

    @property
    def max_num_epochs(self):
        return self._max_num_epochs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_num_batches(self):
        return self._max_num_batches

    @property
    def loss_function_name(self):
        return self._loss_function_name

    @property
    def optimiser(self):
        return self._optimiser

    @property
    def num_workers(self):
        return self._num_workers

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def val_dataloader(self):
        return self._val_dataloader

    @property
    def test_dataloader(self):
        return self._test_dataloader

    @property
    def training_dataloders_types(self):
        return self._training_dataloders_types

    @property
    def training_dataloders(self):
        return self._training_dataloders

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
    def save_best_metric(self):
        return self._save_best_metric

    @property
    def metrics_manager(self):
        return self._metrics_manager

    @property
    def print_every_n_batch(self):
        return self._print_every_n_batch

    @property
    def check_val_every_n_epoch(self):
        return self._check_val_every_n_epoch

    @property
    def save_path(self):
        return self._save_path

    @property
    def has_computed_test (self):
        return self._has_computed_test

    @property
    def has_converged (self):
        return self._has_converged

    @property
    def current_dataloader_type (self):
        return self._current_dataloader_type