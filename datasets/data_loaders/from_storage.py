"""
##################################################
##################################################
## This file contains functions designed to     ##
## load various datasets into memory.           ##
##                                              ##
## Currently loads:                             ##
##                                              ##
## - MNIST dataset                              ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import numpy as np
import struct
import os
import warnings
from pathlib import Path


# *** Import own functions. *** #

import datasets.data_paths as data_paths
# from   datasets.preprocessing.helpers_preprocessing import tokenize_string





####################
##### OUR DATA #####
####################

# # TODO: make it a proper data creator, with splits and so on. Make sure instances x are not present in disjoint splits.
# def load_our_data(task='normal', validation=True, entities=False, mini_test=True):
#
#     data_path = data_paths.UW_RE_UVA
#     if (mini_test):
#         data_path += 'mini_test/' + task + '/'
#
#     datasets = {}
#     datasets['train'] = {}
#     if (validation):
#         datasets['val'] = {}
#     datasets['test'] = {}
#
#     for data_split in datasets:
#         datasets[data_split]['data'] = []
#         datasets[data_split]['relations'] = []
#
#     for data_split in ['train'] + (['val'] if validation else []) + ['test']:
#         # Get the set of relations pertaining the data_split.
#         with open(data_path + data_split + '_relations.txt', "r", encoding='utf-8') as f:
#             for line in f:
#                 datasets[data_split]['relations'].append(line.rstrip("\r\n"))
#
#         # Get the data instances.
#         with open(data_path + data_split + '.txt', "r", encoding='utf-8') as f:
#             for line in f:
#                 split_line = line.rstrip("\r\n").split("\t")
#                 if (not entities):
#                     split_line = split_line[:2]
#
#                 processed_data_instance = []
#                 for num_ele, ele in split_line:
#                     # The sentence, for which the relation will be estimated, needs to be tokenized.
#                     if (num_ele == 1):
#                         processed_data_instance.append(tokenize_string(ele))
#                     else:
#                         processed_data_instance.append(ele)
#                 datasets[data_split]['data'].append(processed_data_instance)
#
#     return datasets





#################
##### MNIST #####
#################

def load_MNIST_data_numpy(vectorize=True, binarize=True, binarize_threshold=150,
                          new_val_set=False, validation_size=5000):
    """
    This method is used to load the MNIST data from file into memory. Additionally, it can binarize the data and create
    a validation split. When a validation split is created, the new train and validation splits are stored in files,
    for easy reloading of the same splits, which this method automatically looks for. If a new validation split is
    created, regardless of one already existing or not, the model will save the previous one as a backup.

    :param vectorize         : Determines whether to vectorize the MNIST images or not.

    :param binarize          : Determines whether the MNIST images are meant to be binarized or not.

    :param binarize_threshold: The pixel intensity that determines the threshold for a pixel to become black or white,
                               when binarizing the MNIST images.

    :param new_val_set       : Whether to create a new validation split or not.

    :param validation_size   : The size of the validation split.


    :return: Either (training & test) (instances & labels) or (training & validation & test) (instances & labels).
    """

    # These are the binary identifiers for the type of data being read. See http://yann.lecun.com/exdb/mnist/
    image_magic_number  = 2051
    vector_magic_number = 2049

    #Auxiliary variable TODO: what?
    load_train = True

    # The paths to the original data, in case a validation split is not required.
    original_train_images = "original/train-images.idx3-ubyte"
    original_train_labels = "original/train-labels.idx1-ubyte"

    # If a validation split is required.
    if (validation_size > 0):
        # Construct the paths to the train data accounting for the validation split with validation_size.
        train_images = "train-images-val-" + str(validation_size) + "-split.idx3-ubyte"
        train_labels = "train-labels-val-" + str(validation_size) + "-split.idx1-ubyte"

        # Construct the paths to the validation data with validation_size.
        val_images = "val-" + str(validation_size) + "-images.idx3-ubyte"
        val_labels = "val-" + str(validation_size) + "-labels.idx1-ubyte"

        # The path to a file that stores which instances of the training data were selected for validation.
        val_original_indices = "val-" + str(validation_size) + "-original-indices.idx1-ubyte"

    # If a validation split is NOT required.
    else:
        # Simply have the paths point to the original training data (as created by Yann LeCun).
        train_images = original_train_images
        train_labels = original_train_labels

    # The paths to the test data (as created by Yann LeCun).
    test_images = "original/t10k-images.idx3-ubyte"
    test_labels = "original/t10k-labels.idx1-ubyte"



    def load_file(data_path):
        """
        Helper function used to actually read the data from file.


        :param data_path: The path to the data that is to be loaded.


        :return: The loaded data
        """

        with open(data_path, 'rb') as f:
            # Get information regarding the type and amount of data to be read.
            magic_number, num_images = struct.unpack(">II", f.read(8))

            # If reading images.
            if (magic_number == image_magic_number):
                num_rows, num_cols = struct.unpack(">II", f.read(8))
                data = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)

            # If reading labels.
            else:
                data = np.fromfile(f, dtype=np.uint8)

        return data



    def save_file(data_path, data, magic_number):
        """
        Helper function used to save the new splits to file.


        :param data_path   :  The path to where the data is meant to be saved.

        :param data        : The data to be saved.

        :param magic_number: The binary number that identifies what kind of data is being saved.


        :return: Nothing
        """

        with open(data_path, 'wb') as f:
            
            # Write Magic Number
            f.write(struct.pack(">I", magic_number)) # ">" --> High Endian
            
            # Write Number of Images/Labels
            f.write(struct.pack(">I", data.shape[0]))
            
            if (magic_number == image_magic_number):
                # Write Number of Rows
                f.write(struct.pack(">I", data.shape[1]))
                
                # Write Number of Columns
                f.write(struct.pack(">I", data.shape[2]))
                
            data.tofile(f)



    def check_all_files_exist():
        """
        Helper function that determines if all files related to an existing split exist or not.


        :return: A boolean indicating whether all files related to an existing split exist or not.
        """

        if(Path(data_paths.MNIST + train_images).exists() and
           Path(data_paths.MNIST + train_labels).exists() and
           Path(data_paths.MNIST + val_images).exists() and
           Path(data_paths.MNIST + val_labels).exists() and
           Path(data_paths.MNIST + val_original_indices).exists()):
            return True
        return False



    def create_backup(original, backup):
        """
        Helper function that creates backup files of the previous train/validation splits.


        :param original: The current path/file_name of the previous split.

        :param backup  : The new path/file_name of the previous split.


        :return: Nothing
        """

        try:
            os.replace(original, backup)
        except WindowsError:
            os.remove(backup)
            os.replace(original, backup)




    if (validation_size > 0):
        
        all_val_files_exist = check_all_files_exist()

        # If some files regarding a previously saved split are missing, or if a new validation split is meant to be
        # created, regardless.
        if (new_val_set or not all_val_files_exist):
            if (all_val_files_exist):
                # Creates backups of the previous splits.
                create_backup(data_paths.MNIST + train_images, data_paths.MNIST + "old-" + train_images)
                create_backup(data_paths.MNIST + train_labels, data_paths.MNIST + "old-" + train_labels)
                create_backup(data_paths.MNIST + val_images, data_paths.MNIST + "old-" + val_images)
                create_backup(data_paths.MNIST + val_labels, data_paths.MNIST + "old-" + val_labels)
                create_backup(data_paths.MNIST + val_original_indices, data_paths.MNIST + "old-" + val_original_indices)


            # Loads the training instances and labels.
            x_train = load_file(data_paths.MNIST + original_train_images)
            y_train = load_file(data_paths.MNIST + original_train_labels)

            # TODO: Error if val_size > original_train_size
            # Selects the indices of the training instances that will be moved to the validation split.
            val_indices = np.random.choice(np.arange(x_train.shape[0]), size=validation_size, replace=False)
            mask = np.ones(x_train.shape[0], dtype=bool)
            mask[val_indices] = False

            # Gets the instances concerning the validation split.
            x_val = x_train[~mask]
            y_val = y_train[~mask]

            # Gets the remaining instances, concerning the training split.
            x_train = x_train[mask]
            y_train = y_train[mask]

            
            # Save new train set to file
            save_file(data_paths.MNIST + train_images, x_train, image_magic_number)
            save_file(data_paths.MNIST + train_labels, y_train, vector_magic_number)
            
            # Save new val set to file
            save_file(data_paths.MNIST + val_images, x_val, image_magic_number)
            save_file(data_paths.MNIST + val_labels, y_val, vector_magic_number)
            save_file(data_paths.MNIST + val_original_indices, val_indices, vector_magic_number)

            warnings.warn("New validation set has been created. Take care when comparing models?")

            # Stops the training set from being loaded from file.
            load_train = False

        # All the files concerning a previous split exist and there is no intention to create a new validation split,
        # regardless.
        else:
            x_val = load_file(data_paths.MNIST + val_images)
            y_val = load_file(data_paths.MNIST + val_labels)
            
            
    # Loads the train set from file, if necessary. Either no validation split was asked for, or there was no need to
    # create a new split and all the files concerning a previous split existed.
    if (load_train):
        x_train = load_file(data_paths.MNIST + train_images)
        y_train = load_file(data_paths.MNIST + train_labels)

    # Loads the test instances and labels.
    x_test = load_file(data_paths.MNIST + test_images)
    y_test = load_file(data_paths.MNIST + test_labels)
    

    # If necessary, binarizes the input images.
    if (binarize):
        x_train = (x_train > binarize_threshold).astype(x_train.dtype)
        if (validation_size > 0):
            x_val = (x_val > binarize_threshold).astype(x_val.dtype)
        x_test = (x_test > binarize_threshold).astype(x_test.dtype)

    # If necessary, vectorizes the input images.
    if (vectorize):
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        if (validation_size > 0):
            x_val = x_val.reshape((x_val.shape[0], x_val.shape[1] * x_val.shape[2]))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    # Status message concerning the loaded data.
    load_message =  "Loaded MNIST data. Info: (shape, data_type)\n\n"
    load_message += "x_train: (" + str(x_train.shape) + ", " + str(x_train.dtype) + ") || "
    load_message += "y_train: (" + str(y_train.shape) + ", " + str(y_train.dtype) + ")\n"
    if (validation_size > 0):
        load_message += "x_val: (" + str(x_val.shape) + ", " + str(x_val.dtype) + ") || "
        load_message += "y_val: (" + str(y_val.shape) + ", " + str(y_val.dtype) + ")\n"
    load_message += "x_test: (" + str(x_test.shape) + ", " + str(x_test.dtype) + ") || "
    load_message += "y_test: (" + str(y_test.shape) + ", " + str(y_test.dtype) + ")\n\n\n"

    print(load_message)


    if (validation_size > 0):
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test