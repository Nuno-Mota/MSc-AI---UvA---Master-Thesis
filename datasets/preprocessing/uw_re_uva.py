"""
##################################################
##################################################
## This module contains functions that are used ##
## to create the UW_RE_UVA datasets and splits. ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import os
import time
from   datetime import timedelta
import sys
# The commented lines below are needed for the code to run on Lisa.
# print(os.getcwd().split(os.sep))
# print(sys.path)
# if (sys.path[0].split(os.sep)[-1] == 'preprocessing'):
#     sys.path.append(sys.path[0] + os.sep + '..' + os.sep + '..' + os.sep)
# print(sys.path)
import math
import numpy as np
import random
import pickle
import hashlib
from   allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from   allennlp.predictors.predictor import Predictor
import allennlp.models.archival as archival
import re


# *** Own modules imports. *** #

from   helpers.general_helpers import pretty_dict_json_dump, pretty_dict_json_load, join_path, get_path_components, file_len
from   datasets.preprocessing.helpers_preprocessing import space_word_tokenize_string, NER_mask_sentence, softmax
import datasets.data_paths as _data_paths






#####################
##### FUNCTIONS #####
#####################

def create_all_capped_masked_data(max_sentence_length=60, cuda_device=None, batch_size=128,
                                  masking_types=['sub_obj_masking', 'NER_masking']):
    """
    This function masks all instances, according to the given masks, and separates them by class, just as
    the original ones.


    :param cuda_device  : The cuda device, if any, on which to run the NER tagger.

    :param masking_types: The different masking types that are to be produced.


    :return: Nothing. Everything is automatically saved to files.
    """

    if (get_path_components(os.getcwd())[-1] == 'Notebooks'):
        _data_paths.UW_RE_UVA                 = join_path(['..', _data_paths.UW_RE_UVA])
        _data_paths.UW_RE_UVA_RELATIONS_NAMES = join_path(['..', _data_paths.UW_RE_UVA_RELATIONS_NAMES])
    elif (get_path_components(os.getcwd())[-1] == 'preprocessing'):
        _data_paths.UW_RE_UVA                 = join_path(['..', '..', _data_paths.UW_RE_UVA])
        _data_paths.UW_RE_UVA_RELATIONS_NAMES = join_path(['..', '..', _data_paths.UW_RE_UVA_RELATIONS_NAMES])
    elif (get_path_components(os.getcwd())[-1] == 'Code'):
        pass
    else:
        raise RuntimeError('Unknown source directory.')


    # Load relation names from file.
    class_list = []
    with open(_data_paths.UW_RE_UVA_RELATIONS_NAMES, "r", encoding='utf-8') as f:
        for line in f:
            class_list.append(line.strip())



    # We get the NER tagger.
    if ('NER_masking' in masking_types):
        model_path = "https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz"
        if (cuda_device is not None):
            arch = archival.load_archive(model_path, cuda_device=cuda_device)
        else:
            arch = archival.load_archive(model_path)
        predictor = Predictor.from_archive(arch)

    # We get out tokenizer.
    tokenizer = WordTokenizer()



    # We define a start_time variable to help estimate how long the entire process will take.
    start_time = time.time()


    max_sentence_length_path = 'max_tokenized_sentence_len_' + str(max_sentence_length) + '_exclusive_and_unique_relation'
    path_dir_capped_exclusive = join_path([_data_paths.UW_RE_UVA, 'Levy_by_relation', max_sentence_length_path])

    for masking_type in masking_types:
        path = join_path([path_dir_capped_exclusive, masking_type]) + os.sep
        directory = os.path.dirname(path)
        if (not os.path.exists(directory)):
            os.makedirs(directory)


    # We get the total number of instances to be processed. Mostly for printing purposes and estimating remaining time.
    num_instances_per_relation = {}
    for relation in class_list:
        num_instances_per_relation[relation] = file_len(join_path([path_dir_capped_exclusive,
                                                                   re.sub('/', '-', re.sub(' ', '_', relation)) + '.txt']))
    total_num_instances = sum(num_instances_per_relation[relation] for relation in num_instances_per_relation)





    # We start the process of masking the sentences.
    instance_num = 0
    for r_num, relation in enumerate(class_list):
        relation_file_name = re.sub('/', '-', re.sub(' ', '_', relation)) + '.txt'
        path_relation_original = join_path([path_dir_capped_exclusive, relation_file_name])

        data = []

        # Load all the data pertaining this relation.
        with open(path_relation_original, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())


        if ('NER_masking' in masking_types):
            NER_data = [{'sentence': line.split('\t')[0]} for line in data]
            NER_results = []
            start = 0
            end = min(batch_size, len(NER_data))
            while (start != len(NER_data)):
                current_time = time.time()
                if (instance_num > 0):
                    eta = (total_num_instances - (instance_num - 1))*(current_time - start_time)/(instance_num - 1)
                print('\x1b[2K\r' + str(r_num + 1) + '/' + str(len(class_list)) + ' ' + relation + ' ||| ' +
                      'Creating NER masks: ' +  '{:7.3f}%'.format(100 * start / len(NER_data)) +
                      ' ||| Elapsed time: ' + str(timedelta(seconds=(current_time - start_time))) + '; ETA: ' +
                      (str(timedelta(seconds=eta)) if instance_num > 0 else 'Still Estimating...'), end="", flush=True)
                preds = predictor.predict_batch_json(NER_data[start:end])
                NER_results += [{'words': result['words'], 'tags': result['tags']} for result in preds]
                instance_num += end - start
                start = end
                end = min(end + batch_size, len(NER_data))


            current_time = time.time()
            if (instance_num > 0):
                eta = (total_num_instances - (instance_num - 1)) * (current_time - start_time) / (instance_num - 1)
            print('\x1b[2K\r' + str(r_num + 1) + '/' + str(len(class_list)) + ' ' + relation + ' ||| ' +
                  'Creating NER masks: ' + '{:7.3f}%'.format(100 * start / len(NER_data)) +
                  ' ||| Elapsed time: ' + str(timedelta(seconds=(current_time - start_time))) + '; ETA: ' +
                  (str(timedelta(seconds=eta)) if instance_num > 0 else 'Still Estimating...'), end="", flush=True)


        for masking_type in masking_types:
            with open(join_path([path_dir_capped_exclusive, masking_type, relation_file_name]),
                      'w', encoding='utf-8') as f:

                for l_num, line in enumerate(data):
                    if (l_num % 25 == 0 or l_num + 1 == num_instances_per_relation[relation]):
                        current_time = time.time()
                        if (instance_num > 0):
                            eta = (total_num_instances - (instance_num - 1)) * (current_time - start_time) / (
                                        instance_num - 1)
                        print('\x1b[2K\r' + str(r_num + 1) + '/' + str(len(class_list)) + ' ' + relation +
                              '(masking type:' + masking_type + ') ||| ' +
                              '{:7.3f}%'.format(100 * (l_num + 1) / num_instances_per_relation[relation]) + ' (' +
                              '{:7.3f}%'.format( 100 * (instance_num + 1) / total_num_instances) +
                              ') ||| Elapsed time: ' + str(timedelta(seconds=(current_time - start_time))),
                              end="", flush=True)

                    line_temp = line.split('\t')

                    if (masking_type == 'sub_obj_masking'):
                        # We assume the first entity is the subject entity and that the second one is the object entity.
                        # Any other entities are disregarded.
                        tokenized_sentence = space_word_tokenize_string(line_temp[0], tokenizer)
                        tokenized_subject_entity = space_word_tokenize_string(line_temp[1], tokenizer)
                        tokenized_object_entity = space_word_tokenize_string(line_temp[2], tokenizer)

                        masked_sentence = []
                        word_pos = 0
                        while (word_pos < len(tokenized_sentence)):
                            current_word = tokenized_sentence[word_pos]

                            match = False
                            if (current_word == tokenized_subject_entity[0] or
                                    current_word == tokenized_object_entity[0]):
                                # We assume that the subject entity is different from the object entity.
                                # First we start by testing the subject entity.
                                if (word_pos + len(tokenized_subject_entity) <= len(tokenized_sentence)):
                                    match = True
                                    for subj_word_num, subj_word in enumerate(tokenized_subject_entity):
                                        if (subj_word != tokenized_sentence[word_pos + subj_word_num]):
                                            match = False
                                            break
                                    if (match):
                                        masked_sentence.append('SUBJECT_ENTITY')
                                        word_pos += len(tokenized_subject_entity)

                                # Now we test the object entity.
                                if (not match and word_pos + len(tokenized_object_entity) <= len(tokenized_sentence)):
                                    match = True
                                    for obj_word_num, obj_word in enumerate(tokenized_object_entity):
                                        if (obj_word != tokenized_sentence[word_pos + obj_word_num]):
                                            match = False
                                            break
                                    if (match):
                                        masked_sentence.append('OBJECT_ENTITY')
                                        word_pos += len(tokenized_object_entity)

                            if (not match):
                                masked_sentence.append(tokenized_sentence[word_pos])
                                word_pos += 1

                        f.write(''.join(masked_sentence) + '\n')

                    ######################################################################
                    # Now we take care of the data for the 'NER_masking' experiment. #
                    ######################################################################
                    if (masking_type == 'NER_masking'):
                        ner_masked_sentence = NER_mask_sentence(line_temp[0], NER_results[l_num], tokenizer)
                        f.write(''.join(space_word_tokenize_string(ner_masked_sentence, tokenizer)) + '\n')





def create_splits_for_masking_types(cuda_device=None, batch_size=128,
                                    masking_types=['original', 'unmasked', 'sub_obj_masking', 'NER_masking']):
    """
    For a proposed splits' configuration for which a valid distribution of the data has already been found
    and a bulk dataset that has had its instances separated by class into different files, this function will
    create the actual proposed splits, for each of the specified masking types (the data instances will be the
    same for each masking type. Only the masking itself will change).


    :param cuda_device  : The cuda device, if any, on which to run the NER tagger.

    :param masking_types: The different masking types that are to be produced.


    :return: Nothing. Everything is automatically saved to files.
    """

    if (get_path_components(os.getcwd())[-1] =='Notebooks'):
        _data_paths.UW_RE_UVA                     = join_path(['..', _data_paths.UW_RE_UVA])
        _data_paths.UW_RE_UVA_RELATIONS_NAMES     = join_path(['..', _data_paths.UW_RE_UVA_RELATIONS_NAMES])
        _data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW = join_path(['..', _data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW])

    elif (get_path_components(os.getcwd())[-1] =='preprocessing'):
        _data_paths.UW_RE_UVA                     = join_path(['..', '..', _data_paths.UW_RE_UVA])
        _data_paths.UW_RE_UVA_RELATIONS_NAMES     = join_path(['..', '..', _data_paths.UW_RE_UVA_RELATIONS_NAMES])
        _data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW = join_path(['..', '..', _data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW])

    elif (get_path_components(os.getcwd())[-1] =='Code'):
        pass

    else:
        raise RuntimeError('Unknown source directory.')

    # Load the proposed splits data distribution.
    with open(join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW, 'original', 'proposed_splits_data_distribution.dat']),
                        'rb') as f:
        num_instances_check = pickle.load(f)
        classes_in_split = pickle.load(f)

    disjoint_HT = sum(np.sum(num_instances_check['HT'][setting], axis=(0, 1)) for setting in num_instances_check['HT'])


    with open(join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW, 'original', 'proposed_splits_meta_data.txt']),
              'r', encoding='utf-8') as f:
        proposed_splits_meta_data = pretty_dict_json_load(f)


    # Load relation names from file.
    class_list = []
    with open(_data_paths.UW_RE_UVA_RELATIONS_NAMES, "r", encoding='utf-8') as f:
        for line in f:
            class_list.append(line.strip())



    # We get the NER tagger.
    if ('NER_masking' in masking_types):
        model_path = "https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz"
        if (cuda_device is not None):
            arch = archival.load_archive(model_path, cuda_device=cuda_device)
        else:
            arch = archival.load_archive(model_path)
        predictor = Predictor.from_archive(arch)

    # We get out tokenizer.
    tokenizer = WordTokenizer()



    # We define a start_time variable to help estimate how long the entire process will take.
    start_time = time.time()

    # We define the fold names of each dataset_type.
    dataset_type_names = {'HT': 'hyperparameter_tuning', 'final': 'final_evaluation', 'DEBUG': 'DEBUG'}

    # We get the path to the data files separated by class.
    max_sentence_length = proposed_splits_meta_data['max_sentence_length']
    max_sentence_length_path = 'max_tokenized_sentence_len_' + str(max_sentence_length) + '_exclusive_and_unique_relation'
    path_dir_capped_exclusive = join_path([_data_paths.UW_RE_UVA, 'Levy_by_relation', max_sentence_length_path])


    # We open the destination files, start keeping track of labels and relations exclusive to each split.
    paths = {}
    labels = {}
    relations_exclusive_to_split = {}
    for masking_type in masking_types:
        paths[masking_type] = {}
        labels[masking_type] = {}
        relations_exclusive_to_split[masking_type] = {}

        path = join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW, masking_type]) + os.sep
        directory = os.path.dirname(path)
        if (not os.path.exists(directory)):
            os.makedirs(directory)

        for dataset_type in proposed_splits_meta_data['dataset_types']:
            paths[masking_type][dataset_type] = {}
            labels[masking_type][dataset_type] = {}
            relations_exclusive_to_split[masking_type][dataset_type] = {}

            path = join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW, masking_type,
                              dataset_type_names[dataset_type]]) + os.sep
            directory = os.path.dirname(path)
            if (not os.path.exists(directory)):
                os.makedirs(directory)

            for setting in proposed_splits_meta_data['settings'][dataset_type]:
                paths[masking_type][dataset_type][setting] = {}
                labels[masking_type][dataset_type][setting] = {}
                relations_exclusive_to_split[masking_type][dataset_type][setting] = {}

                path = join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW, masking_type,
                                  dataset_type_names[dataset_type], setting]) + os.sep
                directory = os.path.dirname(path)
                if (not os.path.exists(directory)):
                    os.makedirs(directory)

                for fold in range(proposed_splits_meta_data['num_folds'][dataset_type]):
                    paths[masking_type][dataset_type][setting][fold] = {}
                    labels[masking_type][dataset_type][setting][fold] = {}
                    relations_exclusive_to_split[masking_type][dataset_type][setting][fold] = {}

                    path = join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW, masking_type,
                                      dataset_type_names[dataset_type], setting, str(fold)]) + os.sep

                    directory = os.path.dirname(path)
                    if (not os.path.exists(directory)):
                        os.makedirs(directory)

                    for split in proposed_splits_meta_data['splits']:
                        path = join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS_NEW, masking_type,
                                      dataset_type_names[dataset_type], setting, str(fold), split + ".txt"])
                        paths[masking_type][dataset_type][setting][fold][split] = open(path, 'w', encoding='utf-8')
                        labels[masking_type][dataset_type][setting][fold][split] = []
                        relations_exclusive_to_split[masking_type][dataset_type][setting][fold][split] = []

    # We get the total number of instances to be processed. Mostly for printing purposes and estimating remaining time.
    num_instances_per_relation = {}
    for relation in class_list:
        num_instances_per_relation[relation] = file_len(join_path([path_dir_capped_exclusive,
                                                                   re.sub('/', '-', re.sub(' ', '_', relation)) + '.txt']))
    total_num_instances = sum(num_instances_per_relation[relation] for relation in num_instances_per_relation)


    # We start the process of creating the splits.
    instance_num = 0
    try:
        for r_num, relation in enumerate(class_list):
            for dataset_type in proposed_splits_meta_data['dataset_types']:
                for setting in proposed_splits_meta_data['settings'][dataset_type]:
                    for fold in range(proposed_splits_meta_data['num_folds'][dataset_type]):
                        for split in proposed_splits_meta_data['splits']:
                            for masking_type in masking_types:
                                n = num_instances_check[dataset_type][setting][fold][
                                    proposed_splits_meta_data['splits'][split]][
                                    proposed_splits_meta_data['class_to_idx'][relation]]
                                labels[masking_type][dataset_type][setting][fold][split] += [relation] * n
                                if (proposed_splits_meta_data['class_to_idx'][relation] in
                                        classes_in_split[dataset_type][setting][fold][split]):
                                    relations_exclusive_to_split[masking_type][dataset_type][setting][fold][split].append(
                                        relation)

            path_relation = join_path([path_dir_capped_exclusive,
                                       re.sub('/', '-', re.sub(' ', '_', relation)) + '.txt'])
            data = []
            data_to_write = {masking_type: [] for masking_type in masking_types}

            # Load all the data pertaining this relation.
            with open(path_relation, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(line)


            err_msg = "The number of instances is different from what was expected."
            n = proposed_splits_meta_data['idx_to_class_and_len'][proposed_splits_meta_data['class_to_idx'][relation]][1]
            assert len(data) == n, err_msg
            random.shuffle(data)


            if ('NER_masking' in masking_types):
                NER_data = [{'sentence': line.split('\t')[0]} for line in data]
                NER_results = []
                start = 0
                end = min(batch_size, len(NER_data))
                while (start != len(NER_data)):
                    current_time = time.time()
                    if (instance_num > 0):
                        eta = (total_num_instances - (instance_num - 1))*(current_time - start_time)/(instance_num - 1)
                    print('\x1b[2K\r' + str(r_num + 1) + '/' + str(len(class_list)) + ' ' + relation + ' ||| ' +
                          'Creating NER masks: ' +  '{:7.3f}%'.format(100 * start / len(NER_data)) +
                          ' ||| Elapsed time: ' + str(timedelta(seconds=(current_time - start_time))) + '; ETA: ' +
                          (str(timedelta(seconds=eta)) if instance_num > 0 else 'Still Estimating...'), end="", flush=True)
                    preds = predictor.predict_batch_json(NER_data[start:end])
                    NER_results += [{'words': result['words'], 'tags': result['tags']} for result in preds]
                    start = end
                    end = min(end + batch_size, len(NER_data))

                current_time = time.time()
                if (instance_num > 0):
                    eta = (total_num_instances - (instance_num - 1)) * (current_time - start_time) / (instance_num - 1)
                print('\x1b[2K\r' + str(r_num + 1) + '/' + str(len(class_list)) + ' ' + relation + ' ||| ' +
                      'Creating NER masks: ' + '{:7.3f}%'.format(100 * start / len(NER_data)) +
                      ' ||| Elapsed time: ' + str(timedelta(seconds=(current_time - start_time))) + '; ETA: ' +
                      (str(timedelta(seconds=eta)) if instance_num > 0 else 'Still Estimating...'), end="", flush=True)

            for l_num, line in enumerate(data):
                if (l_num % 25 == 0 or l_num + 1 == num_instances_per_relation[relation]):
                    current_time = time.time()
                    if (instance_num > 0):
                        eta = (total_num_instances - (instance_num - 1))*(current_time - start_time)/(instance_num - 1)
                    print('\x1b[2K\r' + str(r_num + 1) + '/' + str(len(class_list)) + ' ' + relation + ' ||| ' +
                          '{:7.3f}%'.format(100 * (l_num + 1) / num_instances_per_relation[relation]) + ' (' +
                          '{:7.3f}%'.format(100 * (instance_num + 1) / total_num_instances) + ') ||| Elapsed time: ' +
                          str(timedelta(seconds=(current_time - start_time))) + '; ETA: ' +
                          (str(timedelta(seconds=eta)) if instance_num > 0 else 'Still Estimating...'),
                          end="", flush=True)

                line_temp = line.strip()

                for masking_type in masking_types:
                    ##############################################
                    # We start by registering the original data. #
                    ##############################################
                    if (masking_type == 'original'):
                        data_to_write['original'].append(relation + '\t' + line_temp)
                        line_temp = line.split('\t')

                    ########################################################################################################
                    # We start by creating the data for the 'unmasked' experiment. We also take care of tokenization here. #
                    ########################################################################################################
                    if (masking_type == 'unmasked'):
                        data_to_write['unmasked'].append(''.join(space_word_tokenize_string(line_temp[0], tokenizer)))

                    ######################################################################
                    # Now we take care of the data for the 'sub_obj_masking' experiment. #
                    ######################################################################
                    if (masking_type == 'sub_obj_masking'):
                        # We assume the first entity is the subject entity and that the second one is the object entity.
                        # Any other entities are disregarded.
                        tokenized_sentence       = space_word_tokenize_string(line_temp[0], tokenizer)
                        tokenized_subject_entity = space_word_tokenize_string(line_temp[1], tokenizer)
                        tokenized_object_entity  = space_word_tokenize_string(line_temp[2], tokenizer)

                        masked_sentence = []
                        word_pos = 0
                        while (word_pos < len(tokenized_sentence)):
                            current_word = tokenized_sentence[word_pos]

                            match = False
                            if (current_word == tokenized_subject_entity[0] or
                                    current_word == tokenized_object_entity[0]):
                                # We assume that the subject entity is different from the object entity.
                                # First we start by testing the subject entity.
                                if (word_pos + len(tokenized_subject_entity) <= len(tokenized_sentence)):
                                    match = True
                                    for subj_word_num, subj_word in enumerate(tokenized_subject_entity):
                                        if (subj_word != tokenized_sentence[word_pos + subj_word_num]):
                                            match = False
                                            break
                                    if (match):
                                        masked_sentence.append('SUBJECT_ENTITY')
                                        word_pos += len(tokenized_subject_entity)

                                # Now we test the object entity.
                                if (not match and word_pos + len(tokenized_object_entity) <= len(tokenized_sentence)):
                                    match = True
                                    for obj_word_num, obj_word in enumerate(tokenized_object_entity):
                                        if (obj_word != tokenized_sentence[word_pos + obj_word_num]):
                                            match = False
                                            break
                                    if (match):
                                        masked_sentence.append('OBJECT_ENTITY')
                                        word_pos += len(tokenized_object_entity)

                            if (not match):
                                masked_sentence.append(tokenized_sentence[word_pos])
                                word_pos += 1

                        data_to_write['sub_obj_masking'].append(''.join(masked_sentence))

                    ######################################################################
                    # Now we take care of the data for the 'NER_masking' experiment. #
                    ######################################################################
                    if (masking_type == 'NER_masking'):
                        ner_masked_sentence = NER_mask_sentence(line_temp[0], NER_results[l_num], tokenizer)
                        data_to_write['NER_masking'].append(''.join(space_word_tokenize_string(ner_masked_sentence,
                                                                                               tokenizer)))

                instance_num += 1

            num_instances_disjoint_HT = disjoint_HT[proposed_splits_meta_data['class_to_idx'][relation]]
            HT_instances = {masking_type: data_to_write[masking_type][:num_instances_disjoint_HT]
                            for masking_type in data_to_write}
            non_HT_instances = {masking_type: data_to_write[masking_type][num_instances_disjoint_HT:]
                                for masking_type in data_to_write}

            # We start by writing out the 'DEBUG' and 'HT' datasets.
            for masking_type in masking_types:
                for dataset_type in proposed_splits_meta_data['dataset_types']:
                    if (dataset_type != 'final'):
                        current_slice_start = 0
                        for setting in proposed_splits_meta_data['settings'][dataset_type]:
                            for fold in range(proposed_splits_meta_data['num_folds'][dataset_type]):
                                for split in proposed_splits_meta_data['splits']:
                                    current_slice_end = current_slice_start
                                    current_slice_end += num_instances_check[dataset_type][setting][fold][
                                        proposed_splits_meta_data['splits'][split]][
                                        proposed_splits_meta_data['class_to_idx'][relation]]
                                    # DEBUG classes also belong to the seen classes.
                                    for line in HT_instances[masking_type][current_slice_start:current_slice_end]:
                                        paths[masking_type][dataset_type][setting][fold][split].write(line + '\n')
                                    current_slice_start += num_instances_check[dataset_type][setting][fold][
                                        proposed_splits_meta_data['splits'][split]][
                                        proposed_splits_meta_data['class_to_idx'][relation]]

            # Here we take care of the 'final' dataset
            fully_disjoint_from_HT_settings = {}
            disjoint_test_splits_but_no_disjointness_from_HT_settings = {}
            test_in_other_tests_settings = {}
            for setting in proposed_splits_meta_data['settings']['final']:
                if (setting not in fully_disjoint_from_HT_settings):
                    fully_disjoint_from_HT_settings[setting] = []
                    disjoint_test_splits_but_no_disjointness_from_HT_settings[setting] = []
                    test_in_other_tests_settings[setting] = []

                fold_sums = np.sum(num_instances_check['final'][setting], axis=0)
                num_instances_disjoint_train = fold_sums[0][proposed_splits_meta_data['class_to_idx'][relation]]
                num_instances_disjoint_val = fold_sums[1][proposed_splits_meta_data['class_to_idx'][relation]]
                num_instances_disjoint_test = fold_sums[2][proposed_splits_meta_data['class_to_idx'][relation]]
                num_max_instances_test = np.max(num_instances_check['final'][setting], axis=0)[2][
                    proposed_splits_meta_data['class_to_idx'][relation]]

                train_final_bigger_disjoint_HT = num_instances_disjoint_train > num_instances_disjoint_HT

                fully_disjoint = num_instances_disjoint_train + num_instances_disjoint_val + num_instances_disjoint_test < len(
                    non_HT_instances['original'])
                train_in_non_HT_instances = num_instances_disjoint_train - num_instances_disjoint_HT if train_final_bigger_disjoint_HT else 0
                disjoint_test = train_in_non_HT_instances + num_instances_disjoint_val + num_instances_disjoint_test < len(
                    non_HT_instances['original'])

                final_train_instances = {}
                final_val_instances = {}
                final_test_instances = {}

                if (fully_disjoint):
                    fully_disjoint_from_HT_settings[setting].append(relation)
                    for masking_type in masking_types:
                        final_train_instances[masking_type] = non_HT_instances[masking_type][:num_instances_disjoint_train]
                        final_val_instances[masking_type] = non_HT_instances[masking_type][
                                                            num_instances_disjoint_train:num_instances_disjoint_train + num_instances_disjoint_val]
                        final_test_instances[masking_type] = non_HT_instances[masking_type][-num_instances_disjoint_test:]
                else:
                    if (disjoint_test):
                        disjoint_test_splits_but_no_disjointness_from_HT_settings[setting].append(relation)
                    else:
                        test_in_other_tests_settings[setting].append(relation)

                    for masking_type in masking_types:
                        if (disjoint_test):
                            final_train_instances[masking_type] = HT_instances[masking_type] + non_HT_instances[
                                                                                                   masking_type][:-(
                                        num_instances_disjoint_test + num_instances_disjoint_val)]
                            final_val_instances[masking_type] = non_HT_instances[masking_type][-(
                                        num_instances_disjoint_test + num_instances_disjoint_val):-num_instances_disjoint_test]
                            final_test_instances[masking_type] = non_HT_instances[masking_type][
                                                                 -num_instances_disjoint_test:]
                        else:
                            # This way we can have a bigger pool of possible samples for the test splits.
                            final_train_instances[masking_type] = HT_instances[masking_type] + non_HT_instances[
                                                                                                   masking_type][
                                                                                               :train_in_non_HT_instances]
                            final_val_instances[masking_type] = non_HT_instances[masking_type][
                                                                train_in_non_HT_instances:train_in_non_HT_instances + num_instances_disjoint_val]
                            assert len(non_HT_instances[
                                           masking_type]) - train_in_non_HT_instances + num_instances_disjoint_val >= num_max_instances_test, 'TEST IN TEST ERROR'
                            final_test_instances[masking_type] = non_HT_instances[masking_type][
                                                                 train_in_non_HT_instances + num_instances_disjoint_val:]

                    # We shuffle the training instances to mix HT instances with potential non-HT instances.
                    zipped_train = list(zip(*tuple(final_train_instances[masking_type] for masking_type in masking_types)))
                    random.shuffle(zipped_train)
                    for masking_type, masking_type_train_data in list(
                            zip(final_train_instances.keys(), list(zip(*zipped_train)))):
                        final_train_instances[masking_type] = masking_type_train_data

                # Now we write out the this specific setting of the 'final' dataset.
                current_slice_start = {'train': 0, 'val': 0, 'test': 0}
                for fold in range(proposed_splits_meta_data['num_folds']['final']):
                    for split in proposed_splits_meta_data['splits']:
                        current_slice_end = current_slice_start[split]
                        current_slice_end += \
                        num_instances_check['final'][setting][fold][proposed_splits_meta_data['splits'][split]][
                            proposed_splits_meta_data['class_to_idx'][relation]]
                        current_data = {}
                        if (split == 'train'):
                            for masking_type in masking_types:
                                current_data[masking_type] = final_train_instances[masking_type]
                        elif (split == 'val'):
                            for masking_type in masking_types:
                                current_data[masking_type] = final_val_instances[masking_type]
                        else:
                            if (fully_disjoint or disjoint_test):
                                for masking_type in masking_types:
                                    current_data[masking_type] = final_test_instances[masking_type]
                            else:
                                zipped_test = list(zip(*tuple(final_test_instances[masking_type]
                                                              for masking_type in masking_types)))
                                random.shuffle(zipped_test)
                                for masking_type, masking_type_test_data in list(zip(final_test_instances.keys(),
                                                                                     list(zip(*zipped_test)))):
                                    final_test_instances[masking_type] = masking_type_test_data
                                for masking_type in masking_types:
                                    current_data[masking_type] = final_test_instances[masking_type]

                        for masking_type in masking_types:
                            for line in current_data[masking_type][current_slice_start[split]:current_slice_end]:
                                paths[masking_type]['final'][setting][fold][split].write(line + '\n')

                        if (split == 'test' and not fully_disjoint and not disjoint_test):
                            pass
                        else:
                            current_slice_start[split] += \
                            num_instances_check['final'][setting][fold][proposed_splits_meta_data['splits'][split]][
                                proposed_splits_meta_data['class_to_idx'][relation]]

    except:
        # Close file_pointers in case of error.
        for masking_type in masking_types:
            for dataset_type in proposed_splits_meta_data['dataset_types']:
                for setting in proposed_splits_meta_data['settings'][dataset_type]:
                    for fold in range(proposed_splits_meta_data['num_folds'][dataset_type]):
                        for split in proposed_splits_meta_data['splits']:
                            paths[masking_type][dataset_type][setting][fold][split].close()
        raise

    print('\r' + str(r_num + 1) + '/' + str(len(class_list)) + ' ' + relation + ' ||| ' +
          '{:7.3f}%'.format(100 * (l_num + 1) / num_instances_per_relation[relation]) + ' (' +
          '{:7.3f}%'.format(100 * (instance_num + 1) / total_num_instances) + ') ||| Elapsed time: ' +
          str(timedelta(seconds=(time.time() - start_time))), end="", flush=True)

    print('\n\n\nThe following settings are fully disjoint (also disjoint from the HT dataset):')
    for setting in fully_disjoint_from_HT_settings:
        if (all(relation in fully_disjoint_from_HT_settings[setting] for relation in class_list)):
            print(setting)

    print(
        '\n\nThe following settings have disjoint test splits (However, some of the train instances will also exist in the HT dataset):')
    for setting in disjoint_test_splits_but_no_disjointness_from_HT_settings:
        if (all(relation in fully_disjoint_from_HT_settings[setting] or relation in
                disjoint_test_splits_but_no_disjointness_from_HT_settings[setting] for relation in class_list) and
                any(relation not in fully_disjoint_from_HT_settings[setting] for relation in class_list)):
            print(setting)

    print(
        '\n\nThe following settings do NOT have disjoint test splits (and some of the train instances will also exist in the HT dataset):')
    for setting in test_in_other_tests_settings:
        if (any(relation not in fully_disjoint_from_HT_settings[setting] and relation not in
                disjoint_test_splits_but_no_disjointness_from_HT_settings[setting] for relation in class_list)):
            print(setting)

    # Close file_pointers.
    for masking_type in masking_types:
        for dataset_type in proposed_splits_meta_data['dataset_types']:
            for setting in proposed_splits_meta_data['settings'][dataset_type]:
                for fold in range(proposed_splits_meta_data['num_folds'][dataset_type]):
                    for split in proposed_splits_meta_data['splits']:
                        split_cls = relations_exclusive_to_split[masking_type][dataset_type][setting][fold][split]
                        with open(paths[masking_type][dataset_type][setting][fold][split].name[:-4] + '_relations.txt',
                                  'w', encoding='utf-8') as f:
                            for relation in split_cls:
                                f.write(relation + '\n')
                        if (masking_type != 'original'):
                            with open(paths[masking_type][dataset_type][setting][fold][split].name[:-4] + '.lbs',
                                      'wb') as f:
                                pickle.dump(labels[masking_type][dataset_type][setting][fold][split], f)
                        paths[masking_type][dataset_type][setting][fold][split].close()




# create_all_capped_masked_data(cuda_device= 0, batch_size=250)