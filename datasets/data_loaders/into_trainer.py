"""
##################################################
##################################################
## This file contains functions designed to     ##
## feed data into the models, under the Pytorch ##
## Deep Learning Framework.                     ##
##                                              ##
## Currently loads:                             ##
##                                              ##
## - MNIST                                      ##
## - UW_RE_UVA                                  ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch
from   torch.utils.data import Dataset
from   torch.utils.data import DataLoader
from   torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import pickle
import h5py
import random


# *** Import own functions. *** #

import datasets.data_paths as _data_paths
from   datasets.data_loaders import from_storage
from   helpers.general_helpers import get_path_components, join_path, print_warning





############################
##### DATASET SELECTOR #####
############################

def select_dataset(dataset_name, dataset_arguments):

    if (dataset_name == 'UW-RE-UVA'):
        return get_UW_RE_UVA_datasets(**dataset_arguments)
    elif (dataset_name == 'UW-RE-UVA-DECODER-PRE-TRAIN'):
        return get_UW_RE_UVA_DECODER_PRE_TRAIN_dataset(**dataset_arguments)
    elif (dataset_name == 'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN'):
        return get_UW_RE_UVA_EMBEDDING_LAYER_PRE_TRAIN_dataset(**dataset_arguments)
    elif (dataset_name == 'MNIST'):
        return get_MNIST_datasets(**dataset_arguments)
    else:
        # This should never happen when running main directly.
        raise ValueError("The specified dataset meant to be loaded (" + str(dataset_name) + ") is unknown.")





####################
##### OUR DATA #####
####################

class UW_RE_UVA(Dataset):
    """
    The UW_RE_UVA class extends the Pytorch dataset class, and allows for easily feeding UW_RE_UVA data (both the
    observations and the relations' descriptions)(along with the necessary labels) into a model.
    """

    def __init__(self, file_name, setting, batch_size_xs, labels_to_load, num_workers_xs=0,
                 eval_all_descriptions=True, dataset_debug=None, full_prob_interp=False):
        """
        Instantiates a UW_RE_UVA dataset object for the specific split associated with 'file_name'.
        TODO: Missing variables.


        :param file_name            : The (almost complete) path to the data, in storage, pertaining this dataset.

        :param batch_size_xs        : The number of data instances (sentences x) (and respective labels) to load.

        :param labels_to_load       : A list that indicates what kind of labels are meant to be loaded by this dataset.

        :param num_workers_xs       : The number of sub-processes loading data instances from storage.

        :param eval_all_descriptions: Determines whether to evaluate against all relation descriptions or not.
        """

        # Input variables
        self._file_name             = file_name
        self._batch_size_xs         = batch_size_xs
        self._labels_to_load        = labels_to_load
        self._num_workers_xs        = num_workers_xs
        self._eval_all_descriptions = eval_all_descriptions


        # TODO: Always load the supervised labels, as they are necessary for evaluation in any of the splits.
        # TODO: Instead change the labels_to_load to refer uniquely to which unsupervised labels are meant to be loaded.

        # Instantiate the sub-dataset that will load the data instances and the corresponding labels.
        self._x_sentences_dataset = UW_RE_UVA_xs(file_name, setting, labels_to_load, dataset_debug, full_prob_interp)

        # Determine if this pertains a train split. If so we sample one relation description per relation in the split.
        self._train_split = self._x_sentences_dataset.train_split

        if (self._batch_size_xs == -1 or self._batch_size_xs > self._x_sentences_dataset.num_instances):
            if (self._batch_size_xs > self._x_sentences_dataset.num_instances):
                print_warning("Split: " + self._x_sentences_dataset.split + " | Requested Batch Size is bigger than " +
                              "available number of instances. Setting Batch Size to be equal to the available " +
                              "number of instances.")
            self._batch_size_xs = self._x_sentences_dataset.num_instances
        self._x_sentences_dataloader = DataLoader(self._x_sentences_dataset, batch_size=self._batch_size_xs,
                                                  shuffle=self._train_split, num_workers=num_workers_xs,
                                                  collate_fn=PadCollate(self._x_sentences_dataset.valid_labels))
        self._x_sentences_dataloader_iter = iter(self._x_sentences_dataloader)

        # We create a generator that allows us to sample from the x_sentences_dataset indefinitely.
        def get_subdataset_batch(dl):
            while (True):
                for batch in dl:
                    yield batch
        self._infinite_x_sentences_dl = get_subdataset_batch(self._x_sentences_dataloader)

        # Determine the correct length of this meta-dataset
        num_full_x_batches = len(self._x_sentences_dataset) // self._batch_size_xs
        equal_size_last_x_batch = len(self._x_sentences_dataset) % self._batch_size_xs == 0
        self._len_this_meta_dataset = num_full_x_batches + (0 if equal_size_last_x_batch else 1)

        # Load the list of relations involved in this split (in the file there is one relation per line)
        self._relations_in_split = self._x_sentences_dataset.relations_in_split

        # Load the map between relations and the indices of relation_descriptions
        path_components = get_path_components(file_name)
        with open(join_path(path_components[:-6] + ["ELMO_rel_descs_idxs.map"]), 'rb') as f:
            self._relation_to_idxs_rel_description = pickle.load(f)

        # The map above conveys always the same information for a specific instance of a dataset split, so:
        self._descs_idxs = [self._relation_to_idxs_rel_description[rel] for rel in self._relations_in_split]

        # Remove duplicate indices (for relation descriptions that might describe more than one relation).
        self._descs_idxs_all = []
        for idxs_set in self._descs_idxs:
            for idx in idxs_set:
                if (idx not in self._descs_idxs_all):
                    self._descs_idxs_all.append(idx)

        # Indicates which indices are meant to be aggregated together, when comparing against multiple relation
        # descriptions for the same relation.
        self._aggregate = []
        for idxs_set in self._descs_idxs:
            map_to_output_idxs = []
            for idx in idxs_set:
                map_to_output_idxs.append(self._descs_idxs_all.index(idx))
            self._aggregate.append(torch.LongTensor(map_to_output_idxs))




    def normal_last_batch_sizes(self, batch_size=1):
        """TODO"""
        if (len(self._x_sentences_dataset)%self._batch_size_xs == 0):
            return (self._batch_size_xs, self._batch_size_xs)
        else:
            return (self._batch_size_xs, len(self._x_sentences_dataset)%self._batch_size_xs)




    def __len__(self):
        """
        This method allows easily determining how many data instances this UW_RE_UVA dataset object has. This number
        will be number of batches, given the specified batch size, that the sub-dataset UW_RE_UVA_xs has.


        :return: The number of data instances in this dataset object.
        """

        return self._len_this_meta_dataset




    def __getitem__(self, index):
        """
        This method allows the indexing of the UW_RE_UVA dataset object.


        :param index: This argument is useless in this instance, as this meta-dataset is just a wrapper in order to
                      retrieve a batch of training instances and sample relation descriptions to which the batch will be
                      compared with.


        :return: The batch loaded by UW_RE_UVA_xs and a corresponding set of relation descriptions against which the
                 data instances in the batch will be compared.
                 ((training_instances_batch, relation_descriptions_batch), (labels+))
        """

        # Get batch part concerning the observations, x sentences
        x = next(self._infinite_x_sentences_dl)

        # Get the part concerning the relations descriptions. TODO: This part is always the same (the line below)
        if (self._train_split or not self._eval_all_descriptions):
            sampled_rel_descs = [random.choice(idx_list) for idx_list in self._descs_idxs]
            aggregate = None
        else:
            sampled_rel_descs = self._descs_idxs_all
            aggregate = self._aggregate

        return (x[0], (sampled_rel_descs, aggregate)), x[1]




    def get_classes_info(self):
        """TODO"""

        return self._x_sentences_dataset.get_classes_info()




    @property
    def file_name(self):
        return self._file_name

    @property
    def batch_size_xs(self):
        return self._batch_size_xs

    @property
    def labels_to_load(self):
        return self._labels_to_load

    @property
    def num_workers_xs(self):
        return self._num_workers_xs

    @property
    def eval_all_descriptions(self):
        return self._eval_all_descriptions

    @property
    def train_split(self):
        return self._train_split

    @property
    def x_sentences_dataset(self):
        return self._x_sentences_dataset

    @property
    def x_sentences_dataloader(self):
        return self._x_sentences_dataloader

    @property
    def x_sentences_dataloader_iter(self):
        return self._x_sentences_dataloader_iter

    @property
    def len_this_meta_dataset(self):
        return self._len_this_meta_dataset

    @property
    def relations_in_split(self):
        return self._relations_in_split

    @property
    def relation_to_idxs_rel_description(self):
        return self._relation_to_idxs_rel_description



                                                ##################



class UW_RE_UVA_xs(Dataset):
    """
    The UW_RE_UVA_xs class extends the Pytorch dataset class, and allows for easily feeding UW_RE_UVA observations (read
    from a hdf5 pre-computed file), and corresponding labels, into a model.
    """

    def __init__(self, file_name, setting, labels_to_load, dataset_debug=None, full_prob_interp=False):
        """
        Instantiates a UW_RE_UVA_xs dataset object for the specific split associated with 'file_name'.


        :param file_name     : The (almost complete) path to the data, in storage, pertaining this dataset.

        :param labels_to_load: A list that indicates what kind of labels are meant to be loaded by this dataset.
        """

        self._file_name        = file_name
        self._labels_to_load   = labels_to_load
        self._full_prob_interp = full_prob_interp

        path_components = get_path_components(file_name)
        debug_file = "DEBUG_" if path_components[-4] == 'DEBUG' else ""
        self._sentences_hdf5_file = join_path(path_components[:-4] + [debug_file + "sentences_to_ELMo.hdf5"])
        self._setting = setting
        self._split = path_components[-1]


        # Catch undesired configurations from the start.
        if (self._setting != 'N' and self._setting[0] != 'G' and full_prob_interp):
            raise RuntimeError("A full probabilistic interpretation is desired, but the chosen setting (" +
                               self._setting + ') is incompatible with such specification. Compatible settings are: ' +
                               "Normal (N); Generalised Any-Shot settings (GZS-(O/C), GFS-1, GFS-2, GFS-5, GFS-10).")


        # print('\n\n\n')
        DEBUG_EXPERIMENT = dataset_debug
        # print('DEBUG_EXPERIMENT:', dataset_debug)
        if (DEBUG_EXPERIMENT is not None and self._split == 'test'):
            self._file_name = join_path(get_path_components(self._file_name)[:-1] + ['val'])
            # print('FILE NAME:', self._file_name)

        # Get the instances indices (on the main hdf5 file) and determine the number of instances in this split.
        with open(self._file_name + ".idxs", 'rb') as f:
            loaded_instances = pickle.load(f)
            if (self._split != 'train'):
                # print(self._split, ' ORIGINAL INSTANCE INDICES', loaded_instances)
                if (DEBUG_EXPERIMENT == 'instances'):
                    self._instances_indices = loaded_instances[(0 if self._split == 'val' else 1)::2]
                elif (DEBUG_EXPERIMENT == 'classes'):
                    seen_classes = set()
                    seen_instances = []
                    seen_labels = []
                    if (self._setting[0] == 'G'):
                        with open(join_path(path_components[:-1] + ["train_relations.txt"]), 'r',
                                  encoding='utf-8') as f:
                            for line in f:
                                seen_classes.add(line.strip())
                    with open(self._file_name + '.lbs', 'rb') as f:
                        temp_labels = pickle.load(f)
                    count_class_elements = {}
                    for inst_num, instance in enumerate(loaded_instances):
                        if (temp_labels[inst_num] not in seen_classes):
                            if (temp_labels[inst_num] not in count_class_elements):
                                count_class_elements[temp_labels[inst_num]] = [instance]
                            else:
                                count_class_elements[temp_labels[inst_num]].append(instance)
                        else:
                            seen_instances.append(instance)
                            seen_labels.append(temp_labels[inst_num])
                    sorted_by_class_num_insts = sorted(count_class_elements.items(), key=lambda kv: len(kv[1]), reverse=True)
                    classes_and_instances = sorted_by_class_num_insts[(1 if self._split == 'val' else 0)::2]

                    self._labels = [_class for _class_set, inst_nums in classes_and_instances
                                               for _class in [_class_set] * len(inst_nums)] + seen_labels
                    self._instances_indices = [inst_num for _, inst_nums in classes_and_instances
                                                            for inst_num in inst_nums] + seen_instances

                elif (DEBUG_EXPERIMENT is None):
                    self._instances_indices = loaded_instances
                else:
                    raise ValueError('Wrong DEBUG_EXPERIMENT value.')
            else:
                self._instances_indices = loaded_instances
            self._num_instances = len(self._instances_indices)

        self._valid_labels = []

        # Load the corresponding supervised labels (text based label, i.e. the actual relation name).
        if ('supervised_lbls' in labels_to_load):
            with open(self._file_name + '.lbs', 'rb') as f:
                loaded_labels = pickle.load(f)
                if (self._split != 'train'):
                    # print(self._split, ' ORIGINAL LABELS', loaded_labels)
                    if (DEBUG_EXPERIMENT == 'instances'):
                        self._labels = loaded_labels[(0 if self._split == 'val' else 1)::2]
                    elif (DEBUG_EXPERIMENT == 'classes'):
                        pass
                    elif (DEBUG_EXPERIMENT is None):
                        self._labels = loaded_labels
                    else:
                        raise ValueError('Wrong DEBUG_EXPERIMENT value.')
                else:
                    self._labels = loaded_labels
            self._valid_labels.append('supervised_lbls')

            # TODO: remove this when done debugging weird results on test set.
            set_labels = set(self._labels)

            # Load the list of classes involved in this split. For some settings (like Few-Shot settings or Generalised
            # Any-Shot settings) the classes of another split might be present in this specific split.
            self._relations_in_split = []
            self._relation_in_split_to_idx_map = {}
            self._is_seen_class_indicator = {}

            self._exclusive_split_relations = []
            with open(self._file_name + "_relations.txt", 'r', encoding='utf-8') as f:
                r_num = 0
                for line in f:
                # for r_num, line in enumerate(f):
                    relation = line.strip()
                    if (relation in set_labels):
                        self._exclusive_split_relations.append(relation)
                        self._relations_in_split.append(relation)
                        self._relation_in_split_to_idx_map[relation] = r_num
                        r_num += 1

            if (self._setting[0] == 'G' or self._setting[0] == 'F' or self._setting == 'ZS-C'):
                # If this is a Generalised setting and a validation or test split, get the train classes.
                if (self._setting[0] == 'G' and self._split != 'train'):
                    for relation in self._exclusive_split_relations:
                        self._is_seen_class_indicator[relation] = False

                    self._relations_from_another_split = []
                    with open(join_path(path_components[:-1] + ["train_relations.txt"]), 'r', encoding='utf-8') as f:
                        for r_num, line in enumerate(f):
                            relation = line.strip()
                            self._relations_from_another_split.append(relation)
                            self._relations_in_split.append(relation)
                            self._relation_in_split_to_idx_map[relation] = r_num + len(self._exclusive_split_relations)
                            self._is_seen_class_indicator[relation] = True

                # If this is a Few-Shot setting and a train split, get the validation and test classes.
                # Also, if this is a (Generalised) Zero-Shot Closed setting, get the validation and test classes.
                if ((self._setting[0] == 'F' or self._setting[1] == 'F' or
                     self._setting == 'ZS-C' or self._setting == 'GZS-C') and self._split == 'train'):
                    for relation in self._exclusive_split_relations:
                        self._is_seen_class_indicator[relation] = True

                    # TODO: This assumes there's a validation split, which might not be the case.
                    self._relations_from_another_split = []
                    for split in (['val', 'test'] if DEBUG_EXPERIMENT is None else ['val']):
                        with open(join_path(path_components[:-1] + [split + "_relations.txt"]), 'r', encoding='utf-8') as f:
                            for r_num, line in enumerate(f):
                                relation = line.strip()
                                self._relation_in_split_to_idx_map[relation] = len(self._relations_from_another_split) + len(self._exclusive_split_relations)
                                self._relations_from_another_split.append(relation)
                                self._relations_in_split.append(relation)
                                self._is_seen_class_indicator[relation] = False


        # If this pertains a train split, load the unsupervised sentence labels, if they have been requested.
        # TODO: maybe some of these checks can be done when reading labels_to_load with argparse.
        if ('u_sentence_lbls' in labels_to_load and self._split == 'train'):
            with open(self._file_name + '.ulbs', 'rb') as f:
                self._data_as_idxs = pickle.load(f)
            self._valid_labels.append('u_sentence_lbls')


        # print('\n')
        # print('SPLIT:', self._split)
        # # print('INSTANCES INDICES:', self._instances_indices)
        # # print('LABELS', self._labels)
        # print('EXCLUSIVE SPLIT RELATIONS', self._exclusive_split_relations)
        # if (((self._setting in ['ZS-C', 'GZS-C'] or self._setting.split('-')[0] in ['FS', 'GFS'])
        #      and self._split == 'train') or (self._setting[0] == 'G' and self._split != 'train')):
        #         print('RELATIONS FROM ANOTHER SPLIT', self._relations_from_another_split)
        # print('RELATION IN SPLIT TO IDX MAP:', self._relation_in_split_to_idx_map)




    def __len__(self):
        """
        This method allows easily determining how many data instances this UW_RE_UVA_xs dataset object has.


        :return: The number of data instances in this dataset object.
        """

        return self._num_instances




    def __getitem__(self, index):
        """
        This method allows the indexing of the UW_RE_UVA_xs dataset object.


        :param index: The index corresponding to the training instance (and respective label(s)) intended to be
                      returned.


        :return: The training instance (and respective label) at the index corresponding to the input parameter (param
                 index).
                 (training_instances_batch, (labels+))
        """

        # We open and close the hdf5 for each data instance because this allows Pytorch to parallelize access to the
        # data, by increasing the num_workers related to UW_RE_UVA_xs sub-dataset.
        x_sentences = h5py.File(self._sentences_hdf5_file, 'r')

        # Read instance from file.
        x = torch.from_numpy(x_sentences[str(self._instances_indices[index])][:])

        # Close the hdf5 file.
        x_sentences.close()

        # Concatenate the different ELMO embeddings' types.
        x = x.permute(1, 0, 2).reshape(x.shape[1], x.shape[0]*x.shape[2])


        # Return the data instance and also the requested labels necessary for this split.
        return x, tuple(self._get_label(label_type, index) for label_type in self._valid_labels)




    def _get_label(self, label_type, index):
        """
        Helper function that returns the label of label_type, for easy creation of the tuple that contains all
        labels.


        :param label_type: The type of label to be added to the tuple that contains all labels.

        :param index     : The index of the data instance (and respective labels) to be returned.


        :return: The label, corresponding to label_type, at 'index'.
        """

        if (label_type == 'supervised_lbls'):
            if (bool(self._is_seen_class_indicator)):
                return (self._relation_in_split_to_idx_map[self._labels[index]],
                        self._is_seen_class_indicator[self._labels[index]])
            else:
                return (self._relation_in_split_to_idx_map[self._labels[index]], )
        elif (label_type == 'u_sentence_lbls'):
            return (self._data_as_idxs[index], )




    def get_classes_info(self):
        """TODO"""

        try:
            if (self._setting in ['ZS-C', 'GZS-C'] and self._split == 'train'):
                relations_in_split_to_idx_map = {}
                for relation in self.relation_in_split_to_idx_map:
                    if (relation in self._exclusive_split_relations):
                        relations_in_split_to_idx_map[relation] = self.relation_in_split_to_idx_map[relation]
            else:
                relations_in_split_to_idx_map = self.relation_in_split_to_idx_map
        except:
            relations_in_split_to_idx_map = None

        try:
            is_seen_class_indicator = self.is_seen_class_indicator
        except:
            is_seen_class_indicator = None

        return relations_in_split_to_idx_map, is_seen_class_indicator




    @property
    def file_name(self):
        return self._file_name

    @property
    def labels_to_load(self):
        return self._labels_to_load

    @property
    def sentences_hdf5_file(self):
        return self._sentences_hdf5_file

    @property
    def setting(self):
        return self._setting

    @property
    def split(self):
        return self._split

    @property
    def instances_indices(self):
        return self._instances_indices

    @property
    def num_instances(self):
        return self._num_instances

    @property
    def valid_labels(self):
        return self._valid_labels

    @property
    def labels(self):
        if ('supervised_lbls' in self._valid_labels):
            return self._labels
        else:
            raise ValueError("Supervised Labels were not requested. As such, variable 'labels' was not loaded.")

    @property
    def relations_in_split(self):
        if ('supervised_lbls' in self._valid_labels):
            return self._relations_in_split
        else:
            raise ValueError("Supervised Labels were not requested. As such, variable 'relations_in_split' was not " +
                             "loaded.")

    @property
    def relation_in_split_to_idx_map(self):
        if ('supervised_lbls' in self._valid_labels):
            return self._relation_in_split_to_idx_map
        else:
            raise ValueError("Supervised Labels were not requested. As such, variable 'relation_in_split_to_idx_map' " +
                             "was not loaded.")

    @property
    def is_seen_class_indicator(self):
        if ('supervised_lbls' in self._valid_labels):
            return self._is_seen_class_indicator
        else:
            raise ValueError("Supervised Labels were not requested. As such, variable 'is_seen_class_indicator' " +
                             "was not loaded.")

    @property
    def exclusive_split_relations(self):
        if ('supervised_lbls' in self._valid_labels):
            return self._exclusive_split_relations
        else:
            raise ValueError("Supervised Labels were not requested. As such, variable 'exclusive_split_relations' " +
                             "was not loaded.")

    @property
    def relations_from_another_split(self):
        if ('supervised_lbls' in self._valid_labels):
            if (((self._setting in ['ZS-C', 'GZS-C'] or self._setting.split('-')[0] in ['FS', 'GFS'])
                and self._split == 'train') or (self._setting[0] == 'G' and self._split != 'train')):
                return self._relations_from_another_split
            else:
                raise ValueError("This split, for this setting, does not have classes from another split. As such, " +
                                 "variable 'relations_from_another_split' was not created.")
        else:
            raise ValueError("Supervised Labels were not requested. As such, variable 'exclusive_split_relations' " +
                             "was not loaded.")

    @property
    def train_split(self):
        return self._split == 'train'

    @property
    def data_as_idxs(self):
        if ('u_sentence_lbls' in self._valid_labels):
            return self._data_as_idxs
        else:
            raise ValueError("Unsupervised Sentence Labels were not requested. As such, variable 'data_as_idxs' " +
                             "was not loaded.")




# *** DATASET LOADING HELPERS *** #

class PadCollateMetaDataloader:
    """
    A variant of collate_fn that removes the first dimension of a batch, due to the meta-dataloader being always called
    with a batch size of one.
    """

    def __init__(self):
        """
        Instantiates a PadCollateMetaDataloader object.
        """




    def __call__(self, batch):
        """
        Function that allows calling PadCollateMetaDataloader directly and squeezes the first dimension of all elements
        in the batch, which is always 1 due to the meta-dataloader (always) having a batch_size of 1.


        :param batch: The batch for which the first dimension will be squeezed.


        :return: The batch, after squeezing the first dimension.
        """

        # TODO: This print is just a check for the error of the dataloader's 'pin_memory=True'. Can remove when fixed.
        # print("META-PADCOLLATE - batch[0]:", batch[0])
        return batch[0]



                                                ##################



class PadCollate:
    """
    A variant of collate_fn that pads according to the longest sequence in a batch and creates pack_padded_sequences.
    """

    def __init__(self, valid_labels):
        """
        Instantiates a PadCollate object.


        :param valid_labels: A list with the labels which are meant to be collated, besides the data instances.
        """

        self._valid_labels = valid_labels





    def _get_label(self, label_type):
        """
        Returns the label of label_type, for easy creation of the tuple that contains all labels.


        :param label_type: The type of label to be added to the tuple that contains all labels.


        :return: The label corresponding to label_type
        """

        if (label_type == 'supervised_lbls'):
            return self._supervised_lbls
        elif (label_type == 'u_sentence_lbls'):
            return self._u_labels




    def _pad_collate(self, batch):
        """
        This function collates the different training instances in this batch. It pads them to the length of the longest
        sequence (same with labels, if necessary) and packs the embeddings of the sentences x into a
        Pack_padded_sequence, in order to make use of Pytorch's fast RNN computation methods.


        :param batch: The batch to be collated.


        :return: The collated batch.
        """

        # Find the sequences' lengths
        seq_lengths = torch.LongTensor([x[0].shape[0] for x in batch])

        # Order sequences in order of descending length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        # Stack embeddings and order them, from longest sequence to shortest sequence
        xs = pad_sequence([x[0] for x in batch], batch_first=True)[perm_idx]

        # Collate the supervised labels, if required, and order them in the same way as the embeddings.
        if ('supervised_lbls' in self._valid_labels):
            tuple_idx = self._valid_labels.index('supervised_lbls')
            seen_unseen_setting = len(batch[0][1][tuple_idx]) == 2
            if (seen_unseen_setting):
                self._supervised_lbls = torch.LongTensor([x[1][tuple_idx][0] for x in batch])[perm_idx]
                self._is_seen_class_indicators = torch.LongTensor([x[1][tuple_idx][1] for x in batch])[perm_idx]
                self._supervised_lbls = (self._supervised_lbls, self._is_seen_class_indicators)
            else:
                self._supervised_lbls = (torch.LongTensor([x[1][tuple_idx][0] for x in batch])[perm_idx], )

        # If unsupervised labels are also provided, compute a tensor and pad them to length of the longest sequence.
        # Also order them, as above.
        # This assumes that all kinds of supervision (the distinct ys) are available for all instances in the batch
        if ('u_sentence_lbls' in self._valid_labels):
            tuple_idx = self._valid_labels.index('u_sentence_lbls')
            self._u_labels = (pad_sequence([torch.LongTensor(x[1][tuple_idx]) for x in batch],
                                           batch_first=True)[perm_idx], )

        # Packs the embeddings, in order to make use of Pytorch's RNN's fast methods
        packed_xs = pack_padded_sequence(xs, seq_lengths, batch_first=True)

        return packed_xs, tuple(self._get_label(label_type) for label_type in self._valid_labels)




    def __call__(self, batch):
        """
        Function that allows calling PadCollate directly and, in turn, calls the _pad_collate function.


        :param batch: The batch to be collated.


        :return: The batch, after having been collated.
        """

        return self._pad_collate(batch)




    @property
    def valid_labels(self):
        return self._valid_labels




def get_UW_RE_UVA_datasets(masking_type='sub_obj_masking', dataset_type='DEBUG', setting='N', fold='0', validation=True,
                           batch_size_xs=25, batch_size_xs_eval=10, num_workers_xs=0, eval_all_descriptions=True,
                           labels_to_load=['supervised_lbls'], dataset_debug=None, **dataset_arguments):

    path_setting = setting.split('-')[0] if setting in ['ZS-O', 'ZS-C', 'GZS-O', 'GZS-C'] else setting
    path = join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS, masking_type, dataset_type, path_setting, fold])

    train_dataset = UW_RE_UVA(join_path([path, "train"]), setting, batch_size_xs, labels_to_load, num_workers_xs,
                              eval_all_descriptions, dataset_debug)

    if (validation):
        val_dataset = UW_RE_UVA(join_path([path, "val"]), setting, batch_size_xs_eval, labels_to_load, num_workers_xs,
                                eval_all_descriptions, dataset_debug)
    else:
        val_dataset = None

    test_dataset = UW_RE_UVA(join_path([path, "test"]), setting, batch_size_xs_eval, labels_to_load, num_workers_xs,
                             eval_all_descriptions, dataset_debug)

    return train_dataset, val_dataset, test_dataset





##########################################
##### OUR DATA- DECODER PRE TRAINING #####
##########################################

class UW_RE_UVA_DECODER_PRE_TRAIN(Dataset):
    """Docstring.TODO"""

    def __init__(self, path_to_embeddings, masking_type='sub_obj_masking', dataset_type='DEBUG', setting='N', fold='0',
                 ulbs_type='basic'):
        """Docstring.TODO"""

        path = join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS, masking_type, dataset_type, setting, fold])

        with open(join_path([path, "rel_descs_" + ulbs_type + ".ulbs"]), 'rb') as f:
            self._idxs_of_rels_descs_in_train = pickle.load(f)
            self._ulbs_of_rels_descs_in_train = pickle.load(f)

        self._path_to_embeddings = path_to_embeddings





    def __len__(self):
        """
        This method allows easily determining how many data instances this UW_RE_UVA_DECODER_PRE_TRAIN dataset object
        has.

        :return: The number of data instances in this dataset object.
        """

        return len(self._idxs_of_rels_descs_in_train)




    def normal_last_batch_sizes(self, batch_size):
        """TODO"""
        if (len(self)%batch_size == 0):
            return (batch_size, batch_size)
        else:
            return (batch_size, len(self)%batch_size)




    def __getitem__(self, index):
        """
        This method allows the indexing of the UW_RE_UVA_DECODER_PRE_TRAIN dataset object.


        :param index: The index corresponding to the training instance (and respective label) intended to be returned.


        :return: The training instance (and respective label) at the index corresponding to the input parameter (param
        index).
        """

        # We open and close the hdf5 for each data instance because this allows Pytorch to parallelize access to the
        # data, by increasing the num_workers related to UW_RE_UVA_xs sub-dataset.
        rels_descs = h5py.File(self._path_to_embeddings, 'r')

        # Read instance from file.
        rel_desc = torch.from_numpy(rels_descs[str(self._idxs_of_rels_descs_in_train[index])][:])

        # Close the hdf5 file.
        rels_descs.close()

        # Concatenate the different ELMO embeddings' types.
        rel_desc = rel_desc.permute(1, 0, 2).reshape(rel_desc.shape[1], rel_desc.shape[0] * rel_desc.shape[2])

        return rel_desc, torch.LongTensor(self._ulbs_of_rels_descs_in_train[index])




    def get_classes_info(self):
        """TODO"""

        return None, None




# *** DATASET LOADING HELPERS *** #

class PadCollateDecoderPreTraining:
    """
    A variant of collate_fn that pads according to the longest sequence in a batch and creates pack_padded_sequences.
    """

    def __init__(self, sentence_encoding_type):
        """
        Instantiates a PadCollateDecoderPreTraining object."""

        self._sentence_encoding_type = sentence_encoding_type




    def _pad_collate(self, batch):
        """
        This function collates the different training instances in this batch. It pads them to the length of the longest
        sequence (same with labels, if necessary) and packs the embeddings of the sentences x into a
        Pack_padded_sequence, in order to make use of Pytorch's fast RNN computation methods.


        :param batch: The batch to be collated.


        :return: The collated batch.
        """

        if (self._sentence_encoding_type == 'bilstm'):
            # Find the sequences' lengths
            seq_lengths = torch.LongTensor([x[0].shape[0] for x in batch])

            # Order sequences in order of descending length
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

            # Stack embeddings and order them, from longest sequence to shortest sequence
            xs = pad_sequence([x[0] for x in batch], batch_first=True)[perm_idx]

            # Packs the embeddings, in order to make use of Pytorch's RNN's fast methods
            xs = pack_padded_sequence(xs, seq_lengths, batch_first=True)

            # For the unsupervised labels, compute a tensor and pad them to length of the longest sequence.
            # Also order them, as above.
            u_labels = pad_sequence([torch.LongTensor(x[1]) for x in batch], batch_first=True)[perm_idx]

        elif (self._sentence_encoding_type == 'avg'):
            # TODO: Test with keepdim=True
            xs = torch.cat([x.mean(dim=0).unsqueeze(0) for x in batch[0]])

        else:
            raise NotImplementedError

        return xs, u_labels




    def __call__(self, batch):
        """
        Function that allows calling PadCollate directly and, in turn, calls the _pad_collate function.


        :param batch: The batch to be collated.


        :return: The batch, after having been collated.
        """

        return self._pad_collate(batch)




def get_UW_RE_UVA_DECODER_PRE_TRAIN_dataset(path_to_embeddings, masking_type='sub_obj_masking', dataset_type='DEBUG',
                                            setting='N', fold='0', ulbs_type='basic', **dataset_arguments):
    """Docstring:TODO"""

    return UW_RE_UVA_DECODER_PRE_TRAIN(path_to_embeddings, masking_type, dataset_type, setting,
                                       fold, ulbs_type), None, None





##################################################
##### OUR DATA- EMBEDDING LAYER PRE TRAINING #####
##################################################

class UW_RE_UVA_EMBEDDING_LAYER_PRE_TRAIN(Dataset):
    """Docstring.TODO"""

    def __init__(self, masking_type='sub_obj_masking', dataset_type='DEBUG'):
        """Docstring.TODO"""

        super(UW_RE_UVA_EMBEDDING_LAYER_PRE_TRAIN, self).__init__()

        self._data_path = join_path([_data_paths.UW_RE_UVA_PROPOSED_SPLITS, masking_type,
                                     ("DEBUG_" if dataset_type == 'DEBUG' else "") + "sentences_to_ELMo.hdf5"])

        data = h5py.File(self._data_path, 'r')
        self._num_instances = len(data) - 1
        data.close()





    def __len__(self):
        """
        TODO
        This method allows easily determining how many data instances this UW_RE_UVA_DECODER_PRE_TRAIN dataset object
        has.

        :return: The number of data instances in this dataset object.
        """

        return self._num_instances




    def normal_last_batch_sizes(self, batch_size):
        """TODO"""
        if (len(self)%batch_size == 0):
            return (batch_size, batch_size)
        else:
            return (batch_size, len(self)%batch_size)




    def __getitem__(self, index):
        """
        TODO
        This method allows the indexing of the UW_RE_UVA_DECODER_PRE_TRAIN dataset object.


        :param index: The index corresponding to the training instance (and respective label) intended to be returned.


        :return: The training instance (and respective label) at the index corresponding to the input parameter (param
        index).
        """

        # We open and close the hdf5 for each data instance because this allows Pytorch to parallelize access to the
        # data, by increasing the num_workers related to UW_RE_UVA_xs sub-dataset.
        sentences = h5py.File(self._data_path, 'r')

        # Read instance from file.
        sentence = torch.from_numpy(sentences[str(index)][:])

        # Close the hdf5 file.
        sentences.close()

        # Concatenate the different ELMO embeddings' types.
        sentence = sentence.permute(1, 0, 2).reshape(sentence.shape[1], sentence.shape[0] * sentence.shape[2])

        return sentence




    def get_classes_info(self):
        """TODO"""

        return None, None




# *** DATASET LOADING HELPERS *** #

class PadCollateEmbeddingLayerPreTraining:
    """
    A variant of collate_fn that pads according to the longest sequence in a batch and creates pack_padded_sequences.
    """

    def __init__(self):
        """
        Instantiates a PadCollateDecoderPreTraining object."""




    def _pad_collate(self, batch):
        """
        This function collates the different training instances in this batch. It pads them to the length of the longest
        sequence (same with labels, if necessary) and packs the embeddings of the sentences x into a
        Pack_padded_sequence, in order to make use of Pytorch's fast RNN computation methods.


        :param batch: The batch to be collated.


        :return: The collated batch.
        """

        # Find the sequences' lengths
        seq_lengths = torch.LongTensor([x.shape[0] for x in batch])

        # Order sequences in order of descending length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        # Stack embeddings and order them, from longest sequence to shortest sequence
        xs = pad_sequence([x for x in batch], batch_first=True)[perm_idx]

        # Packs the embeddings, in order to make use of Pytorch's RNN's fast methods
        xs = pack_padded_sequence(xs, seq_lengths, batch_first=True)

        # Get the labels, which are just the embeddings of each word.
        labels = xs.data

        return xs, labels




    def __call__(self, batch):
        """
        Function that allows calling PadCollate directly and, in turn, calls the _pad_collate function.


        :param batch: The batch to be collated.


        :return: The batch, after having been collated.
        """

        return self._pad_collate(batch)




def get_UW_RE_UVA_EMBEDDING_LAYER_PRE_TRAIN_dataset(masking_type='sub_obj_masking', dataset_type='DEBUG',
                                                    **dataset_arguments):
    """Docstring:TODO"""

    return UW_RE_UVA_EMBEDDING_LAYER_PRE_TRAIN(masking_type=masking_type, dataset_type=dataset_type), None, None





#################
##### MNIST #####
#################

class MNIST(Dataset):
    """
    The MNIST class extends the Pytorch dataset class, and allows for easily feeding MNIST data into a model.
    """

    def __init__(self, x, y):
        """
        Instantiates a MNIST dataset object for a specific split.


        :param x: The input data (the digit images (can be vectorized)), concerning a specific split.

        :param y: The labels associated with the input data (param x).
        """

        self._x = torch.from_numpy(x).float()
        self._y = torch.from_numpy(y).long()




    def normal_last_batch_sizes(self, batch_size):
        """TODO"""
        if (len(self)%batch_size == 0):
            return (batch_size, batch_size)
        else:
            return (batch_size, len(self)%batch_size)




    def __len__(self):
        """
        This method allows to easily determine how many data instances the MNIST dataset object has.


        :return: The number of data instances in this dataset object.
        """

        return self._x.shape[0]




    def __getitem__(self, index):
        """
        This method allows the indexing of the MNIST dataset object.


        :param index: The index corresponding to the training instance (and respective label) intended to be returned.


        :return: The training instance (and respective label) at the index corresponding to the input parameter (param
        index).
        """

        return self._x[index], self._y[index]




    def get_classes_info(self):
        """TODO"""

        return None, None





def get_MNIST_datasets(vectorize=True, binarize=True, binarize_threshold=150, new_val_set=False, validation_size=5000):
    """
    This method instantiates the MNIST dataset objects corresponding to the desired datasplits.


    :param vectorize         : Determines whether to vectorize the MNIST images or not.

    :param binarize          : Determines whether the MNIST images are meant to be binarized or not.

    :param binarize_threshold: The pixel intensity that determines the threshold for a pixel to become black or white,
                               when binarizing the MNIST images.

    :param new_val_set       : Whether to create a new validation split or not.

    :param validation_size   : The size of the validation split.


    :return: Either (training & test) MNIST datasets or (training & validation & test) MNIST datasets.
    """

    # If there is meant to be a validation split, load all splits from memory.
    if (validation_size > 0):
        print("Using a validation set.")
        x_train, y_train, x_val, y_val, x_test, y_test = from_storage.load_MNIST_data_numpy(vectorize, binarize,
                                                                                            binarize_threshold,
                                                                                            new_val_set,
                                                                                            validation_size)

        # Create the Pytorch dataset for the validation split.
        val_dataset = MNIST(x_val, y_val)

    # If there is NOT meant to be a validation split, load all splits from memory.
    else:
        print("NOT using a validation set.")
        x_train, y_train, x_test, y_test = from_storage.load_MNIST_data_numpy(vectorize, binarize, binarize_threshold,
                                                                              new_val_set, validation_size)
        val_dataset = None

    # Create the Pytorch datasets for the train and test splits.
    train_dataset = MNIST(x_train, y_train)
    test_dataset  = MNIST(x_test, y_test)

    return train_dataset, val_dataset, test_dataset