"""
##################################################
##################################################
## This file contains the various datasets      ##
## paths.                                       ##
##                                              ##
## Each variable will have a small description. ##
##################################################
##################################################
"""



###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

# Import here!


# *** Import own functions. *** #

from helpers.general_helpers import join_path





######################
##### DATA PATHS #####
######################

DATASETS = join_path(["datasets", "Data"])



# *** MNIST *** #
MNIST = join_path([DATASETS, "MNIST"]) # Raw MNIST data



# *** Levy *** #
LEVY               = join_path([DATASETS, "Levy"])
LEVY_FULL_POSITIVE = join_path([LEVY, "positive_examples"])



# *** Ours *** #
UW_RE_UVA                        = join_path([DATASETS, "Ours"])
UW_RE_UVA_RELATIONS_NAMES        = join_path([UW_RE_UVA, "relation_names.txt"])
UW_RE_UVA_RELATIONS_DESCRIPTIONS = join_path([UW_RE_UVA, "relations_descriptions"])
UW_RE_UVA_PROPOSED_SPLITS        = join_path([UW_RE_UVA, "proposed_splits"])
UW_RE_UVA_PROPOSED_SPLITS_NEW    = join_path([UW_RE_UVA, "proposed_splits_new"])

# print(DATASETS)
# print(MNIST)
# print(LEVY)
# print(LEVY_FULL_POSITIVE)
# print(OURS)
# print(RELATIONS_NAMES)
# print(RELATIONS_DESCRIPTIONS)
# print(PROPOSED_SPLITS)
#
#
# path_levy     = path_datasets + "Levy/"
# path_ours     = path_datasets + "Ours/"
#
# levy_full_dataset_path = path_levy + 'positive_examples'
#
# path_relation_names = path_ours + "relation_names.txt"
# path_relation_descriptions = path_ours + "relations_descriptions/"# + each relation's name
# path_proposed_splits = path_ours + "proposed_splits/"
# path_proposed_splits_copy = path_ours + "proposed_splits - Copy/"
# path_proposed_splits_original = path_proposed_splits + "original/"
# path_proposed_splits_original_copy = path_proposed_splits_copy + "original/"
# UW_RE_UVA                 = join_path(["datasets", "Data", "Ours"])  # Our data
# UW_RE_UVA_proposed_splits = join_path([UW_RE_UVA, "proposed_splits"])  # Our proposed splits