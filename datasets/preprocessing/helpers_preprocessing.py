"""
##################################################
##################################################
## This module contains helper functions used   ##
## when preprocessing datasets.                 ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import numpy as np


# *** Own modules imports. *** #

# Import here!





#####################
##### FUNCTIONS #####
#####################

def space_word_tokenize_string(s, tokenizer):
    """
    This method is used to tokenize (with spaces) a string, making use of an allenNLP's tokenizer's tokenize method.


    :param s        : The string to be space tokenized.

    :param tokenizer: The tokenizer that will be used to tokenize the sentence 's'.


    :return: The space tokenized string.
    """

    tokenized_string = [token.text for token in tokenizer.tokenize(s)]
    sentence = []
    for k, token in enumerate(tokenized_string):
        sentence += [token] + ([' '] if k + 1 < len(tokenized_string) else [])
    return sentence




def NER_space_word_tokenize_string(s, tokenizer):
    """
    This method is used to tokenize a string in a way that matches AllenNLP's NER predictor,
    making use of an allenNLP's tokenizer's tokenize method.


    :param s        : The string to be space tokenized.

    :param tokenizer: The tokenizer that will be used to tokenize the sentence 's'.


    :return: The space tokenized string.
    """

    split_s = s.split()
    sentence = []
    for k, split_s_ele in enumerate(split_s):
        sentence += [token.text for token in tokenizer.tokenize(split_s_ele)]
        if (k + 1 < len(split_s)):
            sentence += [' ']
    return sentence




def NER_mask_sentence(s, results, tokenizer):
    """
    This method masks all entities recognized by an AllenNLP's NER predictor.


    :param s        : The string to be space masked.

    :param predictor: The predictor that will be used to identify the entities present in 's'.

    :param tokenizer: The tokenizer that will be used to tokenize the sentence 's'.


    :return: The entity masked string.
    """

    #results = predictor.predict(sentence=s)

    NER_results = list(zip(results["words"], results["tags"]))
    masked_sentence = []
    NER_pos = 0
    in_entity = False
    for word_pos, word in enumerate(NER_space_word_tokenize_string(s, tokenizer)):
        if (word == NER_results[NER_pos][0]):
            if (NER_results[NER_pos][1] != 'O'):
                if (NER_results[NER_pos][1][0] == 'B' or NER_results[NER_pos][1][0] == 'I'):
                    if (NER_results[NER_pos][1][0] == 'B'):
                        in_entity = True
                else:
                    masked_sentence.append('ENTITY')
                if (NER_results[NER_pos][1][0] == 'L'):
                    in_entity = False
            else:
                if (not in_entity):
                    masked_sentence.append(word)
            NER_pos += 1
        else:
            if (not in_entity):
                masked_sentence.append(word)

    return ''.join(masked_sentence)




def softmax(x):
    """
    This method computes the softmax for a given array.


    :param x: The array to be softmaxed.


    :return: The softmaxed array.
    """

    return np.exp(x)/np.sum(np.exp(x))