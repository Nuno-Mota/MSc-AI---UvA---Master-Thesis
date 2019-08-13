"""
##################################################
##################################################
## This file contains several dicts that allow  ##
## for an easy mapping between common naming    ##
## alternatives.                                ##
##################################################
##################################################
"""

# *** Losses Names *** #
LOSSES = {
    'Cross Entropy' : 'Cross Entropy',
    'Cross_Entropy' : 'Cross Entropy',
    'CE'            : 'Cross Entropy',
    'cross entropy' : 'Cross Entropy',
    'cross_entropy' : 'Cross Entropy',
    'ce'            : 'Cross Entropy',

    'Bag of Words Log Loss': 'Bag of Words Log Loss',
    'bag of words log loss': 'Bag of Words Log Loss',
    'BagofWordsLogLoss'    : 'Bag of Words Log Loss',
    'bagofwordslogloss'    : 'Bag of Words Log Loss',
    'bowll'                : 'Bag of Words Log Loss',
    'BOWLL'                : 'Bag of Words Log Loss',
    'bow_ll'               : 'Bag of Words Log Loss',
    'Bow_ll'               : 'Bag of Words Log Loss',
    'BOW_LL'               : 'Bag of Words Log Loss',
    'BagOfWords_ll'        : 'Bag of Words Log Loss',
    'bagofwords_ll'        : 'Bag of Words Log Loss',
    'BagOfWords_LL'        : 'Bag of Words Log Loss',
    'bagofwords_LL'        : 'Bag of Words Log Loss',

    're_bow_loss': 're_bow_loss',
    're-bow-loss': 're_bow_loss',
    'RE_BOW_LOSS': 're_bow_loss',
    'RE-BOW-LOSS': 're_bow_loss',

    'mse': 'MSE',
    'MSE': 'MSE',
}



# *** Activation functions Names *** #
AFS = {
    'Relu'      : 'relu',
    'relu'      : 'relu',
    'lrelu'     : 'leaky_relu',
    'Lrelu'     : 'leaky_relu',
    'lRelu'     : 'leaky_relu',
    'leaky_relu': 'leaky_relu',
    'Leaky_Relu': 'leaky_relu',
    'Leaky_relu': 'leaky_relu',
    'leaky_Relu': 'leaky_relu',
    'leakyRelu' : 'leaky_relu',
    'LeakyRelu' : 'leaky_relu',
    'LeakyReLU' : 'leaky_relu',
    'Sigmoid'   : 'sigmoid',
    'sigmoid'   : 'sigmoid',
    'Tanh'      : 'tanh',
    'tanh'      : 'tanh',
    'Softmax'   : 'softmax',
    'softmax'   : 'softmax',
}


# *** Metrics Names *** #
METRICS = {
    'Loss' : 'Loss',
    'loss' : 'Loss',
    'L'    : 'Loss',

    # Actual Metrics
    'Accuracy'          : 'Accuracy',
    'Acc'               : 'Accuracy',
    'accuracy'          : 'Accuracy',
    'acc'               : 'Accuracy',
    'F1'                : 'F1',
    'f1'                : 'F1',
    'MacroF1'           : 'MacroF1',
    'macroF1'           : 'MacroF1',
    'Macro_F1'          : 'MacroF1',
    'macro_F1'          : 'MacroF1',
    'Macro_f1'          : 'MacroF1',
    'macro_f1'          : 'MacroF1',
    'MF1'               : 'MacroF1',
    'HarmonicMacroF1'   : 'HarmonicMacroF1',
    'harmonicMacroF1'   : 'HarmonicMacroF1',
    'harmonic_macroF1'  : 'HarmonicMacroF1',
    'harmonic_macro_f1' : 'HarmonicMacroF1',
    'Harmonic_macro_f1' : 'HarmonicMacroF1',
    'harmonic_Macro_f1' : 'HarmonicMacroF1',
    'harmonic_macro_F1' : 'HarmonicMacroF1',
    'Harmonic_Macro_f1' : 'HarmonicMacroF1',
    'Harmonic_macro_F1' : 'HarmonicMacroF1',
    'harmonic_Macro_F1' : 'HarmonicMacroF1',
    'Harmonic_Macro_F1' : 'HarmonicMacroF1',
    'HMMF1'             : 'HarmonicMacroF1',
    'HMF1'              : 'HarmonicMacroF1',
}



# *** Models Names *** #
MODELS = {
    'MLP'                 : 'MLP',
    'mlp'                 : 'MLP',
    'esim_sts'            : 'ESIM_StS',
    'esim-sts'            : 'ESIM_StS',
    'ESIM_StS'            : 'ESIM_StS',
    'ESIM-StS'            : 'ESIM_StS',
    'ESIM_STS'            : 'ESIM_StS',
    'ESIM-STS'            : 'ESIM_StS',
    'baseline'            : 'Baseline_BiLSTM',
    'baseline_bilstm'     : 'Baseline_BiLSTM',
    'baseline_biLSTM'     : 'Baseline_BiLSTM',
    'baseline_BiLSTM'     : 'Baseline_BiLSTM',
    'Baseline_biLSTM'     : 'Baseline_BiLSTM',
    'Baseline_BiLSTM'     : 'Baseline_BiLSTM',
    're_bow_decoder'      : 'RE_BoW_DECODER',
    're-bow-decoder'      : 'RE_BoW_DECODER',
    'RE_BoW_DECODER'      : 'RE_BoW_DECODER',
    'RE-BoW-DECODER'      : 'RE_BoW_DECODER',
    'RE_BOW_DECODER'      : 'RE_BoW_DECODER',
    'RE-BOW-DECODER'      : 'RE_BoW_DECODER',
    're_bow'              : 'RE_BoW',
    're-bow'              : 'RE_BoW',
    'RE_BoW'              : 'RE_BoW',
    'RE-BoW'              : 'RE_BoW',
    'RE_BOW'              : 'RE_BoW',
    'RE-BOW'              : 'RE_BoW',
    'embedding_layer'     : 'ESIM_Embed_Layer_AE',
    'esim_embedding_layer': 'ESIM_Embed_Layer_AE',
    'ESIM_Embed_Layer_AE' : 'ESIM_Embed_Layer_AE',
}



# *** Optimisers Names *** #
OPTIMISERS = {
    'Adam': 'adam',
    'adam': 'adam',
    'SGD' : 'sgd',
    'sgd' : 'sgd',
}



# *** Datasets Names *** #
DATASETS = {
    'mnist'                              : 'MNIST',
    'MNIST'                              : 'MNIST',
    'uw-re-uva'                          : 'UW-RE-UVA',
    'uw_re_uva'                          : 'UW-RE-UVA',
    'UW-RE-UVA'                          : 'UW-RE-UVA',
    'UW_RE_UVA'                          : 'UW-RE-UVA',
    'uw-re-uva-decoder-pre_train'        : 'UW-RE-UVA-DECODER-PRE-TRAIN',
    'uw_re_uva_decoder_pre_train'        : 'UW-RE-UVA-DECODER-PRE-TRAIN',
    'UW-RE-UVA-DECODER-PRE-TRAIN'        : 'UW-RE-UVA-DECODER-PRE-TRAIN',
    'UW_RE_UVA_DECODER_PRE_TRAIN'        : 'UW-RE-UVA-DECODER-PRE-TRAIN',
    'uw-re-uva-embedding-layer-pre-train': 'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN',
    'uw_re_uva_embedding_layer_pre_train': 'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN',
    'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN': 'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN',
    'UW_RE_UVA_EMBEDDING_LAYER_PRE_TRAIN': 'UW-RE-UVA-EMBEDDING-LAYER-PRE-TRAIN',
}



# *** Experiment Types Names *** #
SETTING_TYPES = {
    'DEBUG'                : {'N', 'ZS-O', 'ZS-C', 'GZS-O', 'GZS-C', 'FS-1', 'GFS-1'},
    'hyperparameter_tuning': {'N', 'ZS-O', 'ZS-C', 'GZS-O', 'GZS-C', 'FS-1', 'GFS-1', 'FS-2', 'GFS-2', 'FS-5', 'GFS-5', 'FS-10', 'GFS-10'},
    'final_evaluation'     : {'N', 'ZS-O', 'ZS-C', 'GZS-O', 'GZS-C', 'FS-1', 'GFS-1', 'FS-2', 'GFS-2', 'FS-5', 'GFS-5', 'FS-10', 'GFS-10'},
}

FOLD_NUMS = {
    'DEBUG'                : 2,
    'hyperparameter_tuning': 5,
    'final_evaluation'     : 10,
}



# *** Datasets Names *** #
REMOTE_SERVERS = {
    'lisa': 'Lisa',
    'Lisa': 'Lisa',
    'LISA': 'Lisa',
}



# *** ALL alternative Names *** #
ALL_ALTS = {**LOSSES, **METRICS, **MODELS, **OPTIMISERS, **DATASETS, **SETTING_TYPES, **FOLD_NUMS, **REMOTE_SERVERS}



