
import os
import ctypes

# Note that os.path.abspath only recognize
# the path of current working directory
# the interface 'invoke_rf' is called in
# controller file which is under its parent dir.
PATH = os.path.abspath("RandomForestsSenAnalysis/libRF.so")
RANDOM_FORESTS = ctypes.CDLL(PATH)
# Train model and generate model file
RANDOM_FORESTS.Train.argtypes = [ctypes.c_uint, ctypes.c_uint]
# Test model
RANDOM_FORESTS.Test.argtypes = []
RANDOM_FORESTS.Test.restype = ctypes.c_float
# Tokenize a sentence and do classification
RANDOM_FORESTS.Analyze.argtypes = [ctypes.c_char_p]
RANDOM_FORESTS.Analyze.restype = ctypes.c_void_p
# Free memory used by classification result
RANDOM_FORESTS.FreeMemo.argtypes = [ctypes.c_void_p]

def invoke_rf_train(num_trees, num_features_per_tree):
    """Call random forest training method."""
    RANDOM_FORESTS.Train(
        ctypes.c_uint(num_trees),
        ctypes.c_uint(num_features_per_tree))

def invoke_rf_test():
    """Call random forest test methon."""
    accuracy = RANDOM_FORESTS.Test()
    return accuracy

def invoke_rf_analyze(sentence):
    """Call random forest analyze methon."""
    label = RANDOM_FORESTS.Analyze(sentence)
    label_str = ctypes.cast(label, ctypes.c_char_p).value
    RANDOM_FORESTS.FreeMemo(label)
    return label_str

