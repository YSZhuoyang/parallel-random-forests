
import os
import ctypes

# Note that os.path.abspath only recognize
# the path of current working directory
# the interface 'invoke_rf' is called in
# controller file which is under its parent dir.
PATH = os.path.abspath("RandomForestsSenAnalysis/libRF.so")
RANDOM_FORESTS = ctypes.CDLL(PATH)
RANDOM_FORESTS.Train.argtypes = [ctypes.c_uint, ctypes.c_uint]
RANDOM_FORESTS.Test.argtypes = []
RANDOM_FORESTS.Test.restype = ctypes.c_float

def invoke_rf_train(num_trees, num_features_per_tree):
    """Call random forest training method."""
    RANDOM_FORESTS.Train(
        ctypes.c_uint(num_trees),
        ctypes.c_uint(num_features_per_tree))

def invoke_rf_test():
    """Call random forest test methon."""
    accuracy = RANDOM_FORESTS.Test()
    return accuracy

