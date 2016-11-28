
import os
import ctypes

# Note that os.path.abspath only recognize
# the path of current working directory
# the interface 'invoke_rf' is called in
# controller file which is under its parent dir.
PATH = os.path.abspath("RandomForestsSenAnalysis/libRF.so")
RANDOM_FORESTS = ctypes.CDLL(PATH)
RANDOM_FORESTS.Train.argtypes = [ctypes.c_int]
RANDOM_FORESTS.Test.argtypes = []

def invoke_rf(num_trees):
    """Call random forest component interface."""
    RANDOM_FORESTS.Train(ctypes.c_int(num_trees))
    RANDOM_FORESTS.Test()

