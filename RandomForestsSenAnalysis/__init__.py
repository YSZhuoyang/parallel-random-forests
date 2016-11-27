
import os
import ctypes

# Note that os.path.abspath only recognize
# the path of current working directory
# the interface 'invoke_rf' is called in
# controller file which is under its parent dir.
PATH = os.path.abspath("RandomForestsSenAnalysis/libRF.so")
RANDOM_FORESTS = ctypes.CDLL(PATH)
RANDOM_FORESTS.trainAndTest.argtypes = [ctypes.c_int]

def invoke_rf(num_trees):
    """Call random forest component interface."""
    print PATH
    #RANDOM_FORESTS.trainAndTest(ctypes.c_int(num_trees))

