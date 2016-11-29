
import os
import ctypes

# Note that os.path.abspath only recognize
# the path of current working directory
# the interface 'invoke_rf' is called in
# controller file which is under its parent dir.
#PATH = os.path.abspath("RandomForestsSenAnalysis/libRF.so")
#RANDOM_FORESTS = ctypes.CDLL(PATH)
#RANDOM_FORESTS.Train.argtypes = [ctypes.c_int]
#RANDOM_FORESTS.Test.argtypes = []

#def invoke_rf_train(num_trees):
#    """Call random forest training method."""
#    RANDOM_FORESTS.Train(ctypes.c_int(num_trees))

#def invoke_rf_test():
#    """Call random forest test methon."""
#    RANDOM_FORESTS.Test()

