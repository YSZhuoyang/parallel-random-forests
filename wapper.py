
import os
import ctypes

path = os.path.abspath("libRF.so")
randomForest = ctypes.CDLL(path)
randomForest.invoke.argtypes = [ctypes.c_int]

def our_function(numTrees):
    global randomForest
    array_type = ctypes.c_int * numTrees
    randomForest.invoke(ctypes.c_int(numTrees))
    
our_function(1)
