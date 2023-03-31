# cython: language_level=3
cimport cython
cimport numpy as np
import numpy as np

ctypedef np.uint8_t DTYPE_t

cdef np.ndarray[DTYPE_t] a; 
