# cython: language_level=3
cimport cython
cimport numpy as np
import numpy as np

ctypedef np.uint8_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef jump(np.ndarray[DTYPE_t, ndim=1] cond, int span):
    '''
    Clip the values in a to be between min and max. Result in out
    '''
    cdef int i=0;
    cdef int length = len(cond);
    l = list()
    while (i<length):
        j=cond[i]
        if j==1:
            l.append(i)
            i=i+span
        else:
            i=i+1
            continue
    return l

