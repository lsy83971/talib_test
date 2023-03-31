# cython: language_level=3
cimport cython
cimport numpy as np
import numpy as np

from libcpp.vector cimport vector
from libcpp.deque cimport deque

from libcpp cimport bool

#python build.py build_ext --inplace
ctypedef np.uint8_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef jump(np.ndarray[DTYPE_t, ndim=1] cond, int span):
    '''
    jump span when cond satisfied
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



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict_sum(np.ndarray[dict, ndim=1] detail):
    '''
    '''
    cdef vector[np.float32_t] l1, l2
    cdef np.float32_t s1, s2
    for i in detail:
        s1 = 0; 
        s2 = 0; 
        for j1, j2 in i.items():
            s1 += j2
            s2 += j1 * j2
        l1.push_back(s1)
        l2.push_back(s2)
    return np.asarray(l1), np.asarray(l2)


