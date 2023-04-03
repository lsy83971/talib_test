# cython: language_level=3
# python build.py build_ext --inplace

cimport cython
cimport numpy as npc
import numpy as np

from libcpp.vector cimport vector
from libcpp cimport bool
ctypedef npc.int8_t DTYPE_t
ctypedef npc.float32_t DTYPE_f

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef jump(npc.ndarray[DTYPE_t, ndim=1] cond, int span):
    '''
    jump span when cond satisfied
    '''
    cdef int i=0;
    cdef int length = len(cond);
    l = list()
    while (i<length):
        j=cond[i]
        if j!=0:
            l.append(i)
            i=i+span
        else:
            i=i+1
            continue
    return l

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict_sum(npc.ndarray[dict, ndim=1] detail):
    cdef vector[npc.float32_t] l1, l2
    cdef npc.float32_t s1, s2
    for i in detail:
        s1 = 0; 
        s2 = 0; 
        for j1, j2 in i.items():
            s1 += j2
            s2 += j1 * j2
        l1.push_back(s1)
        l2.push_back(s2)
    return np.asarray(l1), np.asarray(l2)

import datetime
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _EMA(npc.ndarray[DTYPE_f, ndim=1] arr, int rate):
    cdef npc.float32_t r1 = 1 / rate
    cdef npc.float32_t r2 = 1 - 1 / rate
    cdef int a = 3;
    cdef npc.ndarray[DTYPE_f, ndim=1] arr1 = np.empty(arr.shape[0], dtype=np.float32)
    arr1[0] = arr[0];
    for i in range(1, arr.shape[0]):
        arr1[i] = arr1[i - 1] * r2 + arr[i] * r1
    return arr1

@cython.boundscheck(False)
@cython.wraparound(False)
def EMA(arr, rate):
    arr = arr.astype(np.float32)
    arr.fillna(method="bfill", inplace=True)
    arr.fillna(method="ffill", inplace=True)
    return _EMA(arr.values, rate)
