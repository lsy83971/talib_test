# cython: language_level=3
# python build.py build_ext --inplace

cimport cython
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector

ctypedef np.int8_t DTYPE_t
ctypedef np.float64_t DTYPE_f

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef split_cumsum(np.ndarray[DTYPE_f, ndim=1] v,
                    np.ndarray[DTYPE_t, ndim=1] cond,
):
    '''
    jump span when cond satisfied
    '''
    cdef int i=0;
    cdef int length = len(cond);
    cdef DTYPE_f[:] res = np.zeros(length); 
    cdef DTYPE_f j1;
    cdef DTYPE_t j2;

    cdef DTYPE_f k1 = 0;
    for i in range(length):
        j1 = v[i]
        j2 = cond[i]

        if j2 == 1:
            k1 = 0

        k1 += j1
        res[i] = k1
    return np.asarray(res)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef split_cummax(np.ndarray[DTYPE_f, ndim=1] v,
                   np.ndarray[DTYPE_t, ndim=1] cond,
):
    '''
    jump span when cond satisfied
    '''
    cdef int i=0;
    cdef int length = len(cond);
    cdef DTYPE_f[:] res = np.zeros(length); 
    cdef DTYPE_f j1;
    cdef DTYPE_t j2;

    cdef DTYPE_f k1 = 0;
    for i in range(length):
        j1 = v[i]
        j2 = cond[i]

        if j2 == 1:
            k1 = j1
        else:
            k1 = max(k1, j1)
        res[i] = k1
    return np.asarray(res)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef split_cummin(np.ndarray[DTYPE_f, ndim=1] v,
                   np.ndarray[DTYPE_t, ndim=1] cond,
):
    '''
    jump span when cond satisfied
    '''
    cdef int i=0;
    cdef int length = len(cond);
    cdef DTYPE_f[:] res = np.zeros(length); 
    cdef DTYPE_f j1;
    cdef DTYPE_t j2;

    cdef DTYPE_f k1 = 0;
    for i in range(length):
        j1 = v[i]
        j2 = cond[i]

        if j2 == 1:
            k1 = j1
        else:
            k1 = min(k1, j1)
        res[i] = k1
    return np.asarray(res)

@cython.wraparound(False)
cpdef split_idx(np.ndarray[DTYPE_t, ndim=1] cond):
    '''
    jump span when cond satisfied
    '''
    cdef int i=0;
    cdef int length = len(cond);
    cdef DTYPE_f[:] to_front = np.zeros(length);
    cdef DTYPE_f[:] to_end = np.zeros(length);
    cdef DTYPE_f[:] front_idx = np.zeros(length);
    cdef DTYPE_f[:] end_idx = np.zeros(length);     
    
    cdef DTYPE_t j2;
    cdef DTYPE_f k1 = 0;

    cdef int tmp_front = 0; 
    cdef int tmp_end = length;
    
    for i in range(length):
        j2 = cond[i]
        if j2 == 1:
            tmp_end = i
            for k in range(tmp_front, tmp_end):
                end_idx[k] = i - 1
                to_end[k] = i - 1 - k
            tmp_front = i

        front_idx[i] = tmp_front
        to_front[i] = i - tmp_front


    for k in range(tmp_front, length):
        end_idx[k] = length - 1
        to_end[k] = length - 1 - k
    return np.asarray(to_front), np.asarray(to_end), np.asarray(front_idx), np.asarray(end_idx)
        




