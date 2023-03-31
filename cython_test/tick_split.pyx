# distutils: language = c++
# cython: language_level=3
cimport cython
cimport numpy as np
import numpy as np
np.ndarray

from libcpp.vector cimport vector
from libcpp cimport bool

cpdef test1(np.ndarray[np.float32_t] x, np.ndarray[np.float32_t] y):
    cdef vector[np.float32_t] sb;

    for i in :
        print(i)
        print(type(i))
        sb.push_back(i[0])
    return np.asarray([sb, sb])


cpdef np.ndarray[bool, ndim=1] test(int n):
    cdef vector[bool] pr
    cdef vector[int] res
    cdef int i = 0

    for i in range(n + 1):
        pr.push_back(True)
    return np.asarray(pr)
    



