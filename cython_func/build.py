#distutils: language = c++

#python build.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
ext_modules = [
    Extension('split_calc',
              ['split_calc.pyx'], 
              include_dirs=[numpy.get_include()],
              language="c++",              
              )
]

setup(
    name = 'split_calc app',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[numpy.get_include()],

)

