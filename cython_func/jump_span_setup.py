# cython: language_level=3
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension('jump_span',
              ['jump_span.pyx'], 
              include_dirs=[numpy.get_include()]
              )
]

setup(
  name = 'jump_span app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
    include_dirs=[numpy.get_include()]    
)


