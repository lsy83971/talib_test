#distutils: language = c++
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
ext_modules = [
    Extension('tss',
              ['tick_split.pyx'], 
              include_dirs=[numpy.get_include()],
              language="c++",              
              )
]

setup(
    name = 'jump_span app',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[numpy.get_include()],

)

