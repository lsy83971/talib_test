import setuptools
from distutils.core import setup
from Cython.Build import cythonize

setup(name="test", ext_modules=cythonize('logit.py'))
gg = sys.version.find("MSC v.")
sys.version
