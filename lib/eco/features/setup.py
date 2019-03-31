# from setuptools import setup, Extension
from distutils.core import setup, Extension
from numpy.distutils import misc_util

c_ext = Extension("_gradient", ["_gradient.cpp", "gradient.cpp"])

setup(
    ext_modules=[c_ext],
    include_dirs = misc_util.get_numpy_include_dirs(),
)
