from setuptools import setup
from Cython.Build import cythonize
import os
import sys
print(os.getcwd())

from setuptools import setup
from Cython.Build import cythonize
import numpy as np
setup(
    name='My Project',
    ext_modules=cythonize('algorithms.pyx'),
    include_dirs=[np.get_include()]
)
