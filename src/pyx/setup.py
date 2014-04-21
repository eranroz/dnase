from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="BedGraphReader",
    ext_modules=cythonize('*.pyx'),
    include_dirs=[np.get_include()],
    author='Eranroz',
    url='https://github.com/eranroz/BedGraphParser',
)