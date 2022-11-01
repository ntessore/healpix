from setuptools import setup, Extension
import os.path
import numpy as np


setup(
    ext_modules=[
        Extension('chealpix',
            sources=['python/chealpix.c', 'src/healpix.c'],
            include_dirs=[np.get_include(), 'src'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            extra_compile_args=['-std=c99'],
        ),
    ],
)
