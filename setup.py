from setuptools import setup, Extension
import os.path
import numpy as np


setup(
    ext_modules=[
        Extension('healpix.chealpix',
            sources=['python/healpix.c', 'src/healpix.c'],
            include_dirs=[np.get_include(), 'src'],
            library_dirs=[
                os.path.abspath(os.path.join(np.get_include(), '..', '..', 'random', 'lib')),
                os.path.abspath(os.path.join(np.get_include(), '..', 'lib'))
            ],
            libraries=['npyrandom', 'npymath'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            extra_compile_args=['-std=c99', '-O3'],
        ),
    ],
)
