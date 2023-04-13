healpix
=======

**Python and C package for HEALPix discretisation of the sphere**

This package implements a lean set of routines for working with the HEALPix
discretisation of the sphere.  It supports NSIDE parameters up to 2^29.

The C library is based on the *healpix_bare* library, which was released under
the 3-clause BSD license, with the following additions:

* Sub-pixel indexing.
* Conversions between the UNIQ and RING/NEST pixel indexing schemes.

If you are using this code in your research, please consider citing the
original paper in your publications:

* [Gorski et al., 2005, ApJ, 622, 759][Gorski+2005]

[Gorski+2005]: http://adsabs.harvard.edu/abs/2005ApJ...622..759G


Python
------

The Python package provides functions that deal with the discretisation of the
sphere itself.  It is not meant to be a replacement of *healpy*, and does not
contain things such as functions for the visualisation of maps, or spherical
harmonic transforms.

The Python package consists of two modules:

* The low-level `chealpix` module, which is a native C extension, and
  efficiently vectorises the C library functions over arbitrary numpy array
  inputs (including scalars, of course).
* The high-level `healpix` module, which contains a more streamlined interface,
  and additional functionality:

  * Random point picking in HEALPix pixels.

For a function reference, run `pydoc healpix` (or `pydoc chealpix`) locally if
you have the package installed, or see the [online reference][pydoc].

The high-level functions in the `healpix` module can be used more or less
interchangeably with functions from the *healpy* package.  However, in some
cases, compatibility is sacrificed for consistency.

The Python package requires only *numpy*, and can be installed using pip:

    pip install healpix

The vectorised C functions carefully avoid the creation of temporary arrays,
and therefore have minimal memory overhead:

```py
>>> import numpy as np, healpix, tracemalloc
>>> 
>>> # random vectors with 1G of memory per component (less than a NSIDE=4K map)
>>> x, y, z = np.random.randn(3, 125_000_000)
>>> 
>>> tracemalloc.start()
>>> 
>>> lon, lat = healpix.vec2ang(x, y, z, lonlat=True)
>>> 
>>> tracemalloc.get_traced_memory()
(2000010342, 2000013889)  # current, peak
>>> 
>>> # no memory overhead: only the 2G output arrays were used
```

[pydoc]: https://github.com/ntessore/healpix/raw/main/python/reference.txt
