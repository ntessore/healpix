healpix
=======

**Python and C package for HEALPix discretisation of the sphere**

This package implements a lean set of routines for working with the HEALPix
discretisation of the sphere.

* HEALPix pixel functions for NSIDE up to 2^29
* Random sampling of points in HEALPix pixels

The C library is based on the *healpix_bare* library, which was released under
the 3-clause BSD license.

If you are using this code in your research, please consider citing the
original paper in your publications:

* [Gorski et al., 2005, ApJ, 622, 759][Gorski+2005]

[Gorski+2005]: http://adsabs.harvard.edu/abs/2005ApJ...622..759G


Python
------

The Python package is a minimal set of routines for working with the HEALPix
discretrisation of the sphere itself.  It is not meant to be a replacement of
e.g. *healpy*, and does not contain things such as visualisation functions or
spherical harmonic transforms.

**The Python package is work in progress.  Not all library functions have been
added yet.  Feel free to request addition of any function that is in the C
library to speed up the process.**

The Python package is implemented as a pure C extension which essentially only
applies the library functions to inputs which can be scalar or array-like.  The
main goal is to do so as efficiently as possible, and never copy or allocate
(potentially huge) arrays needlessly.

The Python package requires only *numpy*, and can be installed using pip:

    pip install healpix

The Python interface is kept broadly in line with the *healpy* package, which
in turn is similar to the interface of the original HEALPix C library.  The
Python and C interfaces therefore largely resemble one another.

For a reference of the available functions, see `pydoc healpix` if you have
the package installed locally, or the [Python function reference][py-ref]
online.

[py-ref]: https://github.com/ntessore/healpix/raw/main/python-reference.txt
