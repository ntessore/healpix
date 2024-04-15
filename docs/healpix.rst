
:mod:`healpix` -- Python module
===============================

.. currentmodule:: healpix
.. module:: healpix

Python module for the HEALPix discretisation of the sphere.


Resolution
----------

.. autodata:: NSIDE_MAX

.. function:: nside2npix(nside)
              npix2nside(npix)

   Convert between resolution parameter *nside* and number of pixels *npix*.


.. function:: nside2order(nside)
              order2nside(order)

   Convert between resolution parameter *nside* and HEALPix order *order*.
   Requires *nside* to be a power of two.


Pixel routines
--------------

.. function:: ang2pix(nside, theta, phi, nest=False, lonlat=False)
              pix2ang(nside, ipix, nest=False, lonlat=False)

.. function:: pix2vec(nside, ipix, nest=False)
              vec2pix(nside, x, y, z, nest=False)

.. function:: ang2vec(theta, phi, lonlat=False)
              vec2ang(x, y, z, lonlat=False)

.. function:: ring2nest(nside, ipix)
              nest2ring(nside, ipix)


Subpixel indexing
-----------------


Random point picking
--------------------

.. autofunction:: randang
.. autofunction:: randvec


Multi-Order Coverage (MOC)
--------------------------

.. function:: pix2uniq(order, ipix, nest=False)
              uniq2pix(uniq, nest=False)

   Convert pixel indices to or from the UNIQ pixel indexing scheme.
