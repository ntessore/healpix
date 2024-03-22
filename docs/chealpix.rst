
:mod:`chealpix` -- Python bindings
==================================

.. currentmodule:: chealpix
.. module:: chealpix

Vectorised Python bindings to the HEALPix C library.

.. data:: NSIDE_MAX

   Maximum admissible value for the NSIDE parameter.

   If nside > NSIDE_MAX is used, the resulting pixel indices can overflow their
   underlying integer type in the C library functions.
