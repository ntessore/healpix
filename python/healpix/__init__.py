# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: BSD-3-Clause
'''Python package for HEALPix discretisation of the sphere'''

__version__ = '2023.4'


import numpy as np
import chealpix as _chp


NSIDE_MAX = _chp.NSIDE_MAX
'''Maximum admissible value for the NSIDE parameter.

If nside > NSIDE_MAX is used, the resulting pixel indices can overflow their
underlying integer type in the C library functions.

'''

def is_power_of_two(n):
    '''Return True if n is a power of two, or False otherwise.'''
    return (n & (n-1)) == 0


def check_nside(nside, nest=False):
    '''Check whether nside is a valid resolution parameter.'''
    if nside < 1 or nside > NSIDE_MAX:
        raise ValueError('nside must be a positive integer 1 <= nside <= '
                         f'2^{np.log2(NSIDE_MAX):.0f}')
    if nest and (nside & (nside-1)) != 0:
        raise ValueError('nside must be power of two in the NEST scheme');


def thetaphi_from_lonlat(lon, lat, theta=None, phi=None):
    '''convert longitude and latitude in degree to theta and phi in radian'''
    theta, phi = np.deg2rad(lat, out=theta), np.deg2rad(lon, out=phi)
    theta *= -1
    theta += np.pi/2
    return theta, phi


def lonlat_from_thetaphi(theta, phi, lon=None, lat=None):
    '''convert theta and phi in radian to longitude and latitude in degree'''
    lon, lat = np.rad2deg(phi, out=lon), np.rad2deg(theta, out=lat)
    lat *= -1
    lat += 90
    return lon, lat


def ang2vec(theta, phi, lonlat=False):
    out1, out2 = None, None
    if lonlat:
        s = np.broadcast(theta, phi).shape
        out1, out2 = np.empty(s), np.empty(s)
        theta, phi = thetaphi_from_lonlat(theta, phi, theta=out1, phi=out2)
    return _chp.ang2vec(theta, phi, out1, out2)


def vec2ang(x, y, z, lonlat=False):
    theta, phi = _chp.vec2ang(x, y, z)
    if lonlat:
        out1, out2 = (phi, theta) if np.ndim(theta) else (None, None)
        theta, phi = lonlat_from_thetaphi(theta, phi, lon=out1, lat=out2)
    return theta, phi


def ang2pix(nside, theta, phi, nest=False, lonlat=False):
    check_nside(nside, nest)
    if not lonlat:
        if nest:
            return _chp.ang2nest(nside, theta, phi)
        else:
            return _chp.ang2ring(nside, theta, phi)
    with np.nditer([theta, phi, None],
                   ['buffered', 'external_loop', 'zerosize_ok'],
                   [['readonly']]*2 + [['writeonly', 'allocate']]*1,
                   [None, None, int]) as it:
        for lon, lat, ipix in it:
            theta, phi = thetaphi_from_lonlat(lon, lat)
            if nest:
                _chp.ang2nest(nside, theta, phi, ipix)
            else:
                _chp.ang2ring(nside, theta, phi, ipix)
        return it.operands[2]


def pix2ang(nside, ipix, nest=False, lonlat=False):
    if nest:
        theta, phi = _chp.nest2ang(nside, ipix)
    else:
        theta, phi = _chp.ring2ang(nside, ipix)
    if lonlat:
        out1, out2 = (phi, theta) if np.ndim(theta) else (None, None)
        theta, phi = lonlat_from_thetaphi(theta, phi, lon=out1, lat=out2)
    return theta, phi


def vec2pix(nside, x, y, z, nest=False):
    check_nside(nside, nest)
    if nest:
        return _chp.vec2nest(nside, x, y, z)
    else:
        return _chp.vec2ring(nside, x, y, z)


def pix2vec(nside, ipix, nest=False):
    if nest:
        return _chp.nest2vec(nside, ipix)
    else:
        return _chp.ring2vec(nside, ipix)


def nside2npix(nside):
    return _chp.nside2npix(nside)


def npix2nside(npix):
    return _chp.npix2nside(npix)


def randang(nside, ipix, nest=False, lonlat=False, rng=None):
    '''Sample random spherical coordinates from the given HEALPix pixels.

    This function produces one pair of random spherical coordinates from each
    HEALPix pixel listed in `ipix`, which can be scalar or array-like.  The
    indices use the `nside` resolution parameter and either the RING scheme
    (if `nest` is false, the default) or the NEST scheme (if `nest` is true).

    Returns either a tuple `theta, phi` of mathematical coordinates in radians
    if `lonlat` is False (the default), or a tuple `lon, lat` of longitude and
    latitude in degrees if `lonlat` is True.  The output is of the same shape
    as the input.

    An optional numpy random number generator can be provided using `rng`;
    otherwise, a new numpy.random.default_rng() is used.
    '''

    if rng is None:
        rng = np.random.default_rng()

    u = rng.random(ipix.shape, np.double)
    v = rng.random(ipix.shape, np.double)

    if nest:
        theta, phi = _chp.nest2ang_uv(nside, ipix, u, v, u, v)
    else:
        theta, phi = _chp.ring2ang_uv(nside, ipix, u, v, u, v)

    if lonlat:
        out1, out2 = (phi, theta) if np.ndim(theta) else (None, None)
        theta, phi = lonlat_from_thetaphi(theta, phi, lon=out1, lat=out2)

    return theta, phi


def randvec(nside, ipix, nest=False, rng=None):
    '''Sample random unit vectors from the given HEALPix pixels.

    This function produces one random unit vector from each HEALPix pixel
    listed in `ipix`, which can be scalar or array-like.  The pixel indices use
    the `nside` resolution parameter and either the RING scheme (if `nest` is
    false, the default) or the NEST scheme (if `nest` is true).

    Returns a tuple `x, y, z` of normalised vector components with the same
    shape as the input.

    An optional numpy random number generator can be provided using `rng`;
    otherwise, a new numpy.random.default_rng() is used.
    '''

    if rng is None:
        rng = np.random.default_rng()

    u = rng.random(ipix.shape, np.double)
    v = rng.random(ipix.shape, np.double)

    if nest:
        return _chp.nest2vec_uv(nside, ipix, u, v, u, v)
    else:
        return _chp.ring2vec_uv(nside, ipix, u, v, u, v)


def nest2ring(nside, ipix):
    return _chp.nest2ring(nside, ipix)


def ring2nest(nside, ipix):
    return _chp.ring2nest(nside, ipix)


def uniq2pix(uniq, nest=False):
    '''Convert from UNIQ to RING or NEST pixel scheme.

    Returns a tuple `nside, ipix` of resolution parameters and pixel
    indices in the RING scheme (if `nest` is false, the default) or the
    NEST scheme (if `nest` is true).

    '''
    if nest:
        nside, ipix = _chp.uniq2nest(uniq)
    else:
        nside, ipix = _chp.uniq2ring(uniq)
    return nside, ipix


def pix2uniq(nside, ipix, nest=False):
    '''Convert RING or NEST to UNIQ pixel scheme.

    Returns a pixel index in the UNIQ scheme for each pair of resolution
    parameter `nside` and pixel index `ipix` in the RING scheme (if
    `nest` is false, the default) or the NEST scheme (if `nest` is
    true).

    '''
    if nest:
        uniq = _chp.nest2uniq(nside, ipix)
    else:
        uniq = _chp.ring2uniq(nside, ipix)
    return uniq
