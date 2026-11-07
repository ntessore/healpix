import numpy as np
import numpy.testing as npt

from . import assert_shape


def test_ang2vec(lonlat, size, rng):
    from healpix import ang2vec
    if lonlat:
        theta_lon = rng.uniform(0, 360, size=size)
        phi_lat = rng.uniform(-90, 90, size=size)
        x_ = np.cos(np.deg2rad(phi_lat))*np.cos(np.deg2rad(theta_lon))
        y_ = np.cos(np.deg2rad(phi_lat))*np.sin(np.deg2rad(theta_lon))
        z_ = np.sin(np.deg2rad(phi_lat))
    else:
        theta_lon = rng.uniform(0, np.pi, size=size)
        phi_lat = rng.uniform(0, 2*np.pi, size=size)
        x_ = np.sin(theta_lon)*np.cos(phi_lat)
        y_ = np.sin(theta_lon)*np.sin(phi_lat)
        z_ = np.cos(theta_lon)
    x, y, z = ang2vec(theta_lon, phi_lat, lonlat=lonlat)
    npt.assert_allclose(x, x_)
    npt.assert_allclose(y, y_)
    npt.assert_allclose(z, z_)
    assert_shape(x, size)
    assert_shape(y, size)
    assert_shape(z, size)


def test_vec2ang(lonlat, size, rng):
    from healpix import vec2ang
    x = rng.standard_normal(size)
    y = rng.standard_normal(size)
    z = rng.standard_normal(size)
    theta_, phi_ = np.arctan2(np.hypot(x, y), z), np.arctan2(y, x)
    if lonlat:
        theta_, phi_ = np.rad2deg(phi_), np.rad2deg(np.pi/2-theta_)
    theta, phi = vec2ang(x, y, z, lonlat=lonlat)
    npt.assert_allclose(theta, theta_)
    npt.assert_allclose(phi, phi_)
    assert_shape(theta, size)
    assert_shape(phi, size)


def test_ang2pix_base(base_pixel_ang, nest, lonlat):
    from healpix import ang2pix
    npt.assert_array_equal(ang2pix(1, *base_pixel_ang, nest, lonlat), np.arange(12))


def test_pix2ang_base(base_pixel_ang, nest, lonlat):
    from healpix import pix2ang
    npt.assert_allclose(pix2ang(1, np.arange(12), nest, lonlat), base_pixel_ang)


def test_ang_pix(nside, nest, lonlat):
    from healpix import ang2pix, pix2ang
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(ang2pix(nside, *pix2ang(nside, ipix, nest, lonlat), nest, lonlat), ipix)


def test_vec2pix_base(base_pixel_vec, nest):
    from healpix import vec2pix
    npt.assert_array_equal(vec2pix(1, *base_pixel_vec, nest), np.arange(12))


def test_pix2vec_base(base_pixel_vec, nest):
    from healpix import pix2vec
    npt.assert_allclose(pix2vec(1, np.arange(12), nest), base_pixel_vec, atol=1e-15)


def test_vec_pix(nside, nest):
    from healpix import vec2pix, pix2vec
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(vec2pix(nside, *pix2vec(nside, ipix, nest), nest), ipix)


def test_uniq_pix(nest, rng):
    from healpix import uniq2pix, pix2uniq, ring2nest
    order = rng.integers(0, 30, 1000)
    nside = 1 << order
    ipix = rng.integers(0, 12*nside**2)
    ipnest = ipix if nest else [ring2nest(ns, ip) for ns, ip in zip(nside, ipix)]
    uniq = pix2uniq(order, ipix, nest)
    npt.assert_array_equal(uniq, 4*nside**2 + ipnest)
    order_out, ipix_out = uniq2pix(uniq, nest)
    npt.assert_array_equal(order_out, order)
    npt.assert_array_equal(ipix_out, ipix)


def test_zuniq():
    from healpix import uniq2zuniq, zuniq2uniq

    # First and last nested cells at every refinement level.
    order = np.arange(30, dtype=np.int64)
    face_size = 4**order
    for nested in (np.zeros_like(order), 12*face_size - 1):
        uniq = 4*face_size + nested
        zuniq = (2*nested + 1) * 4**(29-order)
        npt.assert_array_equal(uniq2zuniq(uniq), zuniq)
        npt.assert_array_equal(zuniq2uniq(zuniq), uniq)

    # All four children of the first face, with a strided output buffer.
    out = np.full(8, -1, dtype=np.int64)
    uniq2zuniq(np.arange(16, 20), out[::2])
    npt.assert_array_equal(out[::2], np.array([1, 3, 5, 7]) * 4**28)
    npt.assert_array_equal(out[1::2], -1)
    npt.assert_array_equal(zuniq2uniq(out[-2::-2]), [19, 18, 17, 16])
    assert zuniq2uniq(4**29) == 4


def test_invalid_zuniq():
    from healpix import uniq2zuniq, zuniq2uniq

    npt.assert_array_equal(uniq2zuniq([-1, 0, 3, 4**31]), -1)
    npt.assert_array_equal(zuniq2uniq([-1, 0, 2, 4**30, 24*4**29]), -1)
