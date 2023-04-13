import pytest
import numpy as np
import numpy.testing as npt

from . import assert_shape


def test_ang2vec(lonlat, size):
    from healpix import ang2vec
    if lonlat:
        theta_lon = np.random.uniform(0, 360, size=size)
        phi_lat = np.random.uniform(-90, 90, size=size)
        x_ = np.cos(np.deg2rad(phi_lat))*np.cos(np.deg2rad(theta_lon))
        y_ = np.cos(np.deg2rad(phi_lat))*np.sin(np.deg2rad(theta_lon))
        z_ = np.sin(np.deg2rad(phi_lat))
    else:
        theta_lon = np.random.uniform(0, np.pi, size=size)
        phi_lat = np.random.uniform(0, 2*np.pi, size=size)
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


def test_vec2ang(lonlat, size):
    from healpix import vec2ang
    x = np.random.standard_normal(size)
    y = np.random.standard_normal(size)
    z = np.random.standard_normal(size)
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


def test_uniq_pix(nest):
    from healpix import uniq2pix, pix2uniq, ring2nest
    nside = 2**np.random.randint(0, 30, 1000)
    ipix = np.random.randint(0, 12*nside**2)
    ipnest = ipix if nest else [ring2nest(ns, ip) for ns, ip in zip(nside, ipix)]
    uniq = pix2uniq(nside, ipix, nest)
    npt.assert_array_equal(uniq, 4*nside**2 + ipnest)
    nside_out, ipix_out = uniq2pix(uniq, nest)
    npt.assert_array_equal(nside_out, nside)
    npt.assert_array_equal(ipix_out, ipix)
