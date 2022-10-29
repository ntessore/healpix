import numpy as np
import pytest
import healpix


def test_ang2pix_pix2ang():
    for nest in False, True:
        for lonlat in False, True:
            for nside in 1, 16, 256:
                ipix = np.arange(12*nside**2)
                theta, phi = healpix.pix2ang(nside, ipix, nest, lonlat)
                if lonlat:
                    # alias for clarity
                    lon, lat = theta, phi
                    np.testing.assert_array_less(np.nextafter(0, -1), lon)
                    np.testing.assert_array_less(lon, 360)
                    np.testing.assert_array_less(-90, lat)
                    np.testing.assert_array_less(lat, 90)
                else:
                    np.testing.assert_array_less(0, theta)
                    np.testing.assert_array_less(theta, np.pi)
                    np.testing.assert_array_less(np.nextafter(0, -1), phi)
                    np.testing.assert_array_less(phi, 2*np.pi)
                ipix_ = healpix.ang2pix(nside, theta, phi, nest, lonlat)
                np.testing.assert_equal(ipix, ipix_)

    with pytest.raises(ValueError, match='power of two'):
        healpix.pix2ang(11, 3, nest=True)


def test_vec2pix_pix2vec():
    for nest in False, True:
        for nside in 1, 16, 256:
            ipix = np.arange(12*nside**2)
            x, y, z = healpix.pix2vec(nside, ipix, nest)
            np.testing.assert_allclose((x**2 + y**2 + z**2), 1)
            ipix_ = healpix.vec2pix(nside, x, y, z, nest)
            np.testing.assert_equal(ipix, ipix_)

    with pytest.raises(ValueError, match='power of two'):
        healpix.pix2ang(11, 3, nest=True)
