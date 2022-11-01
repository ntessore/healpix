# test that numerical operations retain precision for high nside

import healpix

for k in range(30):
    nside = 1<<k
    npix = 12*nside**2

    # test first 4 pixels, middle 4 pixels, last 4 pixels (in RING scheme)
    ipix = [0, 1, 2, 3,
            npix//2 - 2, npix//2 - 1, npix//2, npix//2 + 1,
            npix - 4, npix - 3, npix - 2, npix - 1]

    # transform forward and backward using ang/vec functions and compute offset
    dang = healpix.ang2pix(nside, *healpix.pix2ang(nside, ipix)) - ipix
    dvec = healpix.vec2pix(nside, *healpix.pix2vec(nside, ipix)) - ipix

    print(f'2^{k:<2d}  {dang} // {dvec}')
