import numpy as np
import matplotlib.pyplot as plt

import healpix

# number of points to sample
npts = 1_000_000

# nsides to sample from
nsides = [1, 8, 64, 512]

# number of bins for histograms
nbin = 40

# show 5 sigma values in hist2d
vmin = (1 - 5*nbin/npts**0.5)/(4*np.pi)
vmax = (1 + 5*nbin/npts**0.5)/(4*np.pi)

# seed the random number generator for git's sake
rng = np.random.default_rng(12345)

# produce a histogram of the randang distribution for each nside
for nside in nsides:

    # sample on average npts points for each side:
    # each pixel gets its counts from a Poisson distribution, since
    # fixing the number of points per pixel changes the distribution
    npix = 12*nside**2
    cnts = np.random.poisson(npts/npix, size=npix)
    ipix = np.repeat(np.arange(npix), cnts)

    theta, phi = healpix.randang(nside, ipix, rng=rng)

    fig, ax = plt.subplots(2, 2, figsize=(5, 5), sharex='col')

    ax[0, 0].hist(phi % (2*np.pi), bins=nbin, range=[0, 2*np.pi], density=True,
                  histtype='step')
    ax[1, 1].hist(np.cos(theta), bins=nbin, range=[-1, 1], density=True,
                  histtype='step')
    ax[1, 1].set_xlabel('$\\cos\\theta$')
    ax[1, 0].hist2d(phi % (2*np.pi), np.cos(theta),
                    bins=nbin, range=[[0, 2*np.pi], [-1, 1]], density=True,
                    cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax[1, 0].set_xlabel('$\\phi$ [rad]')
    ax[1, 0].set_ylabel('$\\cos\\theta$')
    ax[0, 1].axis('off')
    ax[1, 0].set_xlim(-0.1*np.pi, 2.1*np.pi)
    ax[1, 0].set_ylim(-1.1, 1.1)
    ax[1, 1].set_xlim(-1.1, 1.1)

    fig.suptitle(f'randang distribution for NSIDE = {nside}')
    fig.tight_layout()

    plt.show()
