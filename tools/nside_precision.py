# test that numerical operations retain precision for high nside

from chealpix import (ring2ang, ang2ring, nest2ang, ang2nest,
                      ring2vec, vec2ring, nest2vec, vec2nest,
                      nest2ring, ring2nest)


nsec = 12

for k in range(1, 30):
    nside = 1<<k
    npix = 12*nside**2

    ipixv = [list(range(nsec)),
             list(range((npix-nsec)//2, (npix-nsec)//2+nsec)),
             list(range(npix-nsec, npix))]

    print(f'2^{k:<7d}', 'angXnest ....', end='')
    for ipix in ipixv:
        print(' ', (ang2nest(nside, *nest2ang(nside, ipix)) == ipix).astype(int), end='')
    print()
    print(f'         ', 'angXring ....', end='')
    for ipix in ipixv:
        print(' ', (ang2ring(nside, *ring2ang(nside, ipix)) == ipix).astype(int), end='')
    print()
    print(f'         ', 'vecXnest ....', end='')
    for ipix in ipixv:
        print(' ', (vec2nest(nside, *nest2vec(nside, ipix)) == ipix).astype(int), end='')
    print()
    print(f'         ', 'vecXring ....', end='')
    for ipix in ipixv:
        print(' ', (vec2ring(nside, *ring2vec(nside, ipix)) == ipix).astype(int), end='')
    print()
    print(f'         ', 'ringXnest ...', end='')
    for ipix in ipixv:
        print(' ', (nest2ring(nside, ring2nest(nside, ipix)) == ipix).astype(int), end='')
    print()
    print(f'         ', 'nestXring ...', end='')
    for ipix in ipixv:
        print(' ', (ring2nest(nside, nest2ring(nside, ipix)) == ipix).astype(int), end='')
    print()
