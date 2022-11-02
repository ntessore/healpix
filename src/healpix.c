// based on the healpix_bare library
// license: BSD-3-Clause

#include <math.h>
#include "healpix.h"

static const double PI = 3.141592653589793238462643383279502884197;

static const int jrll[] = { 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
static const int jpll[] = { 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7 };


// maximum nside for 64bit signed integer types
const int64_t NSIDE_MAX = 1<<29;


// faster inversion of z than atan2(s, z)
static double invz(double z, double s) {
    return s > 0.7 ? acos(z) : z > 0 ? asin(s) : PI - asin(s);
}


/* conversions between continuous coordinate systems */

// A structure describing a location in cylindrical coordinates.
// z = cos(theta), s = sin(theta), phi
typedef struct {
    double z, s, phi;
} t_loc;


static t_loc ang2loc(t_ang ang) {
    double z = cos(ang.theta), s = sin(ang.theta);
    if (s < 0) {
          s = -s;
          ang.phi += PI;
    }
    return (t_loc){z, s, ang.phi};
}


static t_ang loc2ang(t_loc loc) {
    return (t_ang){invz(loc.z, loc.s), loc.phi};
}


static t_loc vec2loc(t_vec vec) {
    double s = hypot(vec.x, vec.y);
    double r = hypot(s, vec.z);
    return (t_loc){vec.z/r, s/r, atan2(vec.y, vec.x)};
}


static t_vec loc2vec(t_loc loc) {
    return (t_vec){loc.s*cos(loc.phi), loc.s*sin(loc.phi), loc.z};
}


t_vec ang2vec(t_ang ang) {
    return loc2vec(ang2loc(ang));
}


t_ang vec2ang(t_vec vec) {
  return (t_ang){atan2(hypot(vec.x, vec.y), vec.z), atan2(vec.y, vec.x)};
}


// conversions between discrete coordinate systems


static int64_t isqrt(int64_t v) {
    int64_t res = sqrt(v+0.5);
    if (v < ((int64_t)(1)<<50))
        return res;
    if (res*res > v) {
        --res;
    } else if ((res+1)*(res+1) <= v) {
        ++res;
    }
    return res;
}


static int64_t spread_bits(int64_t v) {
  int64_t res = v & 0xffffffff;
  res = (res^(res<<16)) & 0x0000ffff0000ffff;
  res = (res^(res<< 8)) & 0x00ff00ff00ff00ff;
  res = (res^(res<< 4)) & 0x0f0f0f0f0f0f0f0f;
  res = (res^(res<< 2)) & 0x3333333333333333;
  res = (res^(res<< 1)) & 0x5555555555555555;
  return res;
}


static int64_t compress_bits(int64_t v) {
  int64_t res = v & 0x5555555555555555;
  res = (res^(res>> 1)) & 0x3333333333333333;
  res = (res^(res>> 2)) & 0x0f0f0f0f0f0f0f0f;
  res = (res^(res>> 4)) & 0x00ff00ff00ff00ff;
  res = (res^(res>> 8)) & 0x0000ffff0000ffff;
  res = (res^(res>>16)) & 0x00000000ffffffff;
  return res;
}


// A structure describing the discrete Healpix coordinate system.
// f takes values in [0, 11], a and b lie in [0, nside).
typedef struct {
    int64_t a, b;
    int32_t f;
} t_hpd;


static int64_t hpd2nest(int64_t nside, t_hpd hpd) {
    return (hpd.f*nside*nside) + spread_bits(hpd.a) + (spread_bits(hpd.b)<<1);
}


static t_hpd nest2hpd(int64_t nside, int64_t pix) {
    int64_t npface_=nside*nside, p2=pix&(npface_-1);
    return (t_hpd){compress_bits(p2), compress_bits(p2>>1), pix/npface_};
}


static int64_t hpd2ring(int64_t nside, t_hpd hpd) {
  int64_t nl4 = 4*nside;
  int64_t jr = (jrll[hpd.f]*nside) - hpd.a - hpd.b - 1;

  if (jr<nside)
    {
    int64_t jp = (jpll[hpd.f]*jr + hpd.a - hpd.b + 1) / 2;
    jp = (jp>nl4) ? jp-nl4 : ((jp<1) ? jp+nl4 : jp);
    return 2*jr*(jr-1) + jp - 1;
    }
  else if (jr > 3*nside)
    {
    jr = nl4-jr;
    int64_t jp = (jpll[hpd.f]*jr + hpd.a - hpd.b + 1) / 2;
    jp = (jp>nl4) ? jp-nl4 : ((jp<1) ? jp+nl4 : jp);
    return 12*nside*nside - 2*(jr+1)*jr + jp - 1;
    }
  else
    {
    int64_t jp = (jpll[hpd.f]*nside + hpd.a - hpd.b + 1 + ((jr-nside)&1)) / 2;
    jp = (jp>nl4) ? jp-nl4 : ((jp<1) ? jp+nl4 : jp);
    return 2*nside*(nside-1) + (jr-nside)*nl4 + jp - 1;
    }
}


static t_hpd ring2hpd(int64_t nside, int64_t pix) {
  int64_t ncap_=2*nside*(nside-1);
  int64_t npix_=12*nside*nside;

  if (pix<ncap_) /* North Polar cap */
    {
    int64_t iring = (1+isqrt(1+2*pix))>>1; /* counted from North pole */
    int64_t iphi  = (pix+1) - 2*iring*(iring-1);
    int64_t face = (iphi-1)/iring;
    int64_t irt = iring - (jrll[face]*nside) + 1;
    int64_t ipt = 2*iphi- jpll[face]*iring -1;
    if (ipt>=2*nside) ipt-=8*nside;
    return (t_hpd) {(ipt-irt)>>1, (-(ipt+irt))>>1, face};
    }
  else if (pix<(npix_-ncap_)) /* Equatorial region */
    {
    int64_t ip = pix - ncap_;
    int64_t iring = (ip/(4*nside)) + nside; /* counted from North pole */
    int64_t iphi  = (ip%(4*nside)) + 1;
    int64_t kshift = (iring+nside)&1;
    int64_t ire = iring-nside+1;
    int64_t irm = 2*nside+2-ire;
    int64_t ifm = (iphi - ire/2 + nside -1) / nside;
    int64_t ifp = (iphi - irm/2 + nside -1) / nside;
    int64_t face = (ifp==ifm) ? (ifp|4) : ((ifp<ifm) ? ifp : (ifm+8));
    int64_t irt = iring - (jrll[face]*nside) + 1;
    int64_t ipt = 2*iphi- jpll[face]*nside - kshift -1;
    if (ipt>=2*nside) ipt-=8*nside;
    return (t_hpd) {(ipt-irt)>>1, (-(ipt+irt))>>1, face};
    }
  else /* South Polar cap */
    {
    int64_t ip = npix_ - pix;
    int64_t iring = (1+isqrt(2*ip-1))>>1; /* counted from South pole */
    int64_t iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1));
    int64_t face=8+(iphi-1)/iring;
    int64_t irt = 4*nside - iring - (jrll[face]*nside) + 1;
    int64_t ipt = 2*iphi- jpll[face]*iring -1;
    if (ipt>=2*nside) ipt-=8*nside;
    return (t_hpd) {(ipt-irt)>>1, (-(ipt+irt))>>1, face};
    }
}


int64_t nest2ring(int64_t nside, int64_t ipnest) {
    // power of two check
    if ((nside & (nside-1)) != 0)
        return -1;
    return hpd2ring(nside, nest2hpd(nside, ipnest));
}


int64_t ring2nest(int64_t nside, int64_t ipring) {
    // power of two check
    if ((nside&(nside-1)) != 0)
        return -1;
    return hpd2nest(nside, ring2hpd(nside, ipring));
}


// mixed conversions


static t_hpd loc2hpd(int64_t nside, t_loc loc, double* u, double* v) {
    t_hpd hpd;
    double za = fabs(loc.z);
    double x = loc.phi*(1./(2.*PI));
    if (x < 0.) {
        x += (int64_t)x + 1.;
    } else if (x >= 1.) {
        x -= (int64_t)x;
    }
    double tt = 4.*x;

    if (za <= 2./3.) {
        // equatorial region
        double temp1 = 0.5+tt;      // [0.5; 4.5)
        double temp2 = loc.z*0.75;  // [-0.5; +0.5]
        double jp = temp1-temp2;    // index of ascending edge line, [0; 5)
        double jm = temp1+temp2;    // index of descending edge line, [0; 5)
        int ifp = (int)jp;          // in {0,4}
        int ifm = (int)jm;
        hpd.a = (jm-ifm)*nside;
        hpd.b = (1+ifp - jp)*nside;
        hpd.f = (ifp==ifm) ? (ifp|4) : ((ifp<ifm) ? ifp : (ifm+8));
        if (u) {
            *u = (jm-ifm)*nside - hpd.a;
            *v = (1+ifp - jp)*nside - hpd.b;
        }
    } else {
        // polar region
        int64_t ntt = (int64_t)tt;
        if (ntt >= 4)
            ntt = 3;
        double tp = tt - ntt;       // [0;1)
        double tmp = sqrt(3.*(1.-za));

        double jp = tp*tmp;         // increasing edge line index
        double jm = (1.0-tp)*tmp;   // decreasing edge line index
        if (jp > 1.)
            jp = 1.;                // for points too close to the boundary
        if (jm > 1.)
            jm = 1.;
        if (loc.z >= 0) {
            tmp = 1. - jp;
            jp = 1. - jm;
            jm = tmp;
        } else {
            ntt += 8;
        }
        hpd.a = jp*nside;
        hpd.b = jm*nside;
        hpd.f = ntt;
        if (u) {
            *u = jp*nside - hpd.a;
            *v = jm*nside - hpd.b;
        }
    }
    return hpd;
}


static t_loc hpd2loc(int64_t nside, t_hpd hpd, double dx, double dy) {
    double z, s, phi;
    const double x = (hpd.a - hpd.b + dx)/nside;
    const double y = (hpd.a + hpd.b - nside + dy)/nside;
    const int32_t r = 1 - hpd.f/4;
    const double m = 1 - r*y;
    if (m < 1) {
        // polar cap
        const double tmp = m*m*(1./3.);
        z = r*(1. - tmp);
        s = sqrt(tmp*(2.-tmp));
        phi = (PI/4.)*(jpll[hpd.f] + x/m);
    } else {
        // equatorial region
        z = (y+r)*(2./3.);
        s = sqrt((1.+z)*(1.-z));
        phi = (PI/4.)*(jpll[hpd.f] + x);
    }
    return (t_loc){z, s, phi};
}


int64_t npix2nside(int64_t npix) {
    int64_t res = isqrt(npix/12);
    return (res*res*12 == npix) ? res : -1;
}


int64_t nside2npix(int64_t nside) {
    return 12*nside*nside;
}


double vec_angle(t_vec v1, t_vec v2) {
    t_vec cross = {
        v1.y*v2.z - v1.z*v2.y,
        v1.z*v2.x - v1.x*v2.z,
        v1.x*v2.y - v1.y*v2.x
    };
    double len = sqrt(cross.x*cross.x + cross.y*cross.y + cross.z*cross.z);
    double dot = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
    return atan2(len, dot);
}


int64_t ang2ring(int64_t nside, t_ang ang) {
    return hpd2ring(nside, loc2hpd(nside, ang2loc(ang), 0, 0));
}


int64_t ang2nest(int64_t nside, t_ang ang) {
    return hpd2nest(nside, loc2hpd(nside, ang2loc(ang), 0, 0));
}


int64_t vec2ring(int64_t nside, t_vec vec) {
    return hpd2ring(nside, loc2hpd(nside, vec2loc(vec), 0, 0));
}


int64_t vec2nest(int64_t nside, t_vec vec) {
    return hpd2nest(nside, loc2hpd(nside, vec2loc(vec), 0, 0));
}


t_ang ring2ang(int64_t nside, int64_t ipix) {
    return loc2ang(hpd2loc(nside, ring2hpd(nside, ipix), 0, 1));
}


t_ang nest2ang(int64_t nside, int64_t ipix) {
    return loc2ang(hpd2loc(nside, nest2hpd(nside, ipix), 0, 1));
}


t_vec ring2vec(int64_t nside, int64_t ipix) {
    return loc2vec(hpd2loc(nside, ring2hpd(nside, ipix), 0, 1));
}


t_vec nest2vec(int64_t nside, int64_t ipix) {
    return loc2vec(hpd2loc(nside, nest2hpd(nside, ipix), 0, 1));
}


// conversions with sub-pixel positions


int64_t ang2ring_uv(int64_t nside, t_ang ang, double* u, double* v) {
    return hpd2ring(nside, loc2hpd(nside, ang2loc(ang), u, v));
}


int64_t ang2nest_uv(int64_t nside, t_ang ang, double* u, double* v) {
    return hpd2nest(nside, loc2hpd(nside, ang2loc(ang), u, v));
}


int64_t vec2ring_uv(int64_t nside, t_vec vec, double* u, double* v) {
    return hpd2ring(nside, loc2hpd(nside, vec2loc(vec), u, v));
}


int64_t vec2nest_uv(int64_t nside, t_vec vec, double* u, double* v) {
    return hpd2nest(nside, loc2hpd(nside, vec2loc(vec), u, v));
}


t_ang ring2ang_uv(int64_t nside, int64_t ipix, double u, double v) {
    return loc2ang(hpd2loc(nside, ring2hpd(nside, ipix), u-v, u+v));
}


t_ang nest2ang_uv(int64_t nside, int64_t ipix, double u, double v) {
    return loc2ang(hpd2loc(nside, nest2hpd(nside, ipix), u-v, u+v));
}


t_vec ring2vec_uv(int64_t nside, int64_t ipix, double u, double v) {
    return loc2vec(hpd2loc(nside, ring2hpd(nside, ipix), u-v, u+v));
}


t_vec nest2vec_uv(int64_t nside, int64_t ipix, double u, double v) {
    return loc2vec(hpd2loc(nside, nest2hpd(nside, ipix), u-v, u+v));
}
