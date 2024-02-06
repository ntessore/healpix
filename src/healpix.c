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


#ifdef _MSC_VER
#include <intrin.h>  // for _BitScanReverse64
#endif


// count leading zeros
static inline int clz(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#elif defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_X64))
    unsigned long bsr;
    return _BitScanReverse64(&bsr, x) ? 63 - bsr : 64;
#else
    // taken from https://stackoverflow.com/a/70550680 under CC BY-SA 4.0
    static const uint8_t clz64_tab[64] = {
        63,  5, 62,  4, 16, 10, 61,  3, 24, 15, 36,  9, 30, 21, 60,  2,
        12, 26, 23, 14, 45, 35, 43,  8, 33, 29, 52, 20, 49, 41, 59,  1,
         6, 17, 11, 25, 37, 31, 22, 13, 27, 46, 44, 34, 53, 50, 42,  7,
        18, 38, 32, 28, 47, 54, 51, 19, 39, 48, 55, 40, 56, 57, 58,  0,
    };
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return clz64_tab[(uint64_t)(x * 0x03f6eaf2cd271461u) >> 58];
#endif
}

// fast integer log2
int ilog2(uint64_t x) {
    return x == 0 ? -1 : 63 - clz(x);
}


// fast integer square root
// taken from https://stackoverflow.com/a/70550680 under CC BY-SA 4.0
uint32_t isqrt(uint64_t x) {
    // isqrt64_tab[k] = isqrt(256*(k+65)-1) for 0 <= k < 192
    static const uint8_t isqrt64_tab[192] = {
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
        138, 139, 140, 141, 142, 143, 143, 144, 145, 146,
        147, 148, 149, 150, 150, 151, 152, 153, 154, 155,
        155, 156, 157, 158, 159, 159, 160, 161, 162, 163,
        163, 164, 165, 166, 167, 167, 168, 169, 170, 170,
        171, 172, 173, 173, 174, 175, 175, 176, 177, 178,
        178, 179, 180, 181, 181, 182, 183, 183, 184, 185,
        185, 186, 187, 187, 188, 189, 189, 190, 191, 191,
        192, 193, 193, 194, 195, 195, 196, 197, 197, 198,
        199, 199, 200, 201, 201, 202, 203, 203, 204, 204,
        205, 206, 206, 207, 207, 208, 209, 209, 210, 211,
        211, 212, 212, 213, 214, 214, 215, 215, 216, 217,
        217, 218, 218, 219, 219, 220, 221, 221, 222, 222,
        223, 223, 224, 225, 225, 226, 226, 227, 227, 228,
        229, 229, 230, 230, 231, 231, 232, 232, 233, 234,
        234, 235, 235, 236, 236, 237, 237, 238, 238, 239,
        239, 240, 241, 241, 242, 242, 243, 243, 244, 244,
        245, 245, 246, 246, 247, 247, 248, 248, 249, 249,
        250, 250, 251, 251, 252, 252, 253, 253, 254, 254,
        255, 255,
    };

    if (x == 0)
        return 0;

    int lz = clz(x) & 62;
    x <<= lz;
    uint32_t y = isqrt64_tab[(x >> 56) - 64];
    y = (y << 7) + (x >> 41) / y;
    y = (y << 15) + (x >> 17) / y;
    y -= x < (uint64_t)y * y;
    return y >> (lz >> 1);
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
// f takes values in [0, 11], x and y lie in [0, nside).
typedef struct {
    int64_t x, y;
    int32_t f;
} t_hpd;


static int64_t hpd2nest(int64_t nside, t_hpd hpd) {
    return (hpd.f*nside*nside) + spread_bits(hpd.x) + (spread_bits(hpd.y)<<1);
}


static t_hpd nest2hpd(int64_t nside, int64_t pix) {
    int64_t npface_=nside*nside, p2=pix&(npface_-1);
    return (t_hpd){compress_bits(p2), compress_bits(p2>>1), pix/npface_};
}


static int64_t hpd2ring(int64_t nside, t_hpd hpd) {
  int64_t nl4 = 4*nside;
  int64_t jr = (jrll[hpd.f]*nside) - hpd.x - hpd.y - 1;

  if (jr<nside)
    {
    int64_t jp = (jpll[hpd.f]*jr + hpd.x - hpd.y + 1) / 2;
    jp = (jp>nl4) ? jp-nl4 : ((jp<1) ? jp+nl4 : jp);
    return 2*jr*(jr-1) + jp - 1;
    }
  else if (jr > 3*nside)
    {
    jr = nl4-jr;
    int64_t jp = (jpll[hpd.f]*jr + hpd.x - hpd.y + 1) / 2;
    jp = (jp>nl4) ? jp-nl4 : ((jp<1) ? jp+nl4 : jp);
    return 12*nside*nside - 2*(jr+1)*jr + jp - 1;
    }
  else
    {
    int64_t jp = (jpll[hpd.f]*nside + hpd.x - hpd.y + 1 + ((jr-nside)&1)) / 2;
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
        hpd.x = (jm-ifm)*nside;
        hpd.y = (1+ifp - jp)*nside;
        hpd.f = (ifp==ifm) ? (ifp|4) : ((ifp<ifm) ? ifp : (ifm+8));
        if (u) {
            *u = (jm-ifm)*nside - hpd.x;
            *v = (1+ifp - jp)*nside - hpd.y;
        }
    } else {
        // polar region
        int64_t ntt = (int64_t)tt;
        if (ntt >= 4)
            ntt = 3;
        double tp = tt - ntt;       // [0;1)
        /* double tmp = sqrt(3.*(1.-za)); */
        double tmp = loc.s/sqrt((1.+za)*(1./3.)); // FIXME optimize!

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
        hpd.x = jp*nside;
        hpd.y = jm*nside;
        hpd.f = ntt;
        if (u) {
            *u = jp*nside - hpd.x;
            *v = jm*nside - hpd.y;
        }
    }
    return hpd;
}


static t_loc hpd2loc(int64_t nside, t_hpd hpd, double u, double v) {
    double z, s, phi;
    const double x = (hpd.x+u)/nside;
    const double y = (hpd.y+v)/nside;
    const int32_t r = 1 - hpd.f/4;
    const double h = r-1 + (x+y);
    double m = 2 - r*h;
    if (m < 1.) {
        // polar cap
        double tmp = m*m*(1./3.);
        z = r*(1. - tmp);
        s = sqrt(tmp*(2.-tmp));
        phi = (PI/4)*(jpll[hpd.f] + (x-y)/m);
    } else {
        // equatorial region
        z = h*(2./3.);
        s = sqrt((1.+z)*(1.-z));
        phi = (PI/4)*(jpll[hpd.f] + (x-y));
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
    return loc2ang(hpd2loc(nside, ring2hpd(nside, ipix), 0.5, 0.5));
}


t_ang nest2ang(int64_t nside, int64_t ipix) {
    return loc2ang(hpd2loc(nside, nest2hpd(nside, ipix), 0.5, 0.5));
}


t_vec ring2vec(int64_t nside, int64_t ipix) {
    return loc2vec(hpd2loc(nside, ring2hpd(nside, ipix), 0.5, 0.5));
}


t_vec nest2vec(int64_t nside, int64_t ipix) {
    return loc2vec(hpd2loc(nside, nest2hpd(nside, ipix), 0.5, 0.5));
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
    return loc2ang(hpd2loc(nside, ring2hpd(nside, ipix), u, v));
}


t_ang nest2ang_uv(int64_t nside, int64_t ipix, double u, double v) {
    return loc2ang(hpd2loc(nside, nest2hpd(nside, ipix), u, v));
}


t_vec ring2vec_uv(int64_t nside, int64_t ipix, double u, double v) {
    return loc2vec(hpd2loc(nside, ring2hpd(nside, ipix), u, v));
}


t_vec nest2vec_uv(int64_t nside, int64_t ipix, double u, double v) {
    return loc2vec(hpd2loc(nside, nest2hpd(nside, ipix), u, v));
}


// conversions from or to UNIQ pixel scheme


t_pix uniq2nest(int64_t uniq) {
    if (uniq < 4) {
        return (t_pix){-1, -1};
    } else {
        int order = ilog2(uniq)/2 - 1;
        return (t_pix){1ll << order, uniq - 4*(1ll << 2*order)};
    }
}


t_pix uniq2ring(int64_t uniq) {
    t_pix pix = uniq2nest(uniq);
    pix.ipix = nest2ring(pix.nside, pix.ipix);
    return pix;
}


int64_t nest2uniq(int64_t nside, int64_t ipix) {
    if (nside < 0 || ipix < 0) {
        return -1;
    } else {
        return 4*nside*nside + ipix;
    }
}


int64_t ring2uniq(int64_t nside, int64_t ipix) {
    return nest2uniq(nside, ring2nest(nside, ipix));
}
