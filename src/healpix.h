// license: BSD-3-Clause

#ifndef HEALPIX_H_
#define HEALPIX_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* This code is written in C99; you may have to adjust your compiler flags. */


// Continuous coordinate systems
//
// theta is the co-latitude in radians (0 at the north Pole, increasing to pi
// at the south pole)
//
// phi is the azimuth in radians
//
// admissible values for theta: 0 <= theta <= pi
//
// admissible values for phi: in principle unconstrained, but best accuracy is
// obtained for -2pi <= phi <= 2pi


// A structure describing a location on the sphere.
typedef struct {
    double theta;
    double phi;
} t_ang;


// A structure describing a 3-vector with coordinates x, y and z.
typedef struct {
    double x;
    double y;
    double z;
} t_vec;


// Returns a normalized 3-vector pointing in the same direction as ang.
t_vec ang2vec(t_ang ang);


// Returns a t_ang describing the same direction as the 3-vector vec.
// vec need not be normalized
t_ang vec2ang(t_vec vec);


/* Discrete coordinate systems */

/* Admissible values for nside parameters:
   any integer power of 2 with 1 <= nside <= 1<<29

   Admissible values for pixel indices:
   0 <= idx < 12*nside*nside */

extern const int64_t NSIDE_MAX;  // 1<<29

/*! Returns the RING pixel index of pixel \a ipnest at resolution \a nside.
    On error, returns -1. */
int64_t nest2ring(int64_t nside, int64_t ipnest);
/*! Returns the NEST pixel index of pixel \a ipring at resolution \a nside.
    On error, returns -1. */
int64_t ring2nest(int64_t nside, int64_t ipring);


// Conversions between continuous and discrete coordinate systems


// Returns the pixel number in NEST scheme at resolution nside, which contains
// the position ang.
int64_t ang2nest(int64_t nside, t_ang ang);


// Returns the pixel number in RING scheme at resolution nside, which contains
// the position ang.
int64_t ang2ring(int64_t nside, t_ang ang);


// Returns a t_ang corresponding to the angular position of the center of pixel
// ipix in NEST scheme at resolution nside.
t_ang nest2ang(int64_t nside, int64_t ipix);


// Returns a t_ang corresponding to the angular position of the center of pixel
// ipix in RING scheme at resolution nside.
t_ang ring2ang(int64_t nside, int64_t ipix);


// Returns the pixel number in NEST scheme at resolution nside, which contains
// the direction described by the 3-vector vec.
int64_t vec2nest(int64_t nside, t_vec vec);


// Returns the pixel number in RING scheme at resolution nside, which contains
// the direction described by the 3-vector vec.
int64_t vec2ring(int64_t nside, t_vec vec);


// Returns a normalized 3-vector pointing in the direction of the center of
// pixel ipix in NEST scheme at resolution nside.
t_vec nest2vec(int64_t nside, int64_t ipix);


// Returns a normalized 3-vector pointing in the direction of the center of
// pixel ipix in RING scheme at resolution nside.
t_vec ring2vec(int64_t nside, int64_t ipix);


// Miscellaneous utility routines


// Returns 12*nside*nside.
int64_t nside2npix(int64_t nside);


// Returns sqrt(npix/12) if this is an integer number, otherwise -1.
int64_t npix2nside(int64_t npix);


// Returns the angle (in radians) between the vectors v1 and v2.
// The result is accurate even for angles close to 0 and pi.
double vec_angle(t_vec v1, t_vec v2);


// conversions with sub-pixel positions


// Variant of ang2nest that also returns the sub-pixel position in u and v.
int64_t ang2nest_uv(int64_t nside, t_ang ang, double* u, double* v);


// Variant of ang2ring that also returns the sub-pixel position in u and v.
int64_t ang2ring_uv(int64_t nside, t_ang ang, double* u, double* v);


// Variant of nest2ang that also takes the sub-pixel position in u and v.
t_ang nest2ang_uv(int64_t nside, int64_t ipix, double u, double v);


// Variant of ring2ang that also takes the sub-pixel position in u and v.
t_ang ring2ang_uv(int64_t nside, int64_t ipix, double u, double v);


// Variant of vec2nest that also returns the sub-pixel position in u and v.
int64_t vec2nest_uv(int64_t nside, t_vec vec, double* u, double* v);


// Variant of vec2ring that also returns the sub-pixel position in u and v.
int64_t vec2ring_uv(int64_t nside, t_vec vec, double* u, double* v);


// Variant of nest2vec that also takes the sub-pixel position in u and v.
t_vec nest2vec_uv(int64_t nside, int64_t ipix, double u, double v);


// Variant of ring2vec that also takes the sub-pixel position in u and v.
t_vec ring2vec_uv(int64_t nside, int64_t ipix, double u, double v);


// Conversions to UNIQ pixel index scheme


// Describe a pixel index in RING or NEST scheme and its nside parameter
typedef struct {
    int64_t nside;
    int64_t ipix;
} t_pix;


// Convert from UNIQ to NEST scheme and nside parameter
t_pix uniq2nest(int64_t uniq);


// Convert from UNIQ to RING scheme and nside parameter
t_pix uniq2ring(int64_t uniq);


// Convert from NEST scheme and nside parameter to UNIQ
int64_t nest2uniq(int64_t nside, int64_t ipix);


// Convert from RING scheme and nside parameter to UNIQ
int64_t ring2uniq(int64_t nside, int64_t ipix);


#ifdef __cplusplus
} // extern "C"
#endif

#endif  // HEALPIX_H_
