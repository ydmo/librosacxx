#ifndef VAMP_KISS_FFT_GUTS_H
#define VAMP_KISS_FFT_GUTS_H

#include "vamp_kiss_fft_scalar.h"

#define S_MUL(a,b) ( (a)*(b) )

#define C_MUL(m,a,b) do{ (m).r = (a).r*(b).r - (a).i*(b).i; (m).i = (a).r*(b).i + (a).i*(b).r; } while(0)

#define C_FIXDIV(c,div) /* NOOP */

#define C_MULBYSCALAR( c, s ) do{ (c).r *= (s); (c).i *= (s); } while(0)

#define CHECK_OVERFLOW_OP(a,op,b) /* noop */

#define  C_ADD( res, a,b) do { CHECK_OVERFLOW_OP((a).r,+,(b).r); CHECK_OVERFLOW_OP((a).i,+,(b).i); (res).r=(a).r+(b).r;  (res).i=(a).i+(b).i; } while(0)

#define  C_SUB( res, a,b) do { CHECK_OVERFLOW_OP((a).r,-,(b).r); CHECK_OVERFLOW_OP((a).i,-,(b).i); (res).r=(a).r-(b).r;  (res).i=(a).i-(b).i; } while(0)

#define C_ADDTO( res , a) do { CHECK_OVERFLOW_OP((res).r,+,(a).r); CHECK_OVERFLOW_OP((res).i,+,(a).i); (res).r += (a).r;  (res).i += (a).i; } while(0)

#define C_SUBFROM( res , a) do { CHECK_OVERFLOW_OP((res).r,-,(a).r); CHECK_OVERFLOW_OP((res).i,-,(a).i); (res).r -= (a).r;  (res).i -= (a).i; } while(0)

#define VAMP_KISS_FFT_COS(phase) (vamp_kiss_fft_scalar) cos(phase)

#define VAMP_KISS_FFT_SIN(phase) (vamp_kiss_fft_scalar) sin(phase)

#define HALF_OF(x) ((x)*.5)

#define kf_cexp(x,phase) do{ (x)->r = VAMP_KISS_FFT_COS(phase); (x)->i = VAMP_KISS_FFT_SIN(phase); } while(0)

#endif // VAMP_KISS_FFT_GUTS_H
