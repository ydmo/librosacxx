#ifndef ROSACXX_ROSACXX_H
#define ROSACXX_ROSACXX_H

#ifdef _WIN32
#   include <immintrin.h>
#   include <emmintrin.h>
#	ifdef _WIN64
#       pragma mark define something for Windows (64-bit only)
#	endif /* _WIN64 */
#   ifndef __SSE__
#       define __SSE__ 1
#   endif /* __SSE__ */
#   ifndef __AVX__
#       define __AVX__ 1
#   endif /* __AVX__ */
#   if __SSE__
#       define __X86_SSE 1
#   endif // __SSE__
#   if __AVX__
#       define __X86_AVX 1
#   endif // __AVX__
#elif __APPLE__
#    include <Availability.h>
#    include <AvailabilityMacros.h>
#    include <TargetConditionals.h>
#    include <mach/mach_time.h>
#    include <sys/time.h>
#    if TARGET_IPHONE_SIMULATOR
#        define float32_t float
#    elif TARGET_OS_IPHONE
#       include <arm_neon.h>
#       ifdef __LP64__
#           pragma mark define something for ios (64-bit only)
#       else // _LP64
#           pragma mark define something for ios (32-bit only)
#       endif // __LP64__
#    elif TARGET_OS_MAC
#       if TARGET_CPU_ARM64 // Code meant for the arm64 architecture here.
#           include <arm_neon.h>
#           define ASMBLOCK asm volatile
#       elif TARGET_CPU_X86_64 // Code meant for the x86_64 architecture here.
#           include <immintrin.h>
#           include <emmintrin.h>
#           if __SSE__
#               define __X86_SSE 1
#           endif
#           if __AVX__
#               define __X86_AVX 1
#           endif
#       endif
#       ifdef _LP64
#           pragma mark define something for macOS (64-bit only)
#       else // _LP64
#           pragma mark define something for macOS (32-bit only)
#       endif // _LP64
#   else // TARGET_IPHONE_SIMULATOR | TARGET_OS_IPHONE | TARGET_OS_MAC
#       error Unsupported apple platform
#   endif // TARGET_IPHONE_SIMULATOR | TARGET_OS_IPHONE | TARGET_OS_MAC
#elif __ANDROID__
#   error unsupported platform
#elif __linux__
#   include <immintrin.h>
#   include <emmintrin.h>
#else /* _WIN32 | __APPLE__ | __ANDROID__ | __linux__ */
#	error unsupported platform
#endif /* _WIN32 | __APPLE__ | __ANDROID__ | __linux__ */

#include <cfloat>
#include <memory>
#include <algorithm>

namespace rosacxx {

double getTickFrequency(void);
double getTickCount(void);

#if 0
#   define LOGTIC(event) double tic_##event = rosacxx::getTickCount()
#   define LOGTOC(event) printf("[ROSACXX] Event[%s] cost %f ms\n", #event, ( rosacxx::getTickCount() - tic_##event ) / rosacxx::getTickFrequency() * 1e3)
#else
#   define LOGTIC(event)
#   define LOGTOC(event)
#endif

class TimeMetrics {
public:
    typedef std::shared_ptr<TimeMetrics> Ptr;
    TimeMetrics() = default;
    ~TimeMetrics() = default;

    void clear() {
        _tic = 0;
        _sum = 0;
        _cnt = 0;
        _min = FLT_MAX;
        _max = FLT_MIN;
    }

    double sum() const {
        return _sum;
    }

    size_t cnt() const {
        return _cnt;
    }

    double max() const {
        return _max;
    }

    double min() const {
        return _min;
    }

    double avg() const {
        return _sum / _cnt;
    }

    double feq() const {
        return getTickFrequency();
    }

    double tic() {
        _tic = getTickCount();
        return _tic;
    }

    double toc() {
        double toc = getTickCount();
        double elapse = (toc - _tic) / getTickFrequency();
        _min = std::min(_min, elapse);
        _max = std::max(_max, elapse);
        _sum += elapse;
        _cnt += 1;
        return toc;
    }

public:
    double _min = FLT_MAX;
    double _max = FLT_MIN;
    double _sum = 0;
    size_t _cnt = 0;

protected:
    double _tic = 0;
};

} // namespace rosacxx

#endif // ROSACXX_ROSACXX_H
