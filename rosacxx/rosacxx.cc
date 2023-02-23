#include <rosacxx/rosacxx.h>
#include <ctime>

#ifdef _WIN32
#   include <WTypesbase.h>
#   include <algorithm>
#   undef max
#   undef min
#endif // _WIN32

namespace rosacxx {

double getTickFrequency(void) {
#   ifdef _WIN32
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return (double)freq.QuadPart;
#   elif __linux__
    return 1e9;
#   elif __APPLE__
    static double freq = 0;
    if( freq == 0 ) {
        mach_timebase_info_data_t sTimebaseInfo;
        mach_timebase_info(&sTimebaseInfo);
        freq = sTimebaseInfo.denom*1e9/sTimebaseInfo.numer;
    }
    return freq;
#   else
    return 1e6;
#   endif
}

double getTickCount(void) {
#   ifdef _WIN32
    LARGE_INTEGER counter;
    QueryPerformanceCounter( &counter );
    return (double)counter.QuadPart;
#   elif __linux__
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (double)tp.tv_sec*1000000000 + tp.tv_nsec;
#   elif __APPLE__
    return static_cast<double>(mach_absolute_time());
#   else
    struct timeval tv;
    struct timezone tz;
    gettimeofday( &tv, &tz );
    return (double)tv.tv_sec*1000000 + tv.tv_usec;
#   endif
}

} // namespace rosacxx
