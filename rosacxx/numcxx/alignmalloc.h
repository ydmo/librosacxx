#ifndef ALIGNMALLOC_H
#define ALIGNMALLOC_H

#include <stdio.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <cmath>
#include <cassert>

namespace nc {

inline size_t alignUp(const size_t& __size, const size_t& __align) {
    const size_t alignment_mask = __align - 1;
    return ((__size + alignment_mask) & (~alignment_mask));
}

inline void * alignedMalloc(size_t __alignment, size_t __size) {
#   ifdef _WIN32
    return _aligned_malloc(alignUp(__size, __alignment), __alignment);
#   else
    void *p = NULL;
    if (__alignment) {
        int ret = posix_memalign(&p, __alignment, alignUp(__size, __alignment));
        if(ret) { //  failed.
            throw std::runtime_error("posix_memalign failed.");
            return NULL;
        }
    } else {
        p = malloc(__size);
    }
    return p;
#   endif
}

inline void * alignedCalloc(size_t __alignment, size_t __size) {
#   ifdef _WIN32
    size_t alignedSize = alignUp(__size, __alignment);
    void * ptr = _aligned_malloc(alignedSize, __alignment);
    memset(ptr, 0x00, alignedSize);
    return ptr;
#   else
    void *p = NULL;
    if (__alignment) {
        size_t alignedSize = alignUp(__size, __alignment);
        int ret = posix_memalign(&p, __alignment, alignedSize);
        memset(p, 0x00, alignedSize);
        if(ret) { //  failed.
            throw std::runtime_error("posix_memalign failed.");
            return NULL;
        }
    } else {
        p = calloc(__size, 1);
    }
    return p;
#   endif
}

inline void alignedFree(void * __p) {
#   ifdef _WIN32
    _aligned_free(__p);
#   else
    free(__p);
#   endif
}

} // namespace nc

#endif // ALIGNMALLOC_H
