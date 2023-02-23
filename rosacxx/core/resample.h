#ifndef ROSACXX_RESAMPLE_H
#define ROSACXX_RESAMPLE_H

#include <rosacxx/numcxx/ndarray.h>

namespace rosacxx {
namespace core {

nc::NDArrayF32Ptr resample(x, sr_orig, sr_new, axis=-1, filter='kaiser_best', **kwargs)

} // namespace core
} // namespace rosacxx

#endif /* ROSACXX_RESAMPLE_H */
