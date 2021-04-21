#include <rosacxx/core/pitch.h>
#include <rosacxx/core/convert.h>

#include <memory>
#include <cmath>

namespace rosacxx {
namespace core {

float estimate_tuning(
        const nc::NDArrayF32Ptr& y,
        const float& sr,
        const nc::NDArrayF32Ptr& S,
        const int& n_fft,
        const float& resolution,
        const int& bins_per_octave
        ) {
    nc::NDArrayF32Ptr frequencies = nullptr;
    return pitch_tuning(frequencies, resolution, bins_per_octave);
}

float pitch_tuning(
        const nc::NDArrayF32Ptr& frequencies,
        const float& resolution,
        const int& bins_per_octave
        ) {

    std::vector<int> indies_larger_than_zero(0);
    float * ptr_freq = frequencies->data();
    for (int i = 0; i < frequencies->elemCount(); i++) {
        if (ptr_freq[i] > 0) {
            indies_larger_than_zero.push_back(i);
        }
    }

    if (indies_larger_than_zero.size() == 0) {
        printf("[Wraning] Trying to estimate tuning from empty frequency set.");
        return 0.0f;
        throw std::runtime_error("Trying to estimate tuning from empty frequency set.");
    }

    nc::NDArrayF32Ptr newFreq = std::shared_ptr<nc::NDArrayF32>(new nc::NDArrayF32({int(indies_larger_than_zero.size())}));
    float * ptr_newfreq = newFreq->data();
    for (auto i = 0; i < indies_larger_than_zero.size(); i++) {
        ptr_newfreq[i] = ptr_freq[indies_larger_than_zero[i]];
    }

    nc::NDArrayF32Ptr octs = hz_to_octs(newFreq);
    float * ptr_octs = octs->data();

    nc::NDArrayF32Ptr residual = std::shared_ptr<nc::NDArrayF32>(new nc::NDArrayF32(octs->shape()));
    float * ptr_residual = residual->data();
    for (auto i = 0; i < octs->elemCount(); i++) {
        ptr_residual[i] = bins_per_octave * ptr_octs[i];
        ptr_residual[i] = ptr_residual[i] - std::round(ptr_residual[i]);
        if (ptr_residual[i] >= .5f) {
            ptr_residual[i] -= 1.f;
        }
    }

    auto bins = nc::linspace<float>(-0.5f, 0.5f, int(std::ceil(1.0 / resolution)) + 1);
    auto counts = nc::histogram(residual, bins);
    return bins->getitem(counts->argmax());
}

} // namespace core
} // namespace rosacxx
