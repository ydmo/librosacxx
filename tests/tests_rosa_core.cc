#include <iostream>
#include <gtest/gtest.h>

#include <rosacxx/core/pitch.h>
#include <rosacxx/core/convert.h>

class ROSACORETest : public testing::Test {
protected:
    virtual void TearDown() override { }
    virtual void SetUp() override { }
};

TEST_F(ROSACORETest, midi_to_hz) {
    std::vector<double> gt = {
        8.35482560e+00, 8.85162938e+00, 9.37797466e+00, 9.93561805e+00,
        1.05264206e+01, 1.11523542e+01, 1.18155077e+01, 1.25180943e+01,
        1.32624589e+01, 1.40510858e+01, 1.48866068e+01, 1.57718105e+01,
        1.67096512e+01, 1.77032588e+01, 1.87559493e+01, 1.98712361e+01,
        2.10528413e+01, 2.23047084e+01, 2.36310153e+01, 2.50361886e+01,
        2.65249179e+01, 2.81021716e+01, 2.97732137e+01, 3.15436211e+01,
        3.34193024e+01, 3.54065175e+01, 3.75118986e+01, 3.97424722e+01,
        4.21056826e+01, 4.46094167e+01, 4.72620307e+01, 5.00723773e+01,
        5.30498358e+01, 5.62043432e+01, 5.95464273e+01, 6.30872422e+01,
        6.68386048e+01, 7.08130351e+01, 7.50237972e+01, 7.94849444e+01,
        8.42113651e+01, 8.92188335e+01, 9.45240614e+01, 1.00144755e+02,
        1.06099672e+02, 1.12408686e+02, 1.19092855e+02, 1.26174484e+02,
        1.33677210e+02, 1.41626070e+02, 1.50047594e+02, 1.58969889e+02,
        1.68422730e+02, 1.78437667e+02, 1.89048123e+02, 2.00289509e+02,
        2.12199343e+02, 2.24817373e+02, 2.38185709e+02, 2.52348969e+02,
        2.67354419e+02, 2.83252140e+02, 3.00095189e+02, 3.17939778e+02,
        3.36845461e+02, 3.56875334e+02, 3.78096246e+02, 4.00579018e+02,
        4.24398686e+02, 4.49634745e+02, 4.76371419e+02, 5.04697937e+02,
        5.34708838e+02, 5.66504281e+02, 6.00190378e+02, 6.35879555e+02,
        6.73690921e+02, 7.13750668e+02, 7.56192491e+02, 8.01158037e+02,
        8.48797373e+02, 8.99269491e+02, 9.52742837e+02, 1.00939587e+03,
        1.06941768e+03, 1.13300856e+03, 1.20038076e+03, 1.27175911e+03,
        1.34738184e+03, 1.42750134e+03, 1.51238498e+03, 1.60231607e+03,
        1.69759475e+03, 1.79853898e+03, 1.90548567e+03, 2.01879175e+03,
        2.13883535e+03, 2.26601712e+03, 2.40076151e+03, 2.54351822e+03,
        2.69476368e+03, 2.85500267e+03, 3.02476996e+03, 3.20463215e+03,
        3.39518949e+03, 3.59707796e+03, 3.81097135e+03, 4.03758350e+03,
        4.27767071e+03, 4.53203424e+03, 4.80152302e+03, 5.08703644e+03,
        5.38952737e+03, 5.71000534e+03, 6.04953993e+03, 6.40926429e+03,
        6.79037898e+03, 7.19415593e+03, 7.62194270e+03, 8.07516700e+03,
        8.55534141e+03, 9.06406849e+03, 9.60304605e+03, 1.01740729e+04,
        1.07790547e+04, 1.14200107e+04, 1.20990799e+04, 1.28185286e+04
    };
    auto midi = rosacxx::core::midi_to_hz(nc::arange(128)->add(0.375f));
    for (auto i = 0; i < midi->elemCount(); i++) {
        EXPECT_LE(std::abs((midi->getitem(i)-gt[i])/gt[i]), 1e-6);
    }
}

TEST_F(ROSACORETest, pitch_tuning) {
    const std::vector<float> vec_resolution = { 1e-2, 1e-3 };
    const std::vector<float> vec_tuning = { -0.5, -0.375, -0.25, 0.0, 0.25, 0.375 };
    const std::vector<int  > vec_bins_per_octave = { 12, };
    for (auto resolution : vec_resolution) {
        for (auto tuning : vec_tuning) {
            for (auto bins_per_octave : vec_bins_per_octave) {
                auto hz = rosacxx::core::midi_to_hz(nc::arange(128)->add(tuning));
                auto est_tuning = rosacxx::core::pitch_tuning(hz, resolution, bins_per_octave);
                EXPECT_LE(std::abs(tuning - est_tuning), resolution);
            }
        }
    }
}

namespace rosa {
namespace tests {

void tests_rosa_core() {
    ::testing::InitGoogleTest();
    ::testing::GTEST_FLAG(filter) = "ROSACORETest*";
    RUN_ALL_TESTS();
}

} // namespace tests
} // namespace rosa

