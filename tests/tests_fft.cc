#include <iostream>
#include <3rd/fft/fft.h>
#include <gtest/gtest.h>

class FFTTest : public testing::Test {
protected:
    virtual void TearDown() override { }

    virtual void SetUp() override {

    }
};

TEST_F(FFTTest, Case0) {
    EXPECT_EQ(1, 1);
}

namespace rosa {
namespace tests {

void tests_fft() {
    ::testing::InitGoogleTest();
    ::testing::GTEST_FLAG(filter) = "FFTTest*";
    RUN_ALL_TESTS();
}

} // namespace tests
} // namespace rosa

