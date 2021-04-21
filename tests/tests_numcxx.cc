#include <iostream>
#include <gtest/gtest.h>
#include <3rd/numcxx/numcxx.h>

class NCTest : public testing::Test {
protected:
    virtual void TearDown() override { }

    virtual void SetUp() override {

    }
};

TEST_F(NCTest, show) {
    auto arr0 = nc::NDArray<float>::FromVec1D({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    std::cout << "arr0: " << arr0 << std::endl;
}

TEST_F(NCTest, argmax) {

}

namespace rosa {
namespace tests {

void tests_numcxx() {
    ::testing::InitGoogleTest();
    ::testing::GTEST_FLAG(filter) = "NCTest*";
    RUN_ALL_TESTS();
}

} // namespace tests
} // namespace rosa

