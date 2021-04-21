#include <iostream>
#include <gtest/gtest.h>

namespace rosa {
namespace tests {

// declare test functions here ...
void tests_numcxx();
void tests_rosa_core();

void tests_all() {
    tests_numcxx();
    tests_rosa_core();
}

} // namespace tests
} // namespace rosa

int main(int argc, char *argv[]) {
    std::cout << "[LibRosaCXX][Tests] Start. " << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
    std::cout << "[LibRosaCXX][Tests] End. " << std::endl;
    return 0;
}
