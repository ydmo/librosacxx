#include <iostream>
#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
    std::cout << "[##########][LibRosaCXX][Tests] Start. " << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
    std::cout << "[##########][LibRosaCXX][Tests] End. " << std::endl;
    return 0;
}
