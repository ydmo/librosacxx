#include "tests_all.h"

#include <iostream>

namespace rosa {
namespace tests {

void tests_all() {

}

} // namespace tests
} // namespace rosa

int main() {
    std::cout << "[LibRosaCXX][Test] Start. " << std::endl;
    rosa::tests::tests_all();
    std::cout << "[LibRosaCXX][Test] End. " << std::endl;
}
