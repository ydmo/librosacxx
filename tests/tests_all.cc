#include <iostream>

namespace rosa {
namespace tests {

// declare test functions here ...
void tests_fft();
void tests_chroma_cqt();
void tests_numcxx();

void tests_all() {
    // tests_fft();
    tests_numcxx();
}

} // namespace tests
} // namespace rosa

int main() {
    std::cout << "[LibRosaCXX][Tests] Start. " << std::endl;
    rosa::tests::tests_all();
    std::cout << "[LibRosaCXX][Tests] End. " << std::endl;
}
