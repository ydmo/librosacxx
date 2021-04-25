#include <iostream>
#include <gtest/gtest.h>
#include <rosacxx/core/fft.h>

class FFTTest : public testing::Test {
protected:
    virtual void TearDown() override { }

    virtual void SetUp() override {

    }
};

TEST_F(FFTTest, rfft) {
    auto fft_ri = nc::arange(64.f);
    std::cout << "fft input is " << fft_ri << std::endl;
    auto fft_co = rosacxx::core::rfft(fft_ri, 64);
    std::cout << "fft output is " << fft_co << std::endl;
    std::vector<float> fft_co_gt = {
        2016.  ,0.        ,  -32.,651.374964  ,  -32.,324.9014524 ,
        -32.,215.72647697,  -32.,160.87486375,  -32.,127.75116108,
        -32.,105.48986269,  -32. ,89.43400872,  -32. ,77.254834  ,
        -32. ,67.65831544,  -32. ,59.86778918,  -32. ,53.38877458,
        -32. ,47.89138441,  -32. ,43.14700523,  -32. ,38.99211282,
        -32. ,35.30655922,  -32. ,32.        ,  -32. ,29.00310941,
        -32. ,26.26172131,  -32. ,23.73281748,  -32. ,21.38171641,
        -32. ,19.18006188,  -32. ,17.10435635,  -32. ,15.13487283,
        -32. ,13.254834  ,  -32. ,11.44978308,  -32.  ,9.70709388,
        -32.  ,8.01558273,  -32.  ,6.36519576,  -32.  ,4.7467516 ,
        -32.  ,3.15172491,  -32.  ,1.57205919,  -32.  ,0.
    };
    for (auto i = 0; i < fft_co.elemCount(); i++) {
        EXPECT_NEAR(fft_co_gt[i * 2 + 0], fft_co.getitem(i).r, 1e-4);
        EXPECT_NEAR(fft_co_gt[i * 2 + 1], fft_co.getitem(i).i, 1e-4);
    }
}

