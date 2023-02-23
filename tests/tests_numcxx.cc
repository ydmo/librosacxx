#define ENABLE_NC_TESTS 1
#if ENABLE_NC_TESTS

#include <iostream>
#include <gtest/gtest.h>
#include <rosacxx/numcxx/numcxx.h>
#include <rosacxx/numcxx/pad.h>

class NCTest : public testing::Test {
protected:
    virtual void TearDown() override { }

    virtual void SetUp() override {

    }
};

TEST_F(NCTest, FromScalar) {
    auto scalarx = nc::NDArrayPtr<float>::FromScalar(5.f);
    EXPECT_NEAR(scalarx.getitem(0), 5.f, 1e-9);
}

TEST_F(NCTest, FromVec1D) {

    std::vector<float> vecf1d = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    auto arr0 = nc::NDArrayPtr<float>::FromVec1D(vecf1d);
    for (auto i = 0; i < vecf1d.size(); i++) {
        EXPECT_NEAR(arr0.getitem(i), vecf1d[i], 1e-6);
    }

    std::vector<int> vecs1d = {0, 2, 4, 6, 8};
    auto arr1 = nc::NDArrayPtr<int>::FromVec1D(vecs1d);
    for (auto i = 0; i < vecs1d.size(); i++) {
        EXPECT_NEAR(arr1.getitem(i), vecs1d[i], 1e-6);
    }

}

TEST_F(NCTest, FromVec2D) {
    std::vector<std::vector<float>> vec2d = {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}};
    auto arr0 = nc::NDArrayPtr<float>::FromVec2D(vec2d);
    for (auto i = 0; i < vec2d.size(); i++) {
        for (auto j = 0; j < vec2d[0].size(); j++) {
            EXPECT_NEAR(arr0.getitem(i, j), vec2d[i][j], 1e-6);
        }
    }
}

TEST_F(NCTest, Add) {
    {
        std::vector<std::vector<float>> vec2d = {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}};
        auto arr0 = nc::NDArrayPtr<float>::FromVec2D(vec2d);
        auto arr1 = arr0 + 1.f;
        for (auto i = 0; i < vec2d.size(); i++) {
            for (auto j = 0; j < vec2d[0].size(); j++) {
                EXPECT_NEAR(arr1.getitem(i, j), vec2d[i][j] + 1.f, 1e-6);
            }
        }
    }
}

TEST_F(NCTest, linspace) {
    auto arr0 = nc::linspace(-0.5f, 0.5f, 101);
    std::vector<float> gt0 = {-0.5 , -0.49, -0.48, -0.47, -0.46, -0.45, -0.44, -0.43, -0.42,
                             -0.41, -0.4 , -0.39, -0.38, -0.37, -0.36, -0.35, -0.34, -0.33,
                             -0.32, -0.31, -0.3 , -0.29, -0.28, -0.27, -0.26, -0.25, -0.24,
                             -0.23, -0.22, -0.21, -0.2 , -0.19, -0.18, -0.17, -0.16, -0.15,
                             -0.14, -0.13, -0.12, -0.11, -0.1 , -0.09, -0.08, -0.07, -0.06,
                             -0.05, -0.04, -0.03, -0.02, -0.01,  0.  ,  0.01,  0.02,  0.03,
                              0.04,  0.05,  0.06,  0.07,  0.08,  0.09,  0.1 ,  0.11,  0.12,
                              0.13,  0.14,  0.15,  0.16,  0.17,  0.18,  0.19,  0.2 ,  0.21,
                              0.22,  0.23,  0.24,  0.25,  0.26,  0.27,  0.28,  0.29,  0.3 ,
                              0.31,  0.32,  0.33,  0.34,  0.35,  0.36,  0.37,  0.38,  0.39,
                              0.4 ,  0.41,  0.42,  0.43,  0.44,  0.45,  0.46,  0.47,  0.48,
                              0.49,  0.5};
    for (auto i = 0; i < gt0.size(); i++) {
        EXPECT_NEAR(arr0.getitem(i), gt0[i], 1e-6);
    }

    auto arr1 = nc::linspace(-0.5f, 0.5f, 101, false);
    std::vector<float> gt1 = {-0.5       , -0.49009901, -0.48019802, -0.47029703, -0.46039604,
                              -0.45049505, -0.44059406, -0.43069307, -0.42079208, -0.41089109,
                              -0.4009901 , -0.39108911, -0.38118812, -0.37128713, -0.36138614,
                              -0.35148515, -0.34158416, -0.33168317, -0.32178218, -0.31188119,
                              -0.3019802 , -0.29207921, -0.28217822, -0.27227723, -0.26237624,
                              -0.25247525, -0.24257426, -0.23267327, -0.22277228, -0.21287129,
                              -0.2029703 , -0.19306931, -0.18316832, -0.17326733, -0.16336634,
                              -0.15346535, -0.14356436, -0.13366337, -0.12376238, -0.11386139,
                              -0.1039604 , -0.09405941, -0.08415842, -0.07425743, -0.06435644,
                              -0.05445545, -0.04455446, -0.03465347, -0.02475248, -0.01485149,
                              -0.0049505 ,  0.0049505 ,  0.01485149,  0.02475248,  0.03465347,
                               0.04455446,  0.05445545,  0.06435644,  0.07425743,  0.08415842,
                               0.09405941,  0.1039604 ,  0.11386139,  0.12376238,  0.13366337,
                               0.14356436,  0.15346535,  0.16336634,  0.17326733,  0.18316832,
                               0.19306931,  0.2029703 ,  0.21287129,  0.22277228,  0.23267327,
                               0.24257426,  0.25247525,  0.26237624,  0.27227723,  0.28217822,
                               0.29207921,  0.3019802 ,  0.31188119,  0.32178218,  0.33168317,
                               0.34158416,  0.35148515,  0.36138614,  0.37128713,  0.38118812,
                               0.39108911,  0.4009901 ,  0.41089109,  0.42079208,  0.43069307,
                               0.44059406,  0.45049505,  0.46039604,  0.47029703,  0.48019802,
                               0.49009901};
    for (auto i = 0; i < gt1.size(); i++) {
        EXPECT_NEAR(arr1.getitem(i), gt1[i], 1e-6);
    }
}

TEST_F(NCTest, histogram) {
    //    np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    //    (array([0, 2, 1]), array([0, 1, 2, 3]))
    auto arr_src0 = nc::NDArrayPtr<int>::FromVec1D({1, 2, 1});
    auto arr_bins0 = nc::NDArrayPtr<int>::FromVec1D({0, 1, 2, 3});
    auto arr_hist0 = nc::histogram(arr_src0, arr_bins0);
    EXPECT_EQ(arr_hist0.getitem(0), 0);
    EXPECT_EQ(arr_hist0.getitem(1), 2);
    EXPECT_EQ(arr_hist0.getitem(2), 1);

    //    np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    //    (array([1, 4, 1]), array([0, 1, 2, 3]))
    auto arr_src1 = nc::NDArrayPtr<float>::FromVec2D({{1, 2, 1}, {1, 0, 1}});
    auto arr_bins1 = nc::NDArrayPtr<float>::FromVec1D({0, 1, 2, 3});
    auto arr_hist1 = nc::histogram(arr_src1, arr_bins1);
    EXPECT_EQ(arr_hist1.getitem(0), 1);
    EXPECT_EQ(arr_hist1.getitem(1), 4);
    EXPECT_EQ(arr_hist1.getitem(2), 1);
}

TEST_F(NCTest, arange) {
    auto range0 = nc::arange(4.5);
    for (auto i = 0; i < 5; i++) {
        EXPECT_NEAR(range0.getitem(i), i, 1e-6);
    }

    auto range1 = nc::arange(100);
    for (auto i = 0; i < 100; i++) {
        EXPECT_EQ(range1.getitem(i), i);
    }
}

TEST_F(NCTest, argmax) {
    // axis is < 0
    {
        std::vector<std::vector<float>> vec = {
            { 11, 12, 13 },
            { 14, 15, 16 },
        };
        auto arr = nc::NDArrayPtr<float>::FromVec2D(vec);
        auto am0 = nc::argmax(arr);
        EXPECT_EQ(am0.scalar(), 5);
    }

    // axis is >= 0
    {
        auto arr = nc::NDArrayPtr<float>::FromVec3D(
                    {
                        {
                            {-2.18736917,  0.32735344,  0.43235804, -0.2882413 },
                            { 1.62667071, -0.56718496,  1.1210251 , -1.17778305},
                            {-0.57653589,  0.50395653, -0.07219006, -0.97902915}},
                        {
                            {-0.46580697,  0.74178501, -0.84009043, -1.65368115},
                            {-0.32295189,  1.88334314, -1.05390129,  0.50312785},
                            { 0.75639164,  0.51680735,  1.39314147,  0.584609  }
                        }
                    }
                    );
        auto argmax0 = arr.argmax(0);
        auto argmax0_gt = nc::NDArrayPtr<int>::FromVec2D( {{1, 1, 0, 0}, {0, 1, 0, 1}, {1, 1, 1, 1}} );

        for (auto i = 0; i < argmax0.elemCount(); i++) {
            EXPECT_EQ(argmax0.getitem(i), argmax0_gt.getitem(i));
        }

        auto argmax1 = arr.argmax(1);
        auto argmax1_gt = nc::NDArrayPtr<int>::FromVec2D( {{1, 2, 1, 0}, {2, 1, 2, 2}} );
        for (auto i = 0; i < argmax1.elemCount(); i++) {
            EXPECT_EQ(argmax1.getitem(i), argmax1_gt.getitem(i));
        }

        auto argmax2 = arr.argmax(2);
        auto argmax2_gt = nc::NDArrayPtr<int>::FromVec2D( {{2, 0, 1}, {1, 1, 2}} );
        for (auto i = 0; i < argmax2.elemCount(); i++) {
            EXPECT_EQ(argmax2.getitem(i), argmax2_gt.getitem(i));
        }
    }
}

TEST_F(NCTest, argmin) {
    // axis is < 0
    {
        std::vector<std::vector<float>> vec = {
            { 11, 10, 13 },
            { 14, 15, 16 },
        };
        auto arr = nc::NDArrayPtr<float>::FromVec2D(vec);
        EXPECT_EQ(arr.argmin(), 1);
    }

    // axis is >= 0
    {
        auto arr = nc::NDArrayPtr<float>::FromVec3D(
        {{{-0.81122224,  0.67868092, -0.9949474 , -1.57278021},
        { 0.17435885, -0.97180358,  0.54741201,  0.34597885},
        {-1.00464501,  0.50596217,  1.28059368, -0.13757696}},

        {{ 0.17296277, -0.48834356,  0.52242913,  0.89538532},
        { 0.96225951,  0.13264605,  1.96598804,  0.19164458},
        {-0.0550882 , -0.29170627, -0.42841379, -2.31100097}}}
                    );
        auto argmin0 = arr.argmin(0);
        auto argmin0_gt = nc::NDArrayPtr<int>::FromVec2D( {{0, 1, 0, 0}, {0, 0, 0, 1}, {0, 1, 1, 1}} );
        for (auto i = 0; i < argmin0.elemCount(); i++) {
            EXPECT_EQ(argmin0.getitem(i), argmin0_gt.getitem(i));
        }

        auto argmin1 = arr.argmin(1);
        auto argmin1_gt = nc::NDArrayPtr<int>::FromVec2D( {{2, 1, 0, 0}, {2, 0, 2, 2}} );
        for (auto i = 0; i < argmin1.elemCount(); i++) {
            EXPECT_EQ(argmin1.getitem(i), argmin1_gt.getitem(i));
        }

        auto argmin2 = arr.argmin(2);
        auto argmin2_gt = nc::NDArrayPtr<int>::FromVec2D( {{3, 1, 0}, {1, 1, 3}} );
        for (auto i = 0; i < argmin2.elemCount(); i++) {
            EXPECT_EQ(argmin2.getitem(i), argmin2_gt.getitem(i));
        }
    }
}

TEST_F(NCTest, reflect_pad1d) {
    auto arr = nc::NDArrayPtr<float>::FromVec1D({ 1, 2, 3, 4, 5, });
    {
        std::vector<float> pad_gt = { 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, };
        auto pad = nc::reflect_pad1d(arr, {2, 3});
        std::cout << pad << std::endl;
        for (auto i = 0; i < pad.elemCount(); i++) {
            EXPECT_EQ(pad.getitem(i), pad_gt[i]);
        }
    }
    {
        std::vector<float> pad_gt = { 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, };
        auto pad = nc::reflect_pad1d(arr, {6, 7});
        std::cout << pad << std::endl;
        for (auto i = 0; i < pad.elemCount(); i++) {
            EXPECT_EQ(pad.getitem(i), pad_gt[i]);
        }
    }
    {
        std::vector<float> pad_gt = { 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, };
        auto pad = nc::reflect_pad1d(arr, {11, 12});
        std::cout << pad << std::endl;
        for (auto i = 0; i < pad.elemCount(); i++) {
            EXPECT_EQ(pad.getitem(i), pad_gt[i]);
        }
    }
}

#endif //
