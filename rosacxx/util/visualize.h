#ifndef ROSACXX_UTIL_VISUALIZE_H
#define ROSACXX_UTIL_VISUALIZE_H

#if ROSACXX_TESTS_VISUALIZE

#include <opencv2/opencv.hpp>
#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace util {

inline void ShowNDArray2DF32(const nc::NDArrayPtr<float>& S, const char * windowName) {
    auto vec2d = S.toStdVector2D();
    cv::Mat imgS(int(vec2d.size()), int(vec2d[0].size()), CV_32FC1);
    for (auto r = 0; r < imgS.rows; r++) {
        for (auto c = 0; c < imgS.cols; c++) {
            imgS.at<float>(r, c) = vec2d[r][c];
        }
    }
    if (imgS.rows < 512 || imgS.cols < 512) {
        cv::resize(imgS, imgS, cv::Size(0, 0), 2.0, 2.0);
    }
    cv::imshow(windowName, imgS);
    cv::waitKey(0);
    cv::destroyWindow(windowName);
}

inline void ShowNDArray2DBool(const nc::NDArrayPtr<bool>& S, const char * windowName) {
    auto vec2d = S.toStdVector2D();
    cv::Mat imgS(int(vec2d.size()), int(vec2d[0].size()), CV_8UC1);
    for (auto r = 0; r < imgS.rows; r++) {
        for (auto c = 0; c < imgS.cols; c++) {
            imgS.at<unsigned char>(r, c) = vec2d[r][c] ? 0xff : 0x00;
        }
    }
    cv::imshow(windowName, imgS);
    cv::waitKey(0);
    cv::destroyWindow(windowName);
}

}
}

#endif // 

#endif // ROSACXX_UTIL_VISUALIZE_H
