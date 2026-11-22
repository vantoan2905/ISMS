#pragma once
#include <fstream>
#include <opencv2/videoio.hpp>
#include <string>

class Decoder {
public:
    explicit Decoder(const std::string& source);
    bool read(cv::Mat& bgr);
    double fps() const { return fps_; }
private:
    std::ifstream raw_;
    cv::VideoCapture capture_;
    double fps_ = 1;
};
