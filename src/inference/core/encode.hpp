#pragma once
#include <gst/app/gstappsrc.h>
#include <opencv2/core.hpp>
#include <string>

class Encoder {
public:
    Encoder(const std::string& path, cv::Size size, double fps);
    ~Encoder();
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    void write(const cv::Mat& bgr);
    void finish();
private:
    GstElement* pipeline_ = nullptr;
    GstAppSrc* source_ = nullptr;
    cv::Size size_;
    int numerator_, denominator_;
    guint64 frames_ = 0;
};
