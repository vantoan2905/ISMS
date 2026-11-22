#pragma once
#include <opencv2/dnn.hpp>

class Inference {
public:
    explicit Inference(const std::string& onnx);
    cv::Mat run(const cv::Mat& bgr);
    static constexpr int input_size = 640;
private:
    cv::dnn::Net net_;
};
