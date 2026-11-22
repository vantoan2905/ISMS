#include "infer.hpp"

Inference::Inference(const std::string& onnx) : net_(cv::dnn::readNetFromONNX(onnx)) {}

cv::Mat Inference::run(const cv::Mat& bgr) {
    auto blob = cv::dnn::blobFromImage(bgr, 1.0 / 255, {input_size, input_size},
                                      cv::Scalar(), true, false);
    net_.setInput(blob);
    return net_.forward();
}
