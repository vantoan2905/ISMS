#pragma once
#include <opencv2/core.hpp>
#include <vector>

struct Detection {
    int class_id;
    float score;
    cv::Rect box;
};

// YOLO detection output without embedded NMS: [1, 4 + classes, candidates].
std::vector<Detection> parse_detections(const cv::Mat& tensor, cv::Size frame,
                                        int input_size, float threshold = 0.4f);
