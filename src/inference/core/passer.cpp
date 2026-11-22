#include "passer.hpp"
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

std::vector<Detection> parse_detections(const cv::Mat& tensor, cv::Size frame,
                                        int input_size, float threshold) {
    if (tensor.dims != 3 || tensor.size[0] != 1 || tensor.type() != CV_32F ||
        tensor.size[1] < 5 || !tensor.isContinuous() || input_size <= 0)
        throw std::runtime_error("Expected float32 YOLO output [1, 4+classes, candidates]");
    cv::Mat rows;
    cv::transpose(tensor.reshape(1, tensor.size[1]), rows);
    std::map<int, std::vector<Detection>> groups;
    for (int i = 0; i < rows.rows; ++i) {
        const float* row = rows.ptr<float>(i);
        const float* best = std::max_element(row + 4, row + rows.cols);
        if (!std::all_of(row, row + rows.cols, [](float v){ return std::isfinite(v); }) ||
            *best <= threshold || row[2] <= 0 || row[3] <= 0) continue;
        auto pixel = [input_size](double value, int extent) {
            return cvRound(std::clamp(value / input_size, 0.0, 1.0) * extent);
        };
        cv::Point tl(pixel(row[0]-row[2]*0.5, frame.width), pixel(row[1]-row[3]*0.5, frame.height));
        cv::Point br(pixel(row[0]+row[2]*0.5, frame.width), pixel(row[1]+row[3]*0.5, frame.height));
        cv::Rect box(tl, br);
        int id = static_cast<int>(best - (row + 4));
        if (!box.empty()) groups[id].push_back({id, *best, box});
    }
    std::vector<Detection> result;
    for (const auto& entry : groups) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        for (const auto& d : entry.second) {
            boxes.push_back(d.box);
            scores.push_back(d.score);
        }
        std::vector<int> kept;
        cv::dnn::NMSBoxes(boxes, scores, threshold, 0.45f, kept);
        for (int i : kept) result.push_back(entry.second[i]);
    }
    return result;
}
