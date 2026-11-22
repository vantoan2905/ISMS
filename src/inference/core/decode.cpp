#include "decode.hpp"
#include <cmath>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

Decoder::Decoder(const std::string& source) {
    if (std::filesystem::path(source).extension() == ".yuv") {
        raw_.open(source, std::ios::binary);
        if (!raw_) throw std::runtime_error("Cannot open " + source);
    } else {
        if (!capture_.open(source)) throw std::runtime_error("Cannot decode " + source);
        fps_ = capture_.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps_) || fps_ <= 0)
            throw std::runtime_error("Source has no usable frame rate");
    }
}

bool Decoder::read(cv::Mat& bgr) {
    if (!raw_.is_open()) return capture_.read(bgr);
    cv::Mat i420(720 * 3 / 2, 1280, CV_8U);
    raw_.read(reinterpret_cast<char*>(i420.data), i420.total());
    if (raw_.bad()) throw std::runtime_error("I420 read error");
    if (raw_.gcount() == 0 && raw_.eof()) return false;
    if (!raw_) throw std::runtime_error("Incomplete 1280x720 I420 frame");
    cv::cvtColor(i420, bgr, cv::COLOR_YUV2BGR_I420);
    return true;
}
