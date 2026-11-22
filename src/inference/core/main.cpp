#include "decode.hpp"
#include "encode.hpp"
#include "infer.hpp"
#include "passer.hpp"
#include <filesystem>
#include <iostream>
#include <optional>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv) {
    if (argc > 4) {
        std::cerr << "Usage: gst_rtsp_infer_encode [input [output.mov [yolo.onnx]]]\n";
        return 2;
    }
    try {
        std::string source = argc > 1 ? argv[1] : "test.yuv";
        std::string output = argc > 2 ? argv[2] : "output.mov";
        if (std::filesystem::exists(output) && std::filesystem::exists(source) &&
            std::filesystem::equivalent(source, output))
            throw std::runtime_error("Input and output must be different files");
        std::optional<Inference> model;
        if (argc > 3) model.emplace(argv[3]);
        Decoder decoder(source);
        cv::Mat frame;
        if (!decoder.read(frame)) throw std::runtime_error("No input frames");
        Encoder encoder(output, frame.size(), decoder.fps());
        do {
            if (model) {
                auto detections = parse_detections(model->run(frame), frame.size(), Inference::input_size);
                for (const auto& d : detections) {
                    cv::rectangle(frame, d.box, {0, 220, 0}, 2);
                    cv::putText(frame, std::to_string(d.class_id), d.box.tl(),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 220, 0}, 2);
                }
            }
            encoder.write(frame);
        } while (decoder.read(frame));
        encoder.finish();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
