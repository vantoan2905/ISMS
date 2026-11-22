#include "decode.hpp"
#include "encode.hpp"
#include "infer.hpp"
#include "passer.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <cmath>

namespace py = pybind11;
using Frame = py::array_t<uint8_t, py::array::c_style>;
using Tensor = py::array_t<float, py::array::c_style>;

static cv::Mat frame_view(const Frame& frame) {
    if (frame.ndim() != 3 || frame.shape(2) != 3 || !frame.size())
        throw py::value_error("Expected a nonempty H x W x 3 BGR frame");
    return cv::Mat(frame.shape(0), frame.shape(1), CV_8UC3,
                   const_cast<uint8_t*>(frame.data()));
}

PYBIND11_MODULE(_native, m) {
    py::class_<Decoder>(m, "Decoder")
        .def(py::init<const std::string&>())
        .def_property_readonly("fps", &Decoder::fps)
        .def("read", [](Decoder& decoder) -> py::object {
            cv::Mat frame;
            bool ok;
            {
                py::gil_scoped_release release;
                ok = decoder.read(frame);
            }
            if (!ok) return py::none();
            py::array_t<uint8_t> array({frame.rows, frame.cols, 3});
            for (int y = 0; y < frame.rows; ++y)
                std::memcpy(array.mutable_data(y, 0, 0), frame.ptr(y), frame.cols * 3);
            return array;
        });

    py::class_<Inference>(m, "Inference")
        .def(py::init<const std::string&>())
        .def("run", [](Inference& model, Frame frame) {
            cv::Mat input = frame_view(frame);
            cv::Mat output;
            {
                py::gil_scoped_release release;
                output = model.run(input);
            }
            if (output.type() != CV_32F)
                throw py::value_error("Expected float32 model output");
            if (!output.isContinuous()) output = output.clone();
            std::vector<py::ssize_t> shape(output.size.p, output.size.p + output.dims);
            py::array_t<float> array(shape);
            std::memcpy(array.mutable_data(), output.data, array.nbytes());
            return array;
        });
    m.attr("INPUT_SIZE") = int(Inference::input_size);

    m.def("parse_detections", [](Tensor tensor, int width, int height, float threshold) {
        if (tensor.ndim() != 3) throw py::value_error("Expected a rank-3 YOLO tensor");
        int dims[] = {int(tensor.shape(0)), int(tensor.shape(1)), int(tensor.shape(2))};
        cv::Mat output(3, dims, CV_32F, const_cast<float*>(tensor.data()));
        py::list results;
        for (const auto& d : parse_detections(output, {width, height}, Inference::input_size, threshold)) {
            results.append(py::make_tuple(d.class_id, d.score,
                py::make_tuple(d.box.x, d.box.y, d.box.br().x, d.box.br().y)));
        }
        return results;
    }, py::arg("tensor"), py::arg("width"), py::arg("height"), py::arg("threshold") = 0.4f);

    py::class_<Encoder>(m, "Encoder")
        .def(py::init([](const std::string& path, int width, int height, double fps) {
            if (width <= 0 || height <= 0 || !std::isfinite(fps) || fps <= 0)
                throw py::value_error("Invalid dimensions or frame rate");
            return std::make_unique<Encoder>(path, cv::Size(width, height), fps);
        }))
        .def("write", [](Encoder& encoder, Frame frame) {
            cv::Mat input = frame_view(frame);
            py::gil_scoped_release release;
            encoder.write(input);
        })
        .def("finish", &Encoder::finish, py::call_guard<py::gil_scoped_release>());
}
