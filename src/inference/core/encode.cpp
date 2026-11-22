#include "encode.hpp"
#include <cstring>
#include <stdexcept>

namespace {
void check_bus(GstElement* pipeline, bool wait) {
    GstBus* bus = gst_element_get_bus(pipeline);
    auto types = static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS);
    GstMessage* msg = gst_bus_timed_pop_filtered(bus, wait ? 60 * GST_SECOND : 0, types);
    gst_object_unref(bus);
    if (!msg) {
        if (wait) throw std::runtime_error("Timed out finalizing movie");
        return;
    }
    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* error = nullptr;
        gst_message_parse_error(msg, &error, nullptr);
        std::string reason(error->message);
        g_error_free(error);
        gst_message_unref(msg);
        throw std::runtime_error(reason);
    }
    gst_message_unref(msg);
}
}

Encoder::Encoder(const std::string& path, cv::Size size, double fps) : size_(size) {
    gst_init(nullptr, nullptr);
    gst_util_double_to_fraction(fps, &numerator_, &denominator_);
    GError* error = nullptr;
    pipeline_ = gst_parse_launch("appsrc name=input format=time block=true ! videoconvert ! "
                                 "video/x-raw,format=I420 ! x264enc ! h264parse ! qtmux ! filesink name=output", &error);
    if (error) {
        std::string reason(error->message);
        g_error_free(error);
        if (pipeline_) gst_object_unref(pipeline_);
        throw std::runtime_error(reason);
    }
    source_ = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(pipeline_), "input"));
    GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, size.width, "height", G_TYPE_INT, size.height,
        "framerate", GST_TYPE_FRACTION, numerator_, denominator_, nullptr);
    gst_app_src_set_caps(source_, caps);
    gst_caps_unref(caps);
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline_), "output");
    g_object_set(sink, "location", path.c_str(), nullptr);
    gst_object_unref(sink);
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(source_);
        gst_object_unref(pipeline_);
        throw std::runtime_error("Cannot start encoder for " + path);
    }
}

Encoder::~Encoder() {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(source_);
    gst_object_unref(pipeline_);
}

void Encoder::write(const cv::Mat& bgr) {
    check_bus(pipeline_, false);
    if (bgr.size() != size_ || bgr.type() != CV_8UC3)
        throw std::runtime_error("Frame size or format changed");
    const size_t row_bytes = size_.width * size_t{3};
    const size_t stride = GST_ROUND_UP_4(row_bytes);
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, stride * size_.height, nullptr);
    GstMapInfo map;
    if (!buffer || !gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        if (buffer) gst_buffer_unref(buffer);
        throw std::runtime_error("Cannot allocate video buffer");
    }
    for (int y = 0; y < size_.height; ++y)
        std::memcpy(map.data + stride * y, bgr.ptr(y), row_bytes);
    gst_buffer_unmap(buffer, &map);
    auto timestamp = [this](guint64 index) {
        return gst_util_uint64_scale(index, GST_SECOND * denominator_, numerator_);
    };
    GST_BUFFER_PTS(buffer) = timestamp(frames_);
    GST_BUFFER_DURATION(buffer) = timestamp(frames_ + 1) - timestamp(frames_);
    if (gst_app_src_push_buffer(source_, buffer) != GST_FLOW_OK)
        throw std::runtime_error("Encoder rejected frame");
    ++frames_;
}

void Encoder::finish() {
    if (gst_app_src_end_of_stream(source_) != GST_FLOW_OK)
        throw std::runtime_error("Encoder rejected end of stream");
    check_bus(pipeline_, true);
}
