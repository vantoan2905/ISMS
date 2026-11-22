// Python binding of the v2 tracker (module botsort_native_v2). It is a thin
// adapter: options cross as Python dicts and are converted to YAML nodes, so the
// C++ loader is the only place that knows option names, defaults and checks.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "botsort_v2/multi_source_tracker.hpp"
#include "botsort_v2/settings.hpp"

namespace py = pybind11;
using namespace botsort_v2;

namespace {

YAML::Node to_yaml_node(const py::handle &value) {
  if (value.is_none()) return YAML::Node(YAML::NodeType::Null);
  // bool before int: Python's bool is an int subclass.
  if (py::isinstance<py::bool_>(value)) return YAML::Node(value.cast<bool>());
  if (py::isinstance<py::int_>(value)) return YAML::Node(value.cast<long long>());
  if (py::isinstance<py::float_>(value)) return YAML::Node(value.cast<double>());
  if (py::isinstance<py::str>(value)) return YAML::Node(value.cast<std::string>());
  if (py::isinstance<py::dict>(value)) {
    YAML::Node map(YAML::NodeType::Map);
    for (const auto &item : value.cast<py::dict>()) {
      map[py::str(item.first).cast<std::string>()] = to_yaml_node(item.second);
    }
    return map;
  }
  if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
    YAML::Node seq(YAML::NodeType::Sequence);
    for (const auto &item : value) seq.push_back(to_yaml_node(item));
    return seq;
  }
  // numpy scalars and the like.
  if (py::hasattr(value, "item")) return to_yaml_node(value.attr("item")());
  throw py::type_error("unsupported tracker option value: " +
                       py::repr(value).cast<std::string>());
}

py::dict settings_dict(const Settings &settings) {
  py::module_ yaml = py::module_::import("yaml");
  return yaml.attr("safe_load")(YAML::Dump(to_yaml(settings))).cast<py::dict>();
}

Settings layered(const Settings &base, const py::handle &options, const std::string &where) {
  Settings settings = base;
  if (!options.is_none()) apply_options(settings, to_yaml_node(options), where);
  return settings;
}

// A numpy frame plus the borrowed view the tracker reads from it.
struct FrameHolder {
  py::array_t<unsigned char, py::array::c_style | py::array::forcecast> array;
  FrameInput input;
};

FrameHolder frame_input(const py::object &frame) {
  FrameHolder holder;
  holder.array = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>(frame);
  const auto &a = holder.array;
  if (a.ndim() == 2) {
    holder.input.channels = 1;
  } else if (a.ndim() == 3 && (a.shape(2) == 1 || a.shape(2) == 3 || a.shape(2) == 4)) {
    holder.input.channels = static_cast<int>(a.shape(2));
  } else {
    throw py::value_error("frame must be HxW, HxWx3 (BGR) or HxWx4 uint8");
  }
  holder.input.data = a.data();
  holder.input.height = static_cast<int>(a.shape(0));
  holder.input.width = static_cast<int>(a.shape(1));
  holder.input.row_stride = static_cast<size_t>(a.strides(0));
  return holder;
}

}  // namespace

PYBIND11_MODULE(botsort_native_v2, m) {
  m.doc() = "BoT-SORT tracker v2: C++ multi-source tracker configured from YAML";

  py::register_exception<std::invalid_argument>(m, "TrackerConfigError", PyExc_ValueError);

  m.def("option_names", &option_names, "Every accepted tracker option name.");
  m.attr("cmc_available") = cmc_available();
  m.def(
      "load_settings",
      [](const std::string &path, const std::optional<std::string> &camera) {
        ConfigFile config = load_config_file(path);
        return settings_dict(camera ? config.for_camera(*camera) : config.defaults);
      },
      py::arg("path"), py::arg("camera") = py::none(),
      "Effective options from a tracker YAML, for its defaults or one camera.");
  m.def(
      "load_cameras",
      [](const std::string &path) {
        ConfigFile config = load_config_file(path);
        py::module_ yaml = py::module_::import("yaml");
        py::dict out;
        for (const auto &[name, node] : config.cameras) {
          out[py::str(name)] = yaml.attr("safe_load")(YAML::Dump(node));
        }
        return out;
      },
      py::arg("path"), "The per-camera override mappings of a tracker YAML, as written.");

  m.def(
      "compact_mask",
      [](const py::array_t<unsigned char, py::array::c_style | py::array::forcecast> &mask,
         int resolution) -> py::object {
        if (mask.ndim() != 2) throw py::value_error("mask must be two-dimensional");
        auto out = MultiSourceTracker::compact_mask(
            mask.data(), static_cast<int>(mask.shape(0)), static_cast<int>(mask.shape(1)),
            resolution);
        if (out.empty()) return py::none();
        py::array_t<unsigned char> result({resolution, resolution});
        std::copy(out.begin(), out.end(), result.mutable_data());
        return std::move(result);
      },
      py::arg("mask"), py::arg("resolution"),
      "The compact mask the tracker would use (None when empty); for tests.");

  // The tracker's own GMC, exposed so camera-motion estimates can be checked
  // (and tuned) without a tracker around them.
  py::class_<GlobalMotionCompensation>(m, "CameraMotion")
      .def(py::init([](const std::string &method, int downscale) {
             return std::make_unique<GlobalMotionCompensation>(parse_cmc_method(method),
                                                               downscale);
           }),
           py::arg("method") = "sparse_optflow", py::arg("downscale") = 2)
      .def(
          "apply",
          [](GlobalMotionCompensation &self, const py::object &frame) {
            FrameHolder holder = frame_input(frame);
            Affine2x3 warp = self.apply(holder.input);
            py::array_t<double> out({2, 3});
            std::copy(warp.begin(), warp.end(), out.mutable_data());
            return out;
          },
          py::arg("frame"),
          "2x3 warp from the previous frame to this one (identity on the first frame).")
      .def("reset", &GlobalMotionCompensation::reset);

  py::class_<Assignment>(m, "Assignment")
      .def_readonly("detection_index", &Assignment::detection_index)
      .def_readonly("track_id", &Assignment::track_id)
      .def_readonly("association_score", &Assignment::association_score)
      .def_readonly("association_reason", &Assignment::association_reason)
      .def_readonly("lineage_id", &Assignment::lineage_id)
      .def_readonly("predecessor_track_id", &Assignment::predecessor_track_id)
      .def("__repr__", [](const Assignment &a) {
        return "<Assignment det=" + std::to_string(a.detection_index) +
               " track=" + std::to_string(a.track_id) + " reason=" + a.association_reason + ">";
      });

  py::class_<MultiSourceTracker>(m, "MultiSourceTracker")
      .def(py::init([](const std::string &config_path, const py::object &options,
                       const py::object &options_by_source) {
             Settings defaults =
                 config_path.empty() ? Settings() : load_config_file(config_path).defaults;
             defaults = layered(defaults, options, "tracker options");
             auto tracker = std::make_unique<MultiSourceTracker>(defaults);
             if (!options_by_source.is_none()) {
               for (const auto &item : options_by_source.cast<py::dict>()) {
                 int source_id = item.first.cast<int>();
                 tracker->set_source_settings(
                     source_id, layered(defaults, item.second,
                                        "tracker options for source " + std::to_string(source_id)));
               }
             }
             return tracker;
           }),
           py::arg("config_path") = std::string(), py::arg("options") = py::none(),
           py::arg("options_by_source") = py::none(),
           "config_path: tracker YAML ('' = built-in defaults); options: overrides on its\n"
           "defaults; options_by_source: {source_id: overrides} on top of those.")
      .def(
          "settings",
          [](const MultiSourceTracker &self, int source_id) {
            return settings_dict(self.settings_for(source_id));
          },
          py::arg("source_id") = 0, "Effective options of one source, as a dict.")
      .def_property_readonly("uses_masks",
                             py::overload_cast<>(&MultiSourceTracker::uses_masks, py::const_))
      .def("uses_masks_for", py::overload_cast<int>(&MultiSourceTracker::uses_masks, py::const_),
           py::arg("source_id"))
      .def("uses_reid_for", &MultiSourceTracker::uses_reid, py::arg("source_id"))
      .def("uses_frames_for", &MultiSourceTracker::uses_frames, py::arg("source_id"))
      .def(
          "update",
          [](MultiSourceTracker &self, int source_id, double frame_number,
             const py::list &detections, const py::object &frame,
             const py::object &embeddings) {
            std::vector<DetectionInput> inputs;
            // Keeps every mask buffer alive until the native update returns.
            std::vector<py::array_t<unsigned char, py::array::c_style | py::array::forcecast>>
                masks;
            inputs.reserve(detections.size());
            masks.reserve(detections.size());
            for (const auto &handle : detections) {
              py::tuple item = py::cast<py::tuple>(handle);
              if (item.size() != 6 && item.size() != 7) {
                throw py::value_error(
                    "detection must be (left, top, width, height, score, class_id[, mask])");
              }
              DetectionInput in;
              in.left = item[0].cast<double>();
              in.top = item[1].cast<double>();
              in.width = item[2].cast<double>();
              in.height = item[3].cast<double>();
              in.score = item[4].cast<double>();
              in.class_id = item[5].cast<int>();
              if (item.size() == 7 && !item[6].is_none()) {
                auto mask = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>(
                    item[6]);
                if (mask.ndim() == 2 && mask.size() > 0) {
                  in.mask = mask.data();
                  in.mask_height = static_cast<int>(mask.shape(0));
                  in.mask_width = static_cast<int>(mask.shape(1));
                  masks.push_back(std::move(mask));
                }
              }
              inputs.push_back(in);
            }
            // Keeps every embedding buffer alive until the native update returns.
            std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> features;
            if (!embeddings.is_none()) {
              py::sequence items = py::cast<py::sequence>(embeddings);
              if (static_cast<size_t>(py::len(items)) != inputs.size()) {
                throw py::value_error("embeddings must have one entry per detection");
              }
              features.reserve(inputs.size());
              for (size_t i = 0; i < inputs.size(); ++i) {
                py::object item = items[i];
                if (item.is_none()) continue;
                auto feature =
                    py::array_t<float, py::array::c_style | py::array::forcecast>(item);
                if (feature.ndim() != 1) {
                  throw py::value_error("each embedding must be one-dimensional");
                }
                if (feature.size() == 0) continue;
                inputs[i].embedding = feature.data();
                inputs[i].embedding_dim = static_cast<int>(feature.size());
                features.push_back(std::move(feature));
              }
            }
            if (frame.is_none()) return self.update(source_id, frame_number, inputs);
            FrameHolder holder = frame_input(frame);
            return self.update(source_id, frame_number, inputs, &holder.input);
          },
          py::arg("source_id"), py::arg("frame_number"), py::arg("detections"),
          py::arg("frame") = py::none(), py::arg("embeddings") = py::none(),
          "detections: (left, top, width, height, score, class_id[, bbox-local mask]) tuples.\n"
          "frame: optional uint8 HxW / HxWx{3,4} image for camera motion compensation, in the\n"
          "detections' pixel coordinates. embeddings: optional list, one 1-D float array (or\n"
          "None) per detection, for ReID. Returns the accepted Assignment list for this frame.")
      .def(
          "reset",
          [](MultiSourceTracker &self, const py::object &source_id) {
            if (source_id.is_none()) {
              self.reset();
            } else {
              self.reset(source_id.cast<int>());
            }
          },
          py::arg("source_id") = py::none());
}
