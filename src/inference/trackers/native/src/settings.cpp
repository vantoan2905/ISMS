#include "botsort/settings.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace botsort {
namespace {

enum class Kind { Real, Fraction, Int, Bool };

struct Field {
  const char *name;
  Kind kind;
  std::function<void(Settings &, double)> set;
  std::function<double(const Settings &)> get;
};

#define BOTSORT_FIELD(NAME, KIND, EXPR)                                  \
  Field {                                                                \
    #NAME, Kind::KIND, [](Settings &s, double v) { EXPR = v; },          \
        [](const Settings &s) { return static_cast<double>(EXPR); }      \
  }

// The single list of option names. track_buffer_seconds is handled separately
// because it is the only nullable option.
const std::vector<Field> &fields() {
  static const std::vector<Field> table = {
      // Association thresholds.
      BOTSORT_FIELD(track_high_thresh, Real, s.tracker.track_high_thresh),
      BOTSORT_FIELD(track_low_thresh, Real, s.tracker.track_low_thresh),
      BOTSORT_FIELD(new_track_thresh, Real, s.tracker.new_track_thresh),
      BOTSORT_FIELD(match_thresh, Real, s.tracker.match_thresh),
      BOTSORT_FIELD(second_match_thresh, Real, s.tracker.second_match_thresh),
      BOTSORT_FIELD(unconfirmed_match_thresh, Real, s.tracker.unconfirmed_match_thresh),
      BOTSORT_FIELD(duplicate_iou_thresh, Real, s.tracker.duplicate_iou_thresh),
      // Lifecycle.
      Field{"track_buffer", Kind::Int,
            [](Settings &s, double v) { s.tracker.track_buffer = static_cast<int>(v); },
            [](const Settings &s) { return static_cast<double>(s.tracker.track_buffer); }},
      BOTSORT_FIELD(fps, Real, s.fps),
      // Lost-track handling.
      BOTSORT_FIELD(lost_match_last_observation, Bool, s.tracker.lost_match_last_observation),
      BOTSORT_FIELD(lost_in_second_stage, Bool, s.tracker.lost_in_second_stage),
      // Proximity rescue.
      BOTSORT_FIELD(proximity_rescue, Bool, s.tracker.proximity_rescue),
      BOTSORT_FIELD(proximity_max_normalized_distance, Real,
                    s.tracker.proximity_max_normalized_distance),
      // ReID appearance fusion.
      BOTSORT_FIELD(with_reid, Bool, s.tracker.with_reid),
      BOTSORT_FIELD(reid_iou_gate, Fraction, s.tracker.reid_iou_gate),
      BOTSORT_FIELD(reid_appearance_thresh, Fraction, s.tracker.reid_appearance_thresh),
      BOTSORT_FIELD(reid_feature_alpha, Fraction, s.tracker.reid_feature_alpha),
      // Camera motion compensation (cmc_method is a string, handled separately).
      Field{"cmc_downscale", Kind::Int,
            [](Settings &s, double v) { s.cmc_downscale = std::max(1, static_cast<int>(v)); },
            [](const Settings &s) { return static_cast<double>(s.cmc_downscale); }},
      // Mask-conditioned association.
      BOTSORT_FIELD(mask_conditioned_association, Bool,
                    s.tracker.mask_conditioned_association),
      BOTSORT_FIELD(mask_enable_isolated_matching, Bool,
                    s.tracker.mask_enable_isolated_matching),
      BOTSORT_FIELD(mask_min_track_confidence, Fraction, s.tracker.mask_min_track_confidence),
      BOTSORT_FIELD(mask_min_track_coverage, Fraction, s.tracker.mask_min_track_coverage),
      BOTSORT_FIELD(mask_min_detection_precision, Fraction,
                    s.tracker.mask_min_detection_precision),
      BOTSORT_FIELD(mask_similarity_weight, Fraction, s.tracker.mask_similarity_weight),
      BOTSORT_FIELD(mask_collect_gate_samples, Bool, s.tracker.mask_collect_gate_samples),
      Field{"mask_resolution", Kind::Int,
            [](Settings &s, double v) { s.mask_resolution = std::max(4, static_cast<int>(v)); },
            [](const Settings &s) { return static_cast<double>(s.mask_resolution); }},
      // Kalman filter.
      BOTSORT_FIELD(kalman_dt, Real, s.tracker.kalman_dt),
      BOTSORT_FIELD(kalman_std_weight_position, Real, s.tracker.kalman_std_weight_position),
      BOTSORT_FIELD(kalman_std_weight_velocity, Real, s.tracker.kalman_std_weight_velocity),
      // Output pairing.
      BOTSORT_FIELD(match_iou, Real, s.match_iou),
      BOTSORT_FIELD(enforce_match_iou, Bool, s.enforce_match_iou),
      // Lineage.
      BOTSORT_FIELD(lineage_enabled, Bool, s.lineage_enabled),
      Field{"lineage_max_seconds", Kind::Real,
            [](Settings &s, double v) { s.lineage_max_seconds = std::max(0.0, v); },
            [](const Settings &s) { return s.lineage_max_seconds; }},
      Field{"lineage_distance_scale", Kind::Real,
            [](Settings &s, double v) { s.lineage_distance_scale = std::max(0.0, v); },
            [](const Settings &s) { return s.lineage_distance_scale; }},
      BOTSORT_FIELD(max_tracking_distance, Real, s.max_tracking_distance),
  };
  return table;
}

#undef BOTSORT_FIELD

constexpr const char *kTrackBufferSeconds = "track_buffer_seconds";
constexpr const char *kCmcMethod = "cmc_method";

std::string location(const std::string &where, const std::string &name) {
  return where.empty() ? name : where + ": " + name;
}

double as_number(const YAML::Node &value, const std::string &label) {
  try {
    return value.as<double>();
  } catch (const YAML::Exception &) {
  }
  try {
    return value.as<bool>() ? 1.0 : 0.0;
  } catch (const YAML::Exception &) {
  }
  throw std::invalid_argument(label + " must be a number, got '" +
                              (value.IsScalar() ? value.Scalar() : std::string("<non-scalar>")) +
                              "'");
}

double as_flag(const YAML::Node &value, const std::string &label) {
  try {
    return value.as<bool>() ? 1.0 : 0.0;
  } catch (const YAML::Exception &) {
  }
  return as_number(value, label) != 0.0 ? 1.0 : 0.0;
}

}  // namespace

BotSortTracker::Config Settings::resolved_tracker_config() const {
  BotSortTracker::Config config = tracker;
  if (track_buffer_seconds) {
    // nearbyint rounds half to even, like Python's round().
    double frames = std::nearbyint(*track_buffer_seconds * fps);
    config.track_buffer = std::max(1, static_cast<int>(frames));
  }
  return config;
}

void apply_options(Settings &settings, const YAML::Node &options, const std::string &where) {
  if (!options || options.IsNull()) return;
  if (!options.IsMap()) {
    throw std::invalid_argument(location(where, "tracker options must be a mapping"));
  }
  for (const auto &item : options) {
    const std::string name = item.first.as<std::string>();
    const YAML::Node &value = item.second;
    const std::string label = location(where, name);
    if (name == kTrackBufferSeconds) {
      if (value.IsNull()) {
        settings.track_buffer_seconds.reset();
      } else {
        settings.track_buffer_seconds = as_number(value, label);
      }
      continue;
    }
    if (name == kCmcMethod) {
      if (!value.IsScalar()) throw std::invalid_argument(label + " must be a string");
      CmcMethod method;
      try {
        method = parse_cmc_method(value.Scalar());
      } catch (const std::invalid_argument &exc) {
        throw std::invalid_argument(location(where, exc.what()));
      }
      if (method != CmcMethod::None && !cmc_available()) {
        throw std::invalid_argument(label + ": '" + value.Scalar() +
                                    "' needs OpenCV, but the tracker was built with "
                                    "BOTSORT_CMC=OFF");
      }
      settings.cmc_method = method;
      continue;
    }
    const auto &table = fields();
    auto field = std::find_if(table.begin(), table.end(),
                              [&](const Field &f) { return name == f.name; });
    if (field == table.end()) {
      throw std::invalid_argument("unknown tracker option '" + label + "'");
    }
    if (!value.IsScalar()) {
      throw std::invalid_argument(label + " must be a scalar");
    }
    double number = field->kind == Kind::Bool ? as_flag(value, label) : as_number(value, label);
    if (field->kind == Kind::Fraction && !(number >= 0.0 && number <= 1.0)) {
      throw std::invalid_argument(label + " must be between 0 and 1");
    }
    field->set(settings, number);
  }
}

YAML::Node to_yaml(const Settings &settings) {
  YAML::Node out(YAML::NodeType::Map);
  for (const auto &field : fields()) {
    double value = field.get(settings);
    switch (field.kind) {
      case Kind::Bool:
        out[field.name] = value != 0.0;
        break;
      case Kind::Int:
        out[field.name] = static_cast<int>(value);
        break;
      default:
        out[field.name] = value;
    }
  }
  if (settings.track_buffer_seconds) {
    out[kTrackBufferSeconds] = *settings.track_buffer_seconds;
  } else {
    out[kTrackBufferSeconds] = YAML::Node(YAML::NodeType::Null);
  }
  out[kCmcMethod] = cmc_method_name(settings.cmc_method);
  return out;
}

std::vector<std::string> option_names() {
  std::vector<std::string> names;
  for (const auto &field : fields()) names.emplace_back(field.name);
  names.emplace_back(kTrackBufferSeconds);
  names.emplace_back(kCmcMethod);
  return names;
}

Settings ConfigFile::for_camera(const std::string &camera) const {
  Settings settings = defaults;
  auto it = cameras.find(camera);
  if (it != cameras.end()) {
    apply_options(settings, it->second, path + ": cameras." + camera);
  }
  return settings;
}

ConfigFile parse_config(const YAML::Node &root, const std::string &where) {
  ConfigFile config;
  config.path = where;
  if (!root || root.IsNull()) return config;
  if (!root.IsMap()) {
    throw std::invalid_argument(where + ": the tracker config must be a mapping");
  }
  for (const auto &item : root) {
    const std::string key = item.first.as<std::string>();
    if (key != "defaults" && key != "cameras") {
      throw std::invalid_argument(where + ": unknown top-level key '" + key +
                                  "' (expected 'defaults' and/or 'cameras')");
    }
  }

  const YAML::Node defaults = root["defaults"];
  if (defaults && !defaults.IsNull()) {
    if (!defaults.IsMap()) {
      throw std::invalid_argument(where + ": 'defaults' must be a mapping of groups");
    }
    std::set<std::string> seen;
    for (const auto &group : defaults) {
      const std::string group_name = group.first.as<std::string>();
      const std::string group_where = where + ": defaults." + group_name;
      if (!group.second.IsMap()) {
        throw std::invalid_argument(group_where + " must be a mapping");
      }
      for (const auto &option : group.second) {
        const std::string name = option.first.as<std::string>();
        if (!seen.insert(name).second) {
          throw std::invalid_argument(group_where + ": '" + name +
                                      "' is set in more than one group");
        }
      }
      apply_options(config.defaults, group.second, group_where);
    }
  }

  const YAML::Node cameras = root["cameras"];
  if (cameras && !cameras.IsNull()) {
    if (!cameras.IsMap()) {
      throw std::invalid_argument(where + ": 'cameras' must map camera names to options");
    }
    for (const auto &camera : cameras) {
      const std::string name = camera.first.as<std::string>();
      // Validate now so a bad camera entry fails at load, not on first use.
      Settings probe = config.defaults;
      apply_options(probe, camera.second, where + ": cameras." + name);
      config.cameras[name] = YAML::Clone(camera.second);
    }
  }
  return config;
}

ConfigFile load_config_file(const std::string &path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::BadFile &) {
    throw std::invalid_argument("cannot read tracker config " + path);
  } catch (const YAML::Exception &exc) {
    throw std::invalid_argument(path + ": " + exc.what());
  }
  return parse_config(root, path);
}

}  // namespace botsort
