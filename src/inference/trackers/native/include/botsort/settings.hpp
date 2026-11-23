#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "bot_sort_tracker.hpp"

namespace botsort {

struct Settings {
  // Association, lifecycle, mask, proximity and Kalman options of the core tracker.
  BotSortTracker::Config tracker;
  // Optional: overrides tracker.track_buffer as round(seconds * fps).
  std::optional<double> track_buffer_seconds;
  // Stream frame rate; only used to convert track_buffer_seconds.
  double fps = 10.0;

  // Output pairing: a native assignment is dropped when its box IoU with the
  // source detection is below match_iou, but only when enforce_match_iou is on.
  double match_iou = 0.5;
  bool enforce_match_iou = false;

  // Lineage: links a newly born id to a recently ended one of the same class.
  // Gaps are measured in the unit of the frame number passed to update().
  bool lineage_enabled = false;
  double lineage_max_seconds = 1.5;
  double lineage_distance_scale = 1.0;
  double max_tracking_distance = 120.0;

  // Side of the compact bbox-local mask used by mask-conditioned association.
  int mask_resolution = 24;

  // Camera motion compensation (gmc.hpp). Needs a frame per update(); without
  // one the camera is treated as fixed for that frame.
  CmcMethod cmc_method = CmcMethod::None;
  // Integer factor frames are shrunk by before motion estimation (BoT-SORT: 2).
  int cmc_downscale = 2;

  // The core tracker config with track_buffer_seconds resolved.
  BotSortTracker::Config resolved_tracker_config() const;
};

// Applies one flat mapping of options on top of ``settings``. ``where`` names the
// source (file / camera) in error messages.
void apply_options(Settings &settings, const YAML::Node &options, const std::string &where);

// Every option of ``settings`` as a flat mapping, in a stable order.
YAML::Node to_yaml(const Settings &settings);

// Names of every accepted option, in the order to_yaml() writes them.
std::vector<std::string> option_names();

struct ConfigFile {
  std::string path;
  Settings defaults;
  // camera name -> flat options mapping, applied on top of ``defaults``.
  std::map<std::string, YAML::Node> cameras;

  // ``defaults`` with the camera's overrides applied (``defaults`` if the
  // camera has none).
  Settings for_camera(const std::string &camera) const;
};

// root: {defaults: {group: {option: value}}, cameras: {name: {option: value}}}.
// Both keys are optional; groups only exist for readability and are flattened.
ConfigFile parse_config(const YAML::Node &root, const std::string &where);
ConfigFile load_config_file(const std::string &path);

}  // namespace botsort
