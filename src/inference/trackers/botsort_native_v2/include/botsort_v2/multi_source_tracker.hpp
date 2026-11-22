// Track each video source independently.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bot_sort_tracker.hpp"
#include "settings.hpp"

namespace botsort_v2 {

// One input detection. The mask, when given, is the detector's bbox-local mask
// (row-major, non-zero = object) of any size; it is only read during update()
// and only when the source has mask_conditioned_association on.
struct DetectionInput {
  double left = 0.0;
  double top = 0.0;
  double width = 0.0;
  double height = 0.0;
  double score = 0.0;
  int class_id = 0;
  const unsigned char *mask = nullptr;
  int mask_height = 0;
  int mask_width = 0;
  // Optional ReID embedding (any dimension, normalized here). Only read during
  // update() and only when the source has with_reid on.
  const float *embedding = nullptr;
  int embedding_dim = 0;
};

// One accepted (detection -> track) pairing for a frame.
struct Assignment {
  int detection_index;  // index into the vector passed to update()
  int track_id;
  double association_score;  // IoU of the tracked box and the detection box
  std::string association_reason;
  std::optional<int> lineage_id;            // set only when lineage is enabled
  std::optional<int> predecessor_track_id;  // set when this id was linked to an older one
};

class MultiSourceTracker {
 public:
  explicit MultiSourceTracker(Settings defaults = Settings());

  // Settings for one source; a source without its own uses the defaults.
  // Changing them drops that source's tracker state.
  void set_source_settings(int source_id, const Settings &settings);
  const Settings &settings_for(int source_id) const;
  const Settings &default_settings() const { return defaults_; }

  // Whether any configured source (or the defaults, when none is configured)
  // runs mask-conditioned association, i.e. whether callers must decode masks.
  bool uses_masks() const;
  bool uses_masks(int source_id) const;
  // Whether a source fuses ReID embeddings / needs frames for camera motion.
  bool uses_reid(int source_id) const;
  bool uses_frames(int source_id) const;

  // Tracks one frame of one source. ``frame_number`` also clocks lineage gaps.
  // ``frame`` (optional) feeds camera motion compensation when the source has a
  // cmc_method; it must be in the detections' pixel coordinates. A frame left
  // out is treated as "camera did not move" and the next frame given is
  // compared with the last one seen.
  std::vector<Assignment> update(int source_id, double frame_number,
                                 const std::vector<DetectionInput> &detections,
                                 const FrameInput *frame = nullptr);

  void reset();
  void reset(int source_id);

  // Bbox-local nearest-neighbour downsample to resolution x resolution, the same
  // sampling as cv2.resize(..., INTER_NEAREST). Empty when the mask is absent or
  // has no foreground pixel.
  static std::vector<unsigned char> compact_mask(const unsigned char *mask, int height,
                                                 int width, int resolution);
  // L2-normalized copy of an embedding; empty when absent, non-finite or zero.
  static std::vector<float> normalized_embedding(const float *embedding, int dim);

 private:
  struct Observation {
    double timestamp = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    int class_id = 0;
    bool has_velocity = false;
    double vx = 0.0;
    double vy = 0.0;
  };

  struct SourceState {
    std::unique_ptr<BotSortTracker> tracker;
    // Present only when the source has a cmc_method.
    std::unique_ptr<GlobalMotionCompensation> gmc;
    // visible track id -> lineage id (never pruned).
    std::unordered_map<int, int> lineages;
    // visible track id -> last observation, kept in first-insertion order so
    // ties between equally distant predecessors resolve deterministically.
    std::vector<std::pair<int, Observation>> history;
  };

  SourceState &state_for(int source_id);
  std::pair<int, std::optional<int>> lineage_for(SourceState &state, const Settings &settings,
                                                 double timestamp, int track_id, double cx,
                                                 double cy, int class_id,
                                                 const std::vector<int> &current_ids);

  Settings defaults_;
  std::map<int, Settings> source_settings_;
  std::map<int, SourceState> states_;
};

}  // namespace botsort_v2
