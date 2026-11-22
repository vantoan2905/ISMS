#include "botsort_v2/multi_source_tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace botsort_v2 {

MultiSourceTracker::MultiSourceTracker(Settings defaults) : defaults_(std::move(defaults)) {}

void MultiSourceTracker::set_source_settings(int source_id, const Settings &settings) {
  source_settings_[source_id] = settings;
  states_.erase(source_id);
}

const Settings &MultiSourceTracker::settings_for(int source_id) const {
  auto it = source_settings_.find(source_id);
  return it == source_settings_.end() ? defaults_ : it->second;
}

bool MultiSourceTracker::uses_masks(int source_id) const {
  return settings_for(source_id).tracker.mask_conditioned_association;
}

bool MultiSourceTracker::uses_masks() const {
  if (source_settings_.empty()) return defaults_.tracker.mask_conditioned_association;
  for (const auto &item : source_settings_) {
    if (item.second.tracker.mask_conditioned_association) return true;
  }
  return false;
}

bool MultiSourceTracker::uses_reid(int source_id) const {
  return settings_for(source_id).tracker.with_reid;
}

bool MultiSourceTracker::uses_frames(int source_id) const {
  return settings_for(source_id).cmc_method != CmcMethod::None;
}

void MultiSourceTracker::reset() { states_.clear(); }

void MultiSourceTracker::reset(int source_id) { states_.erase(source_id); }

MultiSourceTracker::SourceState &MultiSourceTracker::state_for(int source_id) {
  SourceState &state = states_[source_id];
  if (!state.tracker) {
    const Settings &settings = settings_for(source_id);
    state.tracker = std::make_unique<BotSortTracker>(settings.resolved_tracker_config());
    if (settings.cmc_method != CmcMethod::None) {
      state.gmc = std::make_unique<GlobalMotionCompensation>(settings.cmc_method,
                                                             settings.cmc_downscale);
    }
  }
  return state;
}

std::vector<unsigned char> MultiSourceTracker::compact_mask(const unsigned char *mask, int height,
                                                            int width, int resolution) {
  std::vector<unsigned char> out;
  if (mask == nullptr || height <= 0 || width <= 0 || resolution <= 0) return out;
  const size_t size = static_cast<size_t>(height) * static_cast<size_t>(width);
  if (std::none_of(mask, mask + size, [](unsigned char v) { return v != 0; })) return out;
  // OpenCV INTER_NEAREST: src = min(floor(dst * src_size / dst_size), src_size - 1).
  const double scale_x = static_cast<double>(width) / resolution;
  const double scale_y = static_cast<double>(height) / resolution;
  std::vector<int> src_x(resolution);
  for (int x = 0; x < resolution; ++x) {
    src_x[x] = std::min(static_cast<int>(std::floor(x * scale_x)), width - 1);
  }
  out.resize(static_cast<size_t>(resolution) * resolution);
  for (int y = 0; y < resolution; ++y) {
    const int sy = std::min(static_cast<int>(std::floor(y * scale_y)), height - 1);
    const unsigned char *row = mask + static_cast<size_t>(sy) * width;
    for (int x = 0; x < resolution; ++x) {
      out[static_cast<size_t>(y) * resolution + x] = row[src_x[x]];
    }
  }
  return out;
}

std::vector<float> MultiSourceTracker::normalized_embedding(const float *embedding, int dim) {
  std::vector<float> out;
  if (embedding == nullptr || dim <= 0) return out;
  double norm = 0.0;
  for (int i = 0; i < dim; ++i) {
    if (!std::isfinite(embedding[i])) return out;
    norm += static_cast<double>(embedding[i]) * embedding[i];
  }
  norm = std::sqrt(norm);
  if (!(norm > 0.0)) return out;  // an all-zero vector carries no appearance
  out.resize(static_cast<size_t>(dim));
  for (int i = 0; i < dim; ++i) out[i] = static_cast<float>(embedding[i] / norm);
  return out;
}

std::pair<int, std::optional<int>> MultiSourceTracker::lineage_for(
    SourceState &state, const Settings &settings, double timestamp, int track_id, double cx,
    double cy, int class_id, const std::vector<int> &current_ids) {
  auto known = state.lineages.find(track_id);
  if (known != state.lineages.end()) return {known->second, std::nullopt};

  std::optional<int> predecessor;
  double best_distance = 0.0;
  if (settings.lineage_enabled) {
    const double max_distance = settings.max_tracking_distance * settings.lineage_distance_scale;
    for (const auto &[candidate_id, observation] : state.history) {
      if (candidate_id == track_id ||
          std::find(current_ids.begin(), current_ids.end(), candidate_id) != current_ids.end()) {
        continue;
      }
      const double gap = timestamp - observation.timestamp;
      if (!(gap >= 0.0 && gap <= settings.lineage_max_seconds)) continue;
      if (observation.class_id != class_id) continue;
      const double dx = cx - observation.cx;
      const double dy = cy - observation.cy;
      const double distance = std::hypot(dx, dy);
      if (distance > max_distance) continue;
      // Reject a successor that moves against the predecessor's direction.
      if (observation.has_velocity && dx * observation.vx + dy * observation.vy < 0.0) continue;
      if (!predecessor || distance < best_distance) {
        predecessor = candidate_id;
        best_distance = distance;
      }
    }
  }
  int lineage = track_id;
  if (predecessor) {
    auto inherited = state.lineages.find(*predecessor);
    if (inherited != state.lineages.end()) lineage = inherited->second;
  }
  state.lineages[track_id] = lineage;
  return {lineage, predecessor};
}

std::vector<Assignment> MultiSourceTracker::update(int source_id, double frame_number,
                                                   const std::vector<DetectionInput> &detections,
                                                   const FrameInput *frame) {
  const Settings &settings = settings_for(source_id);
  SourceState &state = state_for(source_id);
  const bool mask_enabled = settings.tracker.mask_conditioned_association;
  const bool reid_enabled = settings.tracker.with_reid;

  std::vector<Detection> dets;
  dets.reserve(detections.size());
  for (size_t index = 0; index < detections.size(); ++index) {
    const DetectionInput &in = detections[index];
    Detection det{};
    det.left = in.left;
    det.top = in.top;
    det.width = in.width;
    det.height = in.height;
    det.score = in.score;
    det.class_id = in.class_id;
    det.source_index = static_cast<int>(index);
    if (mask_enabled) {
      det.mask = compact_mask(in.mask, in.mask_height, in.mask_width, settings.mask_resolution);
      if (!det.mask.empty()) {
        det.mask_height = settings.mask_resolution;
        det.mask_width = settings.mask_resolution;
      }
    }
    if (reid_enabled) det.embedding = normalized_embedding(in.embedding, in.embedding_dim);
    dets.push_back(std::move(det));
  }

  std::optional<Affine2x3> camera_motion;
  if (state.gmc && frame != nullptr && frame->data != nullptr) {
    camera_motion = state.gmc->apply(*frame);
  }
  const std::vector<TrackedObject> tracked =
      state.tracker->update(dets, camera_motion ? &*camera_motion : nullptr);

  std::vector<int> current_ids;
  current_ids.reserve(tracked.size());
  for (const auto &obj : tracked) current_ids.push_back(obj.track_id);

  std::vector<Assignment> assignments;
  std::vector<bool> used(detections.size(), false);
  for (const auto &obj : tracked) {
    const int index = obj.source_detection_index;
    if (index < 0 || index >= static_cast<int>(detections.size())) continue;
    if (used[index]) continue;
    const DetectionInput &detection = detections[index];
    if (detection.class_id != obj.class_id) continue;
    const double association_iou =
        iou(std::array<double, 4>{obj.left, obj.top, obj.width, obj.height},
            std::array<double, 4>{detection.left, detection.top, detection.width,
                                  detection.height});
    if (settings.enforce_match_iou && association_iou < settings.match_iou) continue;

    const double cx = detection.left + detection.width / 2.0;
    const double cy = detection.top + detection.height / 2.0;
    auto [lineage_id, predecessor] = lineage_for(state, settings, frame_number, obj.track_id, cx,
                                                 cy, detection.class_id, current_ids);
    used[index] = true;
    Assignment assignment{index,
                          obj.track_id,
                          association_iou,
                          obj.association_reason,
                          settings.lineage_enabled ? std::optional<int>(lineage_id)
                                                   : std::nullopt,
                          predecessor};
    assignments.push_back(std::move(assignment));

    auto previous = std::find_if(state.history.begin(), state.history.end(),
                                 [&](const auto &item) { return item.first == obj.track_id; });
    Observation observation;
    observation.timestamp = frame_number;
    observation.cx = cx;
    observation.cy = cy;
    observation.class_id = detection.class_id;
    if (previous != state.history.end()) {
      const double elapsed = std::max(frame_number - previous->second.timestamp, 1e-6);
      observation.has_velocity = true;
      observation.vx = (cx - previous->second.cx) / elapsed;
      observation.vy = (cy - previous->second.cy) / elapsed;
      previous->second = observation;  // keeps its position, like a dict update
    } else {
      state.history.emplace_back(obj.track_id, observation);
    }
  }

  const double cutoff = frame_number - std::max(settings.lineage_max_seconds * 2.0, 1.0);
  state.history.erase(std::remove_if(state.history.begin(), state.history.end(),
                                     [&](const auto &item) {
                                       return item.second.timestamp < cutoff;
                                     }),
                      state.history.end());
  return assignments;
}

}  // namespace botsort_v2
