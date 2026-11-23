// Motion-only BoT-SORT: Kalman motion model + ByteTrack-style two-stage IoU
// association (high-score detections first, then low-score ones against
// what's left) + an unconfirmed-track stage, all resolved with a Hungarian
// assignment. No camera motion compensation, no ReID appearance term --
// those are separate add-ons in the reference implementation and were
// deliberately left out of this first pass (see conversation).
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "hungarian.hpp"
#include "kalman_filter_xywh.hpp"
#include "strack.hpp"

namespace botsort {

inline double iou(const std::array<double, 4> &a, const std::array<double, 4> &b) {
  double ax1 = a[0], ay1 = a[1], ax2 = a[0] + a[2], ay2 = a[1] + a[3];
  double bx1 = b[0], by1 = b[1], bx2 = b[0] + b[2], by2 = b[1] + b[3];

  double ix1 = std::max(ax1, bx1);
  double iy1 = std::max(ay1, by1);
  double ix2 = std::min(ax2, bx2);
  double iy2 = std::min(ay2, by2);

  double iw = std::max(0.0, ix2 - ix1);
  double ih = std::max(0.0, iy2 - iy1);
  double inter = iw * ih;
  if (inter <= 0.0) return 0.0;

  double area_a = std::max(0.0, a[2]) * std::max(0.0, a[3]);
  double area_b = std::max(0.0, b[2]) * std::max(0.0, b[3]);
  double denom = area_a + area_b - inter;
  return denom <= 0.0 ? 0.0 : inter / denom;
}

inline std::vector<std::vector<double>> iou_distance(const std::vector<STrack *> &tracks,
                                                       const std::vector<Detection> &dets,
                                                       bool lost_use_last_observation = false) {
  constexpr double kIncompatibleCost = 1e6;
  std::vector<std::vector<double>> cost(tracks.size(), std::vector<double>(dets.size()));
  for (size_t i = 0; i < tracks.size(); ++i) {
    auto tlwh = tracks[i]->tlwh();
    for (size_t j = 0; j < dets.size(); ++j) {
      if (tracks[i]->class_id() != dets[j].class_id) {
        cost[i][j] = kIncompatibleCost;
        continue;
      }
      std::array<double, 4> det_box{dets[j].left, dets[j].top, dets[j].width, dets[j].height};
      double overlap = iou(tlwh, det_box);
      // Observation-centric recovery (OC-SORT): a Lost track's Kalman box keeps
      // coasting while nothing updates it, so an object that stopped where it was
      // last seen no longer overlaps its own prediction. Scoring the pair by the
      // better of the predicted and the last observed box keeps it matchable.
      if (lost_use_last_observation && tracks[i]->state() == TrackState::Lost) {
        overlap = std::max(overlap, iou(tracks[i]->last_observed_tlwh(), det_box));
      }
      cost[i][j] = 1.0 - overlap;
    }
  }
  return cost;
}

inline std::vector<std::vector<double>> proximity_distance(
    const std::vector<STrack *> &tracks, const std::vector<Detection> &dets) {
  constexpr double kIncompatibleCost = 1e6;
  std::vector<std::vector<double>> cost(tracks.size(), std::vector<double>(dets.size()));
  for (size_t i = 0; i < tracks.size(); ++i) {
    auto tlwh = tracks[i]->tlwh();
    double track_cx = tlwh[0] + 0.5 * tlwh[2];
    double track_cy = tlwh[1] + 0.5 * tlwh[3];
    double track_area = std::max(0.0, tlwh[2]) * std::max(0.0, tlwh[3]);
    for (size_t j = 0; j < dets.size(); ++j) {
      if (tracks[i]->class_id() != dets[j].class_id) {
        cost[i][j] = kIncompatibleCost;
        continue;
      }
      double det_cx = dets[j].left + 0.5 * dets[j].width;
      double det_cy = dets[j].top + 0.5 * dets[j].height;
      double det_area = std::max(0.0, dets[j].width) * std::max(0.0, dets[j].height);
      double scale = std::sqrt(std::max(1.0, 0.5 * (track_area + det_area)));
      cost[i][j] = std::hypot(det_cx - track_cx, det_cy - track_cy) / scale;
    }
  }
  return cost;
}

// BoT-SORT's appearance fusion: cost = min(IoU distance, appearance distance),
// where the appearance distance is half the cosine distance between the track's
// smoothed embedding and the detection's. The appearance term only counts when
//   * the pair already overlaps (IoU distance <= iou_gate), so a look-alike
//     elsewhere in the frame can never pull an id across the scene, and
//   * it is confident (appearance distance <= appearance_thresh).
// It can therefore only *lower* the cost of an overlapping pair whose look
// matches -- exactly the tie-break two touching objects need. Pairs without an
// embedding on either side, with mismatched dimensions, or of different classes
// keep their IoU cost. `reid_won[i][j]` marks pairs whose cost the appearance
// term set, for the association audit trail.
inline std::vector<std::vector<double>> fuse_appearance(
    const std::vector<STrack *> &tracks, const std::vector<Detection> &dets,
    std::vector<std::vector<double>> cost, double iou_gate, double appearance_thresh,
    std::vector<std::vector<bool>> &reid_won) {
  reid_won.assign(tracks.size(), std::vector<bool>(dets.size(), false));
  for (size_t i = 0; i < tracks.size(); ++i) {
    const std::vector<float> &track_feat = tracks[i]->smooth_feature();
    if (track_feat.empty()) continue;
    for (size_t j = 0; j < dets.size(); ++j) {
      const std::vector<float> &det_feat = dets[j].embedding;
      if (det_feat.size() != track_feat.size()) continue;
      if (cost[i][j] > iou_gate) continue;  // also skips class-incompatible pairs
      double dot = 0.0;
      for (size_t k = 0; k < det_feat.size(); ++k) {
        dot += static_cast<double>(track_feat[k]) * det_feat[k];
      }
      double appearance = std::max(0.0, 1.0 - dot) / 2.0;
      if (appearance > appearance_thresh) continue;
      if (appearance < cost[i][j]) {
        cost[i][j] = appearance;
        reid_won[i][j] = true;
      }
    }
  }
  return cost;
}

struct AssociationMatch {
  int track_index;
  int detection_index;
  std::string reason;
};

struct ConditionedAssignmentResult {
  std::vector<AssociationMatch> matches;
  std::vector<int> unmatched_rows;
  std::vector<int> unmatched_cols;
};

struct MaskPairMetrics {
  bool valid = false;
  double track_coverage = 0.0;
  double detection_precision = 0.0;
  double dice = 0.0;
};

// Per-pair record for every candidate the mask stage looked at, so gate
// thresholds can be chosen from the measured distribution instead of inherited
// from McByte, whose propagated masks come from a VOS model rather than from a
// detector mask stretched onto a Kalman box.
struct MaskGateSample {
  double base_cost;
  double track_coverage;
  double detection_precision;
  double dice;
  double track_score;
  bool isolated;
  bool valid;
  bool accepted;
};

inline bool compact_mask_value(const std::vector<unsigned char> &mask, int width,
                               int height, double nx, double ny) {
  if (mask.empty() || width <= 0 || height <= 0 || nx < 0.0 || nx >= 1.0 ||
      ny < 0.0 || ny >= 1.0) {
    return false;
  }
  int x = std::min(width - 1, std::max(0, static_cast<int>(nx * width)));
  int y = std::min(height - 1, std::max(0, static_cast<int>(ny * height)));
  size_t index = static_cast<size_t>(y * width + x);
  return index < mask.size() && mask[index] != 0;
}

inline MaskPairMetrics compact_mask_metrics(const STrack &track,
                                            const Detection &detection) {
  MaskPairMetrics result;
  if (track.mask().empty() || detection.mask.empty() || track.mask_width() <= 0 ||
      track.mask_height() <= 0 || detection.mask_width <= 0 ||
      detection.mask_height <= 0) {
    return result;
  }
  auto track_box = track.tlwh();
  std::array<double, 4> detection_box{detection.left, detection.top,
                                      detection.width, detection.height};
  if (track_box[2] <= 0.0 || track_box[3] <= 0.0 || detection_box[2] <= 0.0 ||
      detection_box[3] <= 0.0) {
    return result;
  }

  double left = std::max(track_box[0], detection_box[0]);
  double top = std::max(track_box[1], detection_box[1]);
  double right = std::min(track_box[0] + track_box[2], detection_box[0] + detection_box[2]);
  double bottom = std::min(track_box[1] + track_box[3], detection_box[1] + detection_box[3]);
  if (right <= left || bottom <= top) return result;

  size_t track_ones = static_cast<size_t>(
      std::count_if(track.mask().begin(), track.mask().end(),
                    [](unsigned char value) { return value != 0; }));
  size_t detection_ones = static_cast<size_t>(
      std::count_if(detection.mask.begin(), detection.mask.end(),
                    [](unsigned char value) { return value != 0; }));
  if (track_ones == 0 || detection_ones == 0) return result;

  double track_area = track_box[2] * track_box[3] *
      (static_cast<double>(track_ones) / static_cast<double>(track.mask().size()));
  double detection_area = detection_box[2] * detection_box[3] *
      (static_cast<double>(detection_ones) / static_cast<double>(detection.mask.size()));
  if (track_area <= 0.0 || detection_area <= 0.0) return result;

  int sample_width = std::max(track.mask_width(), detection.mask_width);
  int sample_height = std::max(track.mask_height(), detection.mask_height);
  sample_width = std::max(1, std::min(64, sample_width));
  sample_height = std::max(1, std::min(64, sample_height));
  int both = 0;
  for (int y = 0; y < sample_height; ++y) {
    double gy = top + (static_cast<double>(y) + 0.5) * (bottom - top) / sample_height;
    for (int x = 0; x < sample_width; ++x) {
      double gx = left + (static_cast<double>(x) + 0.5) * (right - left) / sample_width;
      double track_nx = (gx - track_box[0]) / track_box[2];
      double track_ny = (gy - track_box[1]) / track_box[3];
      double detection_nx = (gx - detection_box[0]) / detection_box[2];
      double detection_ny = (gy - detection_box[1]) / detection_box[3];
      if (compact_mask_value(track.mask(), track.mask_width(), track.mask_height(),
                             track_nx, track_ny) &&
          compact_mask_value(detection.mask, detection.mask_width,
                             detection.mask_height, detection_nx, detection_ny)) {
        ++both;
      }
    }
  }
  double overlap_area = (right - left) * (bottom - top);
  double intersection = overlap_area * static_cast<double>(both) /
                        static_cast<double>(sample_width * sample_height);
  if (intersection <= 0.0) return result;
  result.track_coverage = std::min(1.0, intersection / track_area);
  result.detection_precision = std::min(1.0, intersection / detection_area);
  result.dice = std::min(1.0, 2.0 * intersection / (track_area + detection_area));
  result.valid = true;
  return result;
}

class BotSortTracker {
 public:
  struct Config {
    double track_high_thresh = 0.6;   // min det score to attempt first-stage match / spawn a track
    double track_low_thresh = 0.1;    // min det score to even attempt second-stage match
    double new_track_thresh = 0.7;    // min score for an unmatched high det to start a new track
    double match_thresh = 0.8;        // first-stage IoU-distance accept threshold
    double second_match_thresh = 0.5; // second-stage (low-score) IoU-distance accept threshold
    double unconfirmed_match_thresh = 0.7;
    double duplicate_iou_thresh = 0.15; // IoU-distance (1-IoU) below which a Lost track counts as a duplicate of a Tracked one (=> IoU > ~0.85)
    int track_buffer = 30;  // frames a Lost track survives before being dropped
    bool mask_conditioned_association = false;
    bool mask_enable_isolated_matching = false;
    double mask_min_track_confidence = 0.6;
    double mask_min_track_coverage = 0.9;
    double mask_min_detection_precision = 0.05;
    double mask_similarity_weight = 0.5;
    // Offline gate calibration only -- the sample log grows with every candidate
    // pair examined, so a live pipeline must never turn this on.
    bool mask_collect_gate_samples = false;
    bool proximity_rescue = false;
    double proximity_max_normalized_distance = 1.0;
    // ReID appearance fusion in the high-score and unconfirmed stages (see
    // fuse_appearance). Embeddings come from the caller; detections without one
    // are matched by IoU alone. Defaults are BoT-SORT's proximity_thresh,
    // appearance_thresh and feature EMA alpha.
    bool with_reid = false;
    double reid_iou_gate = 0.5;
    double reid_appearance_thresh = 0.25;
    double reid_feature_alpha = 0.9;
    // Constant-velocity Kalman filter (see KalmanFilterXYWH).
    double kalman_dt = 1.0;
    double kalman_std_weight_position = 1.0 / 20;
    double kalman_std_weight_velocity = 1.0 / 160;
    bool lost_match_last_observation = false;
    bool lost_in_second_stage = false;
  };

  explicit BotSortTracker(const Config &config)
      : cfg_(config),
        kf_(config.kalman_dt, config.kalman_std_weight_position,
            config.kalman_std_weight_velocity) {}

  const Config &config() const { return cfg_; }

  // camera_motion: the previous-to-current frame warp from GMC, or nullptr when
  // the camera is treated as fixed.
  std::vector<TrackedObject> update(const std::vector<Detection> &detections,
                                    const Affine2x3 *camera_motion = nullptr) {
    frame_id_ += 1;

    std::vector<Detection> high_dets, low_dets;
    for (const auto &det : detections) {
      if (det.score >= cfg_.track_high_thresh) {
        high_dets.push_back(det);
      } else if (det.score >= cfg_.track_low_thresh) {
        low_dets.push_back(det);
      }
    }

    std::vector<STrack *> unconfirmed;
    std::vector<STrack *> tracked_pool_owned;  // previously-confirmed tracked tracks
    for (auto &t : tracked_stracks_) {
      if (t.is_activated()) {
        tracked_pool_owned.push_back(&t);
      } else {
        unconfirmed.push_back(&t);
      }
    }

    std::vector<STrack *> strack_pool = tracked_pool_owned;
    for (auto &t : lost_stracks_) strack_pool.push_back(&t);
    for (auto *t : strack_pool) t->predict(kf_);

    // --- Camera motion compensation: move every predicted track (and the
    // not-yet-predicted unconfirmed ones, as BoT-SORT does) with the camera, so
    // association compares boxes in the current frame's coordinates. ---
    if (camera_motion != nullptr) {
      for (auto *t : strack_pool) t->apply_camera_motion(*camera_motion);
      for (auto *t : unconfirmed) t->apply_camera_motion(*camera_motion);
    }

    // --- Stage 1: strack_pool vs high-score detections ---
    auto dists1 = iou_distance(strack_pool, high_dets, cfg_.lost_match_last_observation);
    std::vector<std::vector<bool>> reid1;
    if (cfg_.with_reid) {
      dists1 = fuse_appearance(strack_pool, high_dets, std::move(dists1), cfg_.reid_iou_gate,
                               cfg_.reid_appearance_thresh, reid1);
    }
    auto assign1 = associate(strack_pool, high_dets, dists1, cfg_.match_thresh);

    std::vector<bool> pool_matched(strack_pool.size(), false);
    std::vector<bool> high_det_matched(high_dets.size(), false);

    for (const auto &match : assign1.matches) {
      int ti = match.track_index;
      int di = match.detection_index;
      accept(strack_pool[ti], high_dets[di], reid_reason(match, reid1), true);
      pool_matched[ti] = true;
      high_det_matched[di] = true;
    }

    // --- Stage 2: remaining Tracked-state pool tracks vs low-score detections ---
    std::vector<STrack *> remain_tracked;
    std::vector<size_t> remain_tracked_pool_index;
    for (size_t i = 0; i < strack_pool.size(); ++i) {
      if (!pool_matched[i] && (strack_pool[i]->state() == TrackState::Tracked ||
                               (cfg_.lost_in_second_stage &&
                                strack_pool[i]->state() == TrackState::Lost))) {
        remain_tracked.push_back(strack_pool[i]);
        remain_tracked_pool_index.push_back(i);
      }
    }
    auto dists2 = iou_distance(remain_tracked, low_dets, cfg_.lost_match_last_observation);
    auto assign2 = associate(remain_tracked, low_dets, dists2,
                             cfg_.second_match_thresh);

    std::vector<bool> remain_tracked_matched(remain_tracked.size(), false);
    for (const auto &match : assign2.matches) {
      int ti = match.track_index;
      int di = match.detection_index;
      // Low-score detections are IoU-only and never feed the appearance memory
      // (BoT-SORT builds them without features): a dim, half-occluded crop would
      // drag the embedding towards whatever occludes it.
      accept(remain_tracked[ti], low_dets[di], match.reason, false);
      remain_tracked_matched[ti] = true;
      pool_matched[remain_tracked_pool_index[ti]] = true;
    }

    for (size_t i = 0; i < remain_tracked.size(); ++i) {
      if (!remain_tracked_matched[i]) {
        remain_tracked[i]->mark_lost();
      }
    }

    // --- Stage 2b (opt-in): proximity rescue for pool tracks the IoU stages
    // left unmatched. Runs before new-track birth so a drifted-but-live id is
    // preferred over a fresh one; only unclaimed high-score detections are
    // eligible, so this can never take a detection away from an IoU match. ---
    if (cfg_.proximity_rescue) {
      std::vector<STrack *> rescue_tracks;
      std::vector<size_t> rescue_pool_index;
      for (size_t i = 0; i < strack_pool.size(); ++i) {
        if (!pool_matched[i]) {
          rescue_tracks.push_back(strack_pool[i]);
          rescue_pool_index.push_back(i);
        }
      }
      std::vector<Detection> rescue_dets;
      std::vector<size_t> rescue_det_index;
      for (size_t j = 0; j < high_dets.size(); ++j) {
        if (!high_det_matched[j]) {
          rescue_dets.push_back(high_dets[j]);
          rescue_det_index.push_back(j);
        }
      }
      if (!rescue_tracks.empty() && !rescue_dets.empty()) {
        auto dists_prox = proximity_distance(rescue_tracks, rescue_dets);
        auto assign_prox = linear_assignment(
            dists_prox, cfg_.proximity_max_normalized_distance);
        for (const auto &match : assign_prox.matches) {
          accept(rescue_tracks[match.first], rescue_dets[match.second], "proximity_rescue", true);
          pool_matched[rescue_pool_index[match.first]] = true;
          high_det_matched[rescue_det_index[match.second]] = true;
        }
      }
    }

    // --- Stage 3: unconfirmed tracks vs still-unmatched high-score detections ---
    std::vector<Detection> remain_high_dets;
    for (size_t j = 0; j < high_dets.size(); ++j) {
      if (!high_det_matched[j]) {
        remain_high_dets.push_back(high_dets[j]);
      }
    }
    auto dists3 = iou_distance(unconfirmed, remain_high_dets);
    std::vector<std::vector<bool>> reid3;
    if (cfg_.with_reid) {
      dists3 = fuse_appearance(unconfirmed, remain_high_dets, std::move(dists3),
                               cfg_.reid_iou_gate, cfg_.reid_appearance_thresh, reid3);
    }
    auto assign3 = associate(unconfirmed, remain_high_dets, dists3,
                             cfg_.unconfirmed_match_thresh);

    std::vector<bool> unconfirmed_matched(unconfirmed.size(), false);
    std::vector<bool> remain_high_matched(remain_high_dets.size(), false);
    for (const auto &match : assign3.matches) {
      int ti = match.track_index;
      int di = match.detection_index;
      accept(unconfirmed[ti], remain_high_dets[di], reid_reason(match, reid3), true);
      unconfirmed_matched[ti] = true;
      remain_high_matched[di] = true;
    }

    for (size_t i = 0; i < unconfirmed.size(); ++i) {
      if (!unconfirmed_matched[i]) {
        unconfirmed[i]->mark_removed();
      }
    }

    // --- New tracks from detections nothing else claimed ---
    std::vector<STrack> new_tracks;
    for (size_t i = 0; i < remain_high_dets.size(); ++i) {
      if (remain_high_matched[i]) continue;
      if (remain_high_dets[i].score < cfg_.new_track_thresh) continue;
      STrack track(remain_high_dets[i]);
      track.activate(kf_, frame_id_, next_id_++);
      if (cfg_.with_reid) track.update_features(remain_high_dets[i].embedding, cfg_.reid_feature_alpha);
      new_tracks.push_back(std::move(track));
    }

    // --- Expire long-lost tracks ---
    for (auto &t : lost_stracks_) {
      if (frame_id_ - t.end_frame() > cfg_.track_buffer) {
        t.mark_removed();
      }
    }

    rebuild_state(new_tracks);
    dedup_tracked_lost();

    std::vector<TrackedObject> output;
    for (auto &t : tracked_stracks_) {
      if (!t.is_activated()) continue;
      auto tlwh = t.tlwh();
      output.push_back(TrackedObject{
          t.track_id(), tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score(),
          t.class_id(), true, t.source_detection_index(), t.association_reason()});
    }
    return output;
  }

  // Diagnostics for gate calibration. Empty unless mask conditioning is on.
  const std::vector<MaskGateSample> &mask_gate_samples() const {
    return mask_samples_;
  }
  void clear_mask_gate_samples() { mask_samples_.clear(); }

 private:
  // Applies one accepted match: a Tracked track is updated, a Lost one is
  // re-activated under its own id. `feed_appearance` adds the detection's
  // embedding to the track's appearance memory (when ReID is on).
  void accept(STrack *track, const Detection &det, const std::string &reason,
              bool feed_appearance) {
    if (track->state() == TrackState::Tracked) {
      track->update(kf_, det, frame_id_, reason);
    } else {
      track->re_activate(kf_, det, frame_id_, reason);
    }
    if (cfg_.with_reid && feed_appearance) {
      track->update_features(det.embedding, cfg_.reid_feature_alpha);
    }
  }

  // "reid" when the appearance term set the cost of an IoU-stage match, so the
  // audit trail shows which ids appearance kept; mask reasons are left alone.
  static std::string reid_reason(const AssociationMatch &match,
                                 const std::vector<std::vector<bool>> &reid_won) {
    if (reid_won.empty() || !reid_won[match.track_index][match.detection_index]) {
      return match.reason;
    }
    if (match.reason == "iou_hungarian" || match.reason == "clear_iou") return "reid";
    return match.reason;
  }

  ConditionedAssignmentResult associate(
      const std::vector<STrack *> &tracks, const std::vector<Detection> &detections,
      const std::vector<std::vector<double>> &base_cost, double threshold) const {
    ConditionedAssignmentResult result;
    int rows = static_cast<int>(tracks.size());
    int cols = static_cast<int>(detections.size());
    if (!cfg_.mask_conditioned_association) {
      auto plain = linear_assignment(base_cost, threshold);
      for (const auto &match : plain.matches) {
        result.matches.push_back(
            AssociationMatch{match.first, match.second, "iou_hungarian"});
      }
      result.unmatched_rows = std::move(plain.unmatched_rows);
      result.unmatched_cols = std::move(plain.unmatched_cols);
      return result;
    }
    if (rows == 0 || cols == 0) {
      for (int row = 0; row < rows; ++row) result.unmatched_rows.push_back(row);
      for (int col = 0; col < cols; ++col) result.unmatched_cols.push_back(col);
      return result;
    }

    std::vector<int> row_candidates(rows, 0), col_candidates(cols, 0);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        if (base_cost[row][col] < threshold) {
          ++row_candidates[row];
          ++col_candidates[col];
        }
      }
    }

    std::vector<bool> locked_rows(rows, false), locked_cols(cols, false);
    for (int row = 0; row < rows; ++row) {
      if (row_candidates[row] != 1) continue;
      for (int col = 0; col < cols; ++col) {
        if (base_cost[row][col] < threshold && col_candidates[col] == 1) {
          result.matches.push_back(AssociationMatch{row, col, "clear_iou"});
          locked_rows[row] = true;
          locked_cols[col] = true;
          break;
        }
      }
    }

    std::vector<int> remaining_rows, remaining_cols;
    for (int row = 0; row < rows; ++row) {
      if (!locked_rows[row]) remaining_rows.push_back(row);
    }
    for (int col = 0; col < cols; ++col) {
      if (!locked_cols[col]) remaining_cols.push_back(col);
    }
    if (remaining_rows.empty() || remaining_cols.empty()) {
      result.unmatched_rows = remaining_rows;
      result.unmatched_cols = remaining_cols;
      return result;
    }

    std::vector<std::vector<double>> conditioned(
        remaining_rows.size(), std::vector<double>(remaining_cols.size(), 0.0));
    std::vector<std::vector<bool>> mask_boosted(
        remaining_rows.size(), std::vector<bool>(remaining_cols.size(), false));
    std::vector<std::vector<bool>> mask_isolated(
        remaining_rows.size(), std::vector<bool>(remaining_cols.size(), false));
    for (size_t local_row = 0; local_row < remaining_rows.size(); ++local_row) {
      int row = remaining_rows[local_row];
      for (size_t local_col = 0; local_col < remaining_cols.size(); ++local_col) {
        int col = remaining_cols[local_col];
        double cost = base_cost[row][col];
        bool ambiguous = row_candidates[row] > 1 || col_candidates[col] > 1;
        // McByte also exposes an opt-in rescue for a mutually isolated pair
        // just outside the IoU gate. Keep it separate from ordinary ambiguity
        // handling because it changes recall and needs camera-level validation.
        bool isolated = cfg_.mask_enable_isolated_matching &&
                        row_candidates[row] == 0 && col_candidates[col] == 0;
        if (((ambiguous && cost < threshold) || isolated) &&
            tracks[row]->score() >= cfg_.mask_min_track_confidence) {
          MaskPairMetrics metrics = compact_mask_metrics(*tracks[row], detections[col]);
          bool accepted = metrics.valid &&
                          metrics.track_coverage >= cfg_.mask_min_track_coverage &&
                          metrics.detection_precision >= cfg_.mask_min_detection_precision;
          if (accepted) {
            // Keep the bonus additive (and allow a negative cost). A pair of
            // identical boxes already has IoU cost 0, which is precisely the
            // ambiguous case where masks must still be able to break the tie.
            cost -= cfg_.mask_similarity_weight * metrics.dice;
            mask_boosted[local_row][local_col] = true;
            mask_isolated[local_row][local_col] = isolated;
          }
          if (cfg_.mask_collect_gate_samples) {
            mask_samples_.push_back(MaskGateSample{
                base_cost[row][col], metrics.track_coverage,
                metrics.detection_precision, metrics.dice, tracks[row]->score(),
                isolated, metrics.valid, accepted});
          }
        }
        conditioned[local_row][local_col] = cost;
      }
    }

    auto reduced = linear_assignment(conditioned, threshold);
    for (const auto &match : reduced.matches) {
      int local_row = match.first;
      int local_col = match.second;
      const char *reason = "iou_hungarian";
      if (mask_boosted[local_row][local_col]) {
        // Separate the two mask paths in the audit trail: an ambiguity tie-break
        // keeps a pair the IoU gate already admitted, whereas an isolated rescue
        // creates a match the IoU gate rejected outright. Only the latter can
        // manufacture an identity chain, so counters must be able to tell them
        // apart before trusting an inferred placement.
        reason = mask_isolated[local_row][local_col] ? "mask_isolated"
                                                     : "mask_conditioned";
      }
      result.matches.push_back(AssociationMatch{
          remaining_rows[local_row], remaining_cols[local_col], reason});
    }
    for (int local_row : reduced.unmatched_rows) {
      result.unmatched_rows.push_back(remaining_rows[local_row]);
    }
    for (int local_col : reduced.unmatched_cols) {
      result.unmatched_cols.push_back(remaining_cols[local_col]);
    }
    return result;
  }

  // Re-derives tracked_stracks_/lost_stracks_ for the next frame purely from
  // each track's current state -- every mutation this frame (update,
  // re_activate, mark_lost, mark_removed) already happened in place via the
  // strack_pool/unconfirmed pointers, which alias into these same vectors.
  void rebuild_state(std::vector<STrack> &new_tracks) {
    std::unordered_map<int, STrack *> by_id;
    for (auto &t : tracked_stracks_) by_id[t.track_id()] = &t;
    for (auto &t : lost_stracks_) by_id[t.track_id()] = &t;

    std::vector<STrack> next_tracked, next_lost;
    for (auto &[id, t] : by_id) {
      if (t->state() == TrackState::Tracked) {
        next_tracked.push_back(*t);
      } else if (t->state() == TrackState::Lost) {
        next_lost.push_back(*t);
      }
      // Removed (or still-New/unconfirmed-but-not-reactivated) tracks are
      // dropped by simply not being copied into either vector.
    }
    for (auto &t : new_tracks) next_tracked.push_back(t);

    tracked_stracks_ = std::move(next_tracked);
    lost_stracks_ = std::move(next_lost);
  }

  // Drops a Lost track only when it is a *genuine* duplicate of a live Tracked
  // one -- i.e. they overlap so heavily they must be the same object -- matching
  // BoT-SORT's remove_duplicate_stracks. duplicate_iou_thresh is an IoU-DISTANCE
  // threshold (pdist = 1 - IoU), so a duplicate means IoU > 1 - thresh (~0.85),
  // not IoU > thresh. Using it as a raw IoU threshold (the previous behaviour)
  // deleted any Lost track that merely brushed a Tracked box, which is exactly
  // what happens during a temporary occlusion: the occluded object's Lost track
  // was erased before it could be re-activated, so its id was lost forever and
  // it came back with a fresh (larger) id once the objects separated.
  void dedup_tracked_lost() {
    for (size_t i = 0; i < tracked_stracks_.size(); ++i) {
      for (size_t j = 0; j < lost_stracks_.size(); ++j) {
        if (tracked_stracks_[i].class_id() != lost_stracks_[j].class_id()) continue;
        double dist = 1.0 - iou(tracked_stracks_[i].tlwh(), lost_stracks_[j].tlwh());
        if (dist >= cfg_.duplicate_iou_thresh) continue;  // not a true duplicate; keep the Lost track
        // Lost track is stale by definition next to a live Tracked one; drop it.
        lost_stracks_.erase(lost_stracks_.begin() + j);
        --j;
      }
    }
  }

  Config cfg_;
  KalmanFilterXYWH kf_;
  std::vector<STrack> tracked_stracks_;
  std::vector<STrack> lost_stracks_;
  int frame_id_ = 0;
  int next_id_ = 1;
  // associate() is const because it derives an assignment rather than mutating
  // track state; the sample log is pure instrumentation, hence mutable.
  mutable std::vector<MaskGateSample> mask_samples_;
};

}  // namespace botsort
