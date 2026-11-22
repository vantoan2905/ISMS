// A single tracklet: its Kalman state plus BoT-SORT's lifecycle bookkeeping
// (New/unconfirmed -> Tracked -> Lost -> Removed).
#pragma once

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "gmc.hpp"
#include "kalman_filter_xywh.hpp"

namespace botsort_v2 {

enum class TrackState { New, Tracked, Lost, Removed };

// A detection for one frame: bbox is (left, top, width, height), matching
// the Detection.bbox convention already used in src/core/models.py.
struct Detection {
  double left;
  double top;
  double width;
  double height;
  double score;
  int class_id;
  // Compact bbox-local binary mask. Python downsamples the detector mask before
  // crossing the binding so association cost is bounded by mask_resolution^2,
  // not by the source object's pixel area. An empty vector means bbox-only.
  std::vector<unsigned char> mask;
  int mask_width = 0;
  int mask_height = 0;
  // L2-normalized ReID appearance embedding, or empty when the caller has none
  // (or ReID is off). Its dimension only has to match across one source.
  std::vector<float> embedding;
  // Index in the exact input vector passed for this update. It survives every
  // high/low/unconfirmed association stage so Python can address the source
  // detection directly instead of performing a second greedy IoU rematch.
  int source_index = -1;
};

struct TrackedObject {
  int track_id;
  double left, top, width, height;
  double score;
  int class_id;
  bool is_activated;
  int source_detection_index;
  std::string association_reason;
};

class STrack {
 public:
  explicit STrack(const Detection &det) : det_(det), score_(det.score) {}

  // First observation of a brand-new tracklet. `track_id` is assigned by the
  // caller (BotSortTracker owns the id counter) but only becomes "confirmed"
  // (is_activated) immediately when this is the very first frame the
  // tracker has ever seen -- otherwise it stays provisional until it
  // survives to a second frame via update().
  void activate(const KalmanFilterXYWH &kf, int frame_id, int track_id) {
    track_id_ = track_id;
    kf.initiate(tlwh_to_xywh(det_), mean_, covariance_);
    tracklet_len_ = 0;
    state_ = TrackState::Tracked;
    is_activated_ = (frame_id == 1);
    frame_id_ = frame_id;
    start_frame_ = frame_id;
  }

  // Re-associates a Lost track with a new detection, optionally reusing a
  // fresh id (BoT-SORT keeps the same id on re-find, so new_id defaults to
  // false; kept as a parameter for parity with the reference implementation).
  void re_activate(const KalmanFilterXYWH &kf, const Detection &det, int frame_id,
                    const std::string &association_reason = "iou_hungarian",
                    bool new_id = false, int new_track_id = -1) {
    det_ = det;
    kf.update(mean_, covariance_, tlwh_to_xywh(det_));
    tracklet_len_ = 0;
    state_ = TrackState::Tracked;
    is_activated_ = true;
    frame_id_ = frame_id;
    if (new_id) {
      track_id_ = new_track_id;
    }
    score_ = det.score;
    association_reason_ = association_reason;
  }

  void predict(const KalmanFilterXYWH &kf) {
    if (state_ != TrackState::Tracked) {
      // Zero out velocity for non-tracked (still-lost) tracks so a long-lost
      // object doesn't keep coasting on stale velocity, matching BoT-SORT.
      mean_(6) = 0;
      mean_(7) = 0;


    }
    kf.predict(mean_, covariance_);
  }

  void update(const KalmanFilterXYWH &kf, const Detection &det, int frame_id,
              const std::string &association_reason = "iou_hungarian") {
    det_ = det;
    frame_id_ = frame_id;
    tracklet_len_ += 1;
    kf.update(mean_, covariance_, tlwh_to_xywh(det_));
    state_ = TrackState::Tracked;
    is_activated_ = true;
    score_ = det.score;
    association_reason_ = association_reason;
  }

  // BoT-SORT's appearance memory: an exponential moving average of the matched
  // detections' embeddings, re-normalized so cosine distance stays a dot product.
  // An empty embedding leaves the memory untouched; a dimension change restarts it.
  void update_features(const std::vector<float> &embedding, double alpha) {
    if (embedding.empty()) return;
    if (smooth_feat_.size() != embedding.size()) {
      smooth_feat_ = embedding;
      return;
    }
    double norm = 0.0;
    for (size_t i = 0; i < embedding.size(); ++i) {
      smooth_feat_[i] = static_cast<float>(alpha * smooth_feat_[i] + (1.0 - alpha) * embedding[i]);
      norm += static_cast<double>(smooth_feat_[i]) * smooth_feat_[i];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0) {
      for (auto &value : smooth_feat_) value = static_cast<float>(value / norm);
    }
  }

  // Moves the Kalman state with the camera (BoT-SORT's multi_gmc): the 2x2
  // linear part acts on every (x, y)-pair of the state -- position, size and
  // both velocities -- and the translation on the position only.
  void apply_camera_motion(const Affine2x3 &warp) {
    Eigen::Matrix2d linear;
    linear << warp[0], warp[1], warp[3], warp[4];
    KalmanCov rotation = KalmanCov::Zero();
    for (int k = 0; k < 4; ++k) rotation.block<2, 2>(2 * k, 2 * k) = linear;
    mean_ = rotation * mean_;
    mean_(0) += warp[2];
    mean_(1) += warp[5];
    covariance_ = rotation * covariance_ * rotation.transpose();
  }

  void mark_lost() { state_ = TrackState::Lost; }
  void mark_removed() { state_ = TrackState::Removed; }

  // Box of the last detection this track was matched to (not the Kalman state).
  std::array<double, 4> last_observed_tlwh() const {
    return {det_.left, det_.top, det_.width, det_.height};
  }

  // Current bbox as (left, top, width, height), from the Kalman mean once
  // activated, otherwise the raw detection that seeded this tracklet.
  std::array<double, 4> tlwh() const {
    double cx = mean_(0), cy = mean_(1), w = mean_(2), h = mean_(3);
    return {cx - w / 2.0, cy - h / 2.0, w, h};
  }

  int track_id() const { return track_id_; }
  TrackState state() const { return state_; }
  bool is_activated() const { return is_activated_; }
  int frame_id() const { return frame_id_; }
  int end_frame() const { return frame_id_; }
  double score() const { return score_; }
  int class_id() const { return det_.class_id; }
  int source_detection_index() const { return det_.source_index; }
  const std::vector<unsigned char> &mask() const { return det_.mask; }
  int mask_width() const { return det_.mask_width; }
  int mask_height() const { return det_.mask_height; }
  const std::string &association_reason() const { return association_reason_; }
  const std::vector<float> &smooth_feature() const { return smooth_feat_; }

 private:
  static Eigen::Vector4d tlwh_to_xywh(const Detection &det) {
    return Eigen::Vector4d(det.left + det.width / 2.0, det.top + det.height / 2.0, det.width, det.height);
  }

  Detection det_;
  KalmanMean mean_ = KalmanMean::Zero();
  KalmanCov covariance_ = KalmanCov::Zero();

  int track_id_ = -1;
  TrackState state_ = TrackState::New;
  bool is_activated_ = false;
  int frame_id_ = 0;
  int start_frame_ = 0;
  int tracklet_len_ = 0;
  double score_ = 0;
  std::string association_reason_ = "new_track";
  std::vector<float> smooth_feat_;
};

}  // namespace botsort_v2
