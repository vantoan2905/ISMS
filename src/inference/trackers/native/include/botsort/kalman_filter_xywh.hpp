// Constant-velocity Kalman filter over (center_x, center_y, width, height),
// matching BoT-SORT's KalmanFilterXYWH (motion-only, no appearance term).
// Unlike the original SORT/DeepSORT filter (which tracks aspect ratio), this
// regresses width/height directly, which BoT-SORT found more stable.
//
// The filter is stateless: mean/covariance live on each STrack and are
// threaded through initiate/predict/update/project, mirroring the reference
// Python implementation so the two stay easy to compare.
#pragma once

#include <Eigen/Dense>

namespace botsort {

using KalmanMean = Eigen::Matrix<double, 8, 1>;
using KalmanCov = Eigen::Matrix<double, 8, 8>;

class KalmanFilterXYWH {
 public:
  // dt is the time step in frames; the two weights scale the process and
  // measurement noise by the box size (BoT-SORT's 1/20 and 1/160). All three
  // come from the tracker YAML (``kalman`` group).
  explicit KalmanFilterXYWH(double dt = 1.0, double std_weight_position = 1.0 / 20,
                            double std_weight_velocity = 1.0 / 160)
      : dt_(dt),
        std_weight_position_(std_weight_position),
        std_weight_velocity_(std_weight_velocity) {
    motion_mat_.setIdentity();
    for (int i = 0; i < 4; ++i) {
      motion_mat_(i, 4 + i) = dt_;
    }
    update_mat_.setZero();
    for (int i = 0; i < 4; ++i) {
      update_mat_(i, i) = 1.0;
    }
  }

  // measurement: (cx, cy, w, h)
  void initiate(const Eigen::Vector4d &measurement, KalmanMean &mean, KalmanCov &covariance) const {
    mean.setZero();
    mean.head<4>() = measurement;

    Eigen::Matrix<double, 8, 1> std_dev;
    std_dev(0) = 2 * std_weight_position_ * measurement(2);
    std_dev(1) = 2 * std_weight_position_ * measurement(3);
    std_dev(2) = 2 * std_weight_position_ * measurement(2);
    std_dev(3) = 2 * std_weight_position_ * measurement(3);
    std_dev(4) = 10 * std_weight_velocity_ * measurement(2);
    std_dev(5) = 10 * std_weight_velocity_ * measurement(3);
    std_dev(6) = 10 * std_weight_velocity_ * measurement(2);
    std_dev(7) = 10 * std_weight_velocity_ * measurement(3);

    covariance.setZero();
    covariance.diagonal() = std_dev.cwiseProduct(std_dev);
  }

  void predict(KalmanMean &mean, KalmanCov &covariance) const {
    Eigen::Matrix<double, 8, 1> std_dev;
    std_dev(0) = std_weight_position_ * mean(2);
    std_dev(1) = std_weight_position_ * mean(3);
    std_dev(2) = std_weight_position_ * mean(2);
    std_dev(3) = std_weight_position_ * mean(3);
    std_dev(4) = std_weight_velocity_ * mean(2);
    std_dev(5) = std_weight_velocity_ * mean(3);
    std_dev(6) = std_weight_velocity_ * mean(2);
    std_dev(7) = std_weight_velocity_ * mean(3);

    KalmanCov motion_cov = KalmanCov::Zero();
    motion_cov.diagonal() = std_dev.cwiseProduct(std_dev);

    mean = motion_mat_ * mean;
    covariance = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;
  }

  void project(const KalmanMean &mean, const KalmanCov &covariance, Eigen::Vector4d &projected_mean,
               Eigen::Matrix4d &projected_cov) const {
    Eigen::Vector4d std_dev;
    std_dev(0) = std_weight_position_ * mean(2);
    std_dev(1) = std_weight_position_ * mean(3);
    std_dev(2) = std_weight_position_ * mean(2);
    std_dev(3) = std_weight_position_ * mean(3);

    Eigen::Matrix4d innovation_cov = Eigen::Matrix4d::Zero();
    innovation_cov.diagonal() = std_dev.cwiseProduct(std_dev);

    projected_mean = update_mat_ * mean;
    projected_cov = update_mat_ * covariance * update_mat_.transpose() + innovation_cov;
  }

  void update(KalmanMean &mean, KalmanCov &covariance, const Eigen::Vector4d &measurement) const {
    Eigen::Vector4d projected_mean;
    Eigen::Matrix4d projected_cov;
    project(mean, covariance, projected_mean, projected_cov);

    // Kalman gain K = covariance * H^T * projected_cov^-1, solved rather than
    // inverted explicitly for numerical stability.
    Eigen::Matrix<double, 8, 4> kalman_gain =
        (projected_cov.ldlt().solve(update_mat_ * covariance.transpose())).transpose();

    Eigen::Vector4d innovation = measurement - projected_mean;
    mean = mean + kalman_gain * innovation;
    covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();
  }

 private:
  double dt_;
  double std_weight_position_;
  double std_weight_velocity_;
  Eigen::Matrix<double, 8, 8> motion_mat_;
  Eigen::Matrix<double, 4, 8> update_mat_;
};

}  // namespace botsort
