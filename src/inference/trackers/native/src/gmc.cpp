#include "botsort/gmc.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef BOTSORT_HAS_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#endif

namespace botsort {

CmcMethod parse_cmc_method(const std::string &name) {
  if (name == "none") return CmcMethod::None;
  if (name == "sparse_optflow") return CmcMethod::SparseOptFlow;
  if (name == "ecc") return CmcMethod::Ecc;
  throw std::invalid_argument("cmc_method must be one of none, sparse_optflow, ecc; got '" +
                              name + "'");
}

std::string cmc_method_name(CmcMethod method) {
  switch (method) {
    case CmcMethod::SparseOptFlow:
      return "sparse_optflow";
    case CmcMethod::Ecc:
      return "ecc";
    default:
      return "none";
  }
}

#ifdef BOTSORT_HAS_OPENCV

bool cmc_available() { return true; }

struct GlobalMotionCompensation::Impl {
  int downscale = 1;
  bool initialized = false;
  cv::Size full_size;
  cv::Mat prev_frame;                   // downscaled gray
  std::vector<cv::Point2f> prev_points; // sparse_optflow only

  cv::Mat prepare(const FrameInput &frame, bool blur) const {
    const int type = frame.channels == 1   ? CV_8UC1
                     : frame.channels == 3 ? CV_8UC3
                                           : CV_8UC4;
    const std::size_t step = frame.row_stride != 0
                                 ? frame.row_stride
                                 : static_cast<std::size_t>(frame.width) * frame.channels;
    cv::Mat view(frame.height, frame.width, type, const_cast<unsigned char *>(frame.data), step);
    cv::Mat gray;
    if (frame.channels == 1) {
      gray = view.clone();
    } else if (frame.channels == 3) {
      cv::cvtColor(view, gray, cv::COLOR_BGR2GRAY);
    } else {
      // BGRA vs RGBA only changes the luma weights, which motion does not care about.
      cv::cvtColor(view, gray, cv::COLOR_BGRA2GRAY);
    }
    if (blur) cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.5);
    if (downscale > 1) {
      cv::resize(gray, gray, cv::Size(frame.width / downscale, frame.height / downscale));
    }
    return gray;
  }

  // Undo the downscale on the translation (the linear part is scale-invariant).
  Affine2x3 to_full_resolution(const cv::Mat &warp) const {
    Affine2x3 out{warp.at<double>(0, 0), warp.at<double>(0, 1), warp.at<double>(0, 2),
                  warp.at<double>(1, 0), warp.at<double>(1, 1), warp.at<double>(1, 2)};
    out[2] *= downscale;
    out[5] *= downscale;
    return out;
  }

  static std::vector<cv::Point2f> corners(const cv::Mat &gray) {
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(gray, points, /*maxCorners=*/1000, /*qualityLevel=*/0.01,
                            /*minDistance=*/1, cv::noArray(), /*blockSize=*/3,
                            /*useHarrisDetector=*/false, /*k=*/0.04);
    return points;
  }

  Affine2x3 sparse_optflow(const cv::Mat &gray) {
    std::vector<cv::Point2f> points = corners(gray);
    Affine2x3 result = identity_affine();
    if (initialized && !prev_points.empty()) {
      std::vector<cv::Point2f> moved;
      std::vector<unsigned char> status;
      std::vector<float> error;
      cv::calcOpticalFlowPyrLK(prev_frame, gray, prev_points, moved, status, error);
      std::vector<cv::Point2f> from, to;
      for (std::size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
          from.push_back(prev_points[i]);
          to.push_back(moved[i]);
        }
      }
      if (from.size() > 4) {
        cv::Mat warp = cv::estimateAffinePartial2D(from, to, cv::noArray(), cv::RANSAC);
        if (!warp.empty()) result = to_full_resolution(warp);
      }
    }
    prev_frame = gray;
    prev_points = std::move(points);
    initialized = true;
    return result;
  }

  Affine2x3 ecc(const cv::Mat &gray) {
    Affine2x3 result = identity_affine();
    if (initialized) {
      cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
      try {
        cv::findTransformECC(prev_frame, gray, warp, cv::MOTION_EUCLIDEAN,
                             cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                                              5000, 1e-6),
                             cv::noArray(), 1);
        cv::Mat warp64;
        warp.convertTo(warp64, CV_64F);
        result = to_full_resolution(warp64);
      } catch (const cv::Exception &) {
        // Not converged: keep identity, like BoT-SORT.
      }
    }
    prev_frame = gray;
    initialized = true;
    return result;
  }
};

GlobalMotionCompensation::GlobalMotionCompensation(CmcMethod method, int downscale)
    : method_(method), impl_(std::make_unique<Impl>()) {
  impl_->downscale = std::max(1, downscale);
}

Affine2x3 GlobalMotionCompensation::apply(const FrameInput &frame) {
  if (method_ == CmcMethod::None) return identity_affine();
  if (frame.data == nullptr || frame.height <= 0 || frame.width <= 0 ||
      (frame.channels != 1 && frame.channels != 3 && frame.channels != 4)) {
    throw std::invalid_argument("CMC frame must be 8-bit with 1, 3 or 4 channels");
  }
  const cv::Size size(frame.width, frame.height);
  if (impl_->initialized && size != impl_->full_size) reset();
  impl_->full_size = size;
  try {
    cv::Mat gray = impl_->prepare(frame, method_ == CmcMethod::Ecc);
    if (gray.cols < 8 || gray.rows < 8) return identity_affine();
    return method_ == CmcMethod::Ecc ? impl_->ecc(gray) : impl_->sparse_optflow(gray);
  } catch (const cv::Exception &) {
    // Never let a bad frame take the tracker down; restart from the next one.
    reset();
    return identity_affine();
  }
}

void GlobalMotionCompensation::reset() {
  impl_->initialized = false;
  impl_->prev_frame.release();
  impl_->prev_points.clear();
}

#else  // !BOTSORT_HAS_OPENCV

bool cmc_available() { return false; }

struct GlobalMotionCompensation::Impl {};

GlobalMotionCompensation::GlobalMotionCompensation(CmcMethod method, int /*downscale*/)
    : method_(method), impl_(std::make_unique<Impl>()) {
  if (method != CmcMethod::None) {
    throw std::invalid_argument("cmc_method '" + cmc_method_name(method) +
                                "' needs OpenCV; this build has BOTSORT_CMC=OFF");
  }
}

Affine2x3 GlobalMotionCompensation::apply(const FrameInput &) { return identity_affine(); }

void GlobalMotionCompensation::reset() {}

#endif

GlobalMotionCompensation::~GlobalMotionCompensation() = default;
GlobalMotionCompensation::GlobalMotionCompensation(GlobalMotionCompensation &&) noexcept = default;
GlobalMotionCompensation &GlobalMotionCompensation::operator=(GlobalMotionCompensation &&) noexcept =
    default;

}  // namespace botsort
