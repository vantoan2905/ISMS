// Camera motion compensation (BoT-SORT's GMC): estimate the global affine
// motion between consecutive frames of one source, so the Kalman state of every
// track can be moved with the camera before association.
//
// The estimate is a 2x3 affine warp mapping previous-frame pixel coordinates to
// current-frame ones. Frames must be in the same pixel coordinate system as the
// detection boxes passed to the tracker, or the warp moves tracks to the wrong
// place.
//
// OpenCV is only included by src/gmc.cpp, so this header (and everything that
// includes it) stays OpenCV-free. When the library is built without OpenCV
// (BOTSORT_CMC=OFF), cmc_available() is false and only CmcMethod::None works.
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace botsort {

// Row-major 2x3 affine warp: x' = a*x + b*y + tx, y' = c*x + d*y + ty,
// stored as {a, b, tx, c, d, ty}.
using Affine2x3 = std::array<double, 6>;

inline Affine2x3 identity_affine() { return {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}; }

enum class CmcMethod {
  None,
  // Shi-Tomasi corners tracked with pyramidal Lucas-Kanade, then a RANSAC
  // similarity fit (BoT-SORT's default "sparseOptFlow").
  SparseOptFlow,
  // Dense intensity alignment with a Euclidean motion model (BoT-SORT's "ecc").
  // More robust on low-texture views, much slower.
  Ecc,
};

// "none", "sparse_optflow" or "ecc"; throws std::invalid_argument otherwise.
CmcMethod parse_cmc_method(const std::string &name);
std::string cmc_method_name(CmcMethod method);

// Whether this build links OpenCV (needed by every method except None).
bool cmc_available();

// A borrowed view of one frame: 8-bit, 1 (gray), 3 (BGR) or 4 (BGRA/RGBA)
// channels, rows `row_stride` bytes apart (0 = tightly packed).
struct FrameInput {
  const unsigned char *data = nullptr;
  int height = 0;
  int width = 0;
  int channels = 0;
  std::size_t row_stride = 0;
};

class GlobalMotionCompensation {
 public:
  // downscale: integer factor the frame is shrunk by before estimation (>= 1).
  GlobalMotionCompensation(CmcMethod method, int downscale);
  ~GlobalMotionCompensation();
  GlobalMotionCompensation(GlobalMotionCompensation &&) noexcept;
  GlobalMotionCompensation &operator=(GlobalMotionCompensation &&) noexcept;

  // Warp from the previous frame given to apply() to this one. Identity on the
  // first frame, after reset(), after a frame-size change, and whenever the
  // estimate fails (too few features, ECC not converging).
  Affine2x3 apply(const FrameInput &frame);
  void reset();
  CmcMethod method() const { return method_; }

 private:
  struct Impl;
  CmcMethod method_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace botsort
