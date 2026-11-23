# ISMS tracker components

This directory contains optional object tracking implementations and their
registry for the Industrial Safety Monitoring System. It includes centroid,
IoU, passthrough, assignment-based, mask-based and optional native trackers.
Tracking state is separated by source. The current ISMS pose pipeline uses the
separate centroid tracker in `src/isms/tracking/person_tracker.py`.

## Components

| Module | Purpose |
|---|---|
| `tracker_base.py`, `tracker_config.py` | Tracker interfaces, assignments and defaults |
| `registry.py` | Lookup of registered implementations |
| `centroid_tracker.py`, `iou_tracker.py` | Centroid and overlap association |
| `passthrough_tracker.py` | Use IDs supplied by an upstream component |
| `association_tracker.py` | Assignment-based multi-object association |
| `mask_tracker.py` | Mask-aware tracking |
| `native_tracker.py`, `native_tracker.yaml` | Python adapter and configuration for the optional native implementation |
| `native/` | Native implementation and build instructions |

The native tracker is registered as `botsort_native`. Some names refer to
established tracking algorithms and do not assert ownership of those algorithms.

## License and commercial use

The [restricted-use notice](LICENSE) applies to original material in this
directory only to the extent that its copyright holder has the authority to
license it. Copying, modification, redistribution and commercial use require a
written license from the relevant copyright holder, except where permitted by
applicable law or an applicable third-party license. Contact the repository
owner at **vantoan0807s@gmail.com** to request permission.

The optional native build obtains Eigen, yaml-cpp, pybind11 and, in some
configurations, OpenCV. These and other dependencies have their own terms.
Confirm the origin and applicable licenses of any third-party source before
redistributing or using the tracker commercially. This directory-level notice
does not replace third-party notices or license terms.

The restrictions in this directory do not by themselves apply to the rest of
the repository. The notice is not an OSI-approved open-source license.
