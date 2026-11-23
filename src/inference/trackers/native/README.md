# BoT-SORT

The native tracker associates detections across frames and keeps separate tracking state for each video source. Its Python adapter is in `../native_tracker.py`. The default settings are in `../native_tracker.yaml`. The optional `cameras` mapping accepts overrides by camera name.

Run `bash src/inference/trackers/native/build.sh` from the project root. Set `PYTHON` to the interpreter that will import the extension. Set `BOTSORT_CMC=OFF` to build without camera motion compensation. Build products are written to `build/botsort_native/`.

The standalone executable accepts detection rows as CSV and writes track assignments to standard output:

```bash
build/botsort_native/botsort_track src/inference/trackers/native_tracker.yaml < detections.csv > tracks.csv
```

Input columns are source ID, frame number, left, top, width, height, confidence and class ID. Optional trailing columns contain the appearance embedding.
