# BoT-SORT v2

The native tracker associates detections across frames and keeps separate tracking state for each video source. Its Python adapter is in `../botsort_hungarian_v2.py`. The default settings are in `../botsort_hungarian_v2.yaml`. The optional `cameras` mapping accepts overrides by camera name.

Run `bash src/inference/trackers/botsort_native_v2/build.sh` from the project root. Set `PYTHON` to the interpreter that will import the extension. Set `BOTSORT_V2_CMC=OFF` to build without camera motion compensation. Build products are written to `build/botsort_native_v2/`.

The standalone executable accepts detection rows as CSV and writes track assignments to standard output:

```bash
build/botsort_native_v2/botsort_v2_track src/inference/trackers/botsort_hungarian_v2.yaml < detections.csv > tracks.csv
```

Input columns are source ID, frame number, left, top, width, height, confidence and class ID. Optional trailing columns contain the appearance embedding.
