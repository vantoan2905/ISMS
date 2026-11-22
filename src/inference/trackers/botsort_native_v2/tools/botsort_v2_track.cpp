// Standalone v2 tracker run, no Python: CSV detections in, CSV assignments out.
//
//   botsort_v2_track CONFIG.yaml [CAMERA] < detections.csv > tracks.csv
//
// Input rows:  source_id,frame_number,left,top,width,height,score,class_id[,e0,e1,...]
//              (trailing columns, when present, are the detection's ReID
//              embedding, used when the config has with_reid; a header line and blank lines are skipped; rows of one frame
//              and source must be consecutive)
// Output rows: source_id,frame_number,detection_index,track_id,association_score,
//              association_reason,lineage_id,predecessor_track_id
// detection_index counts within the input rows of that frame and source. CAMERA
// selects the config's per-camera overrides for every source; without it the
// config's defaults are used. Masks are not read (bbox-only association), and
// there are no frames, so a cmc_method is ignored (the camera is taken as fixed).
//
// Frame numbers are per-source frame counters, as in the pipeline: a frame with
// no detections has no row, so a gap between a source's consecutive frame numbers
// is replayed as empty frames (the tracker must still predict and age its tracks
// on them). Use an empty row "source_id,frame_number" to mark trailing empty frames.
#include <cstdio>
#include <exception>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "botsort_v2/multi_source_tracker.hpp"
#include "botsort_v2/settings.hpp"

using namespace botsort_v2;

namespace {

struct Row {
  int source_id;
  double frame_number;
  DetectionInput detection;
  std::vector<float> embedding;
};

// Returns 1 for a detection row, 0 for an empty-frame marker, -1 when unparsable.
int parse_row(const std::string &line, Row &row) {
  std::stringstream ss(line);
  std::string cell;
  std::vector<std::string> cells;
  while (std::getline(ss, cell, ',')) cells.push_back(cell);
  if (cells.size() == 2) {
    try {
      row.source_id = std::stoi(cells[0]);
      row.frame_number = std::stod(cells[1]);
    } catch (const std::exception &) {
      return -1;
    }
    return 0;
  }
  if (cells.size() < 8) return -1;
  try {
    row.source_id = std::stoi(cells[0]);
    row.frame_number = std::stod(cells[1]);
    row.detection.left = std::stod(cells[2]);
    row.detection.top = std::stod(cells[3]);
    row.detection.width = std::stod(cells[4]);
    row.detection.height = std::stod(cells[5]);
    row.detection.score = std::stod(cells[6]);
    row.detection.class_id = std::stoi(cells[7]);
    for (size_t i = 8; i < cells.size(); ++i) row.embedding.push_back(std::stof(cells[i]));
  } catch (const std::exception &) {
    return -1;
  }
  return 1;
}

std::string optional_text(const std::optional<int> &value) {
  return value ? std::to_string(*value) : std::string();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: " << argv[0] << " CONFIG.yaml [CAMERA] < detections.csv\n";
    return 2;
  }
  try {
    ConfigFile config = load_config_file(argv[1]);
    Settings settings = argc == 3 ? config.for_camera(argv[2]) : config.defaults;
    if (settings.cmc_method != CmcMethod::None) {
      std::cerr << "note: cmc_method " << cmc_method_name(settings.cmc_method)
                << " needs frames, which CSV input does not carry; running without CMC\n";
    }
    MultiSourceTracker tracker(settings);
    std::cout.precision(12);

    std::cout << "source_id,frame_number,detection_index,track_id,association_score,"
                 "association_reason,lineage_id,predecessor_track_id\n";
    std::vector<DetectionInput> batch;
    std::vector<std::vector<float>> batch_embeddings;  // parallel to batch
    int batch_source = 0;
    double batch_frame = 0.0;
    bool have_batch = false;
    std::map<int, double> last_frame;  // source -> last frame number fed
    auto run = [&](int source, double frame, const std::vector<DetectionInput> &dets) {
      for (const auto &a : tracker.update(source, frame, dets)) {
        std::cout << source << ',' << frame << ',' << a.detection_index << ',' << a.track_id
                  << ',' << a.association_score << ',' << a.association_reason << ','
                  << optional_text(a.lineage_id) << ',' << optional_text(a.predecessor_track_id)
                  << '\n';
      }
    };
    auto flush = [&]() {
      if (!have_batch) return;
      auto previous = last_frame.find(batch_source);
      if (previous != last_frame.end()) {
        if (batch_frame <= previous->second) {
          throw std::invalid_argument("source " + std::to_string(batch_source) +
                                      ": frame numbers must increase");
        }
        for (double frame = previous->second + 1.0; frame < batch_frame; frame += 1.0) {
          run(batch_source, frame, {});
        }
      }
      // Pointers are set only now, once batch_embeddings stops growing.
      for (size_t i = 0; i < batch.size(); ++i) {
        batch[i].embedding = batch_embeddings[i].empty() ? nullptr : batch_embeddings[i].data();
        batch[i].embedding_dim = static_cast<int>(batch_embeddings[i].size());
      }
      run(batch_source, batch_frame, batch);
      last_frame[batch_source] = batch_frame;
      batch.clear();
      batch_embeddings.clear();
      have_batch = false;
    };

    std::string line;
    size_t line_number = 0;
    while (std::getline(std::cin, line)) {
      ++line_number;
      if (line.empty() || line.find_first_not_of(" \t\r") == std::string::npos) continue;
      Row row;
      const int kind = parse_row(line, row);
      if (kind < 0) {
        if (line_number == 1) continue;  // header
        std::cerr << "line " << line_number << ": cannot parse '" << line << "'\n";
        return 1;
      }
      if (have_batch && (row.source_id != batch_source || row.frame_number != batch_frame)) {
        flush();
      }
      batch_source = row.source_id;
      batch_frame = row.frame_number;
      have_batch = true;
      if (kind == 1) {
        batch.push_back(row.detection);
        batch_embeddings.push_back(std::move(row.embedding));
      }
    }
    flush();
  } catch (const std::exception &exc) {
    std::cerr << "error: " << exc.what() << '\n';
    return 1;
  }
  return 0;
}
