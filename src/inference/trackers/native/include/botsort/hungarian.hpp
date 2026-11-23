// Minimal O(n^2 * m) Hungarian algorithm (Kuhn-Munkres) for rectangular cost
// matrices, used to resolve track<->detection assignment. This is the
// classic potentials-based formulation; self-contained, no external deps.
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

namespace botsort {

// Solves min-cost assignment for a cost matrix with `rows` rows and `cols`
// columns (rows may be <= or > cols). Returns, for each row, the assigned
// column index, or -1 if that row was left unassigned (only possible when
// rows > cols, since every row must be matched to a distinct column when
// rows <= cols). This performs a *complete* assignment on the smaller side;
// callers are expected to reject individual pairs whose cost exceeds a
// threshold themselves (see linear_assignment below).
inline std::vector<int> hungarian_solve(const std::vector<std::vector<double>> &cost) {
  int rows = static_cast<int>(cost.size());
  if (rows == 0) {
    return {};
  }
  int cols = static_cast<int>(cost[0].size());
  if (cols == 0) {
    return std::vector<int>(rows, -1);
  }

  bool transposed = rows > cols;
  int n = transposed ? cols : rows;  // n <= m always
  int m = transposed ? rows : cols;

  const double INF = std::numeric_limits<double>::max() / 4;
  std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
  std::vector<int> p(m + 1, 0), way(m + 1, 0);

  auto at = [&](int i, int j) -> double {
    // 1-indexed (i, j) lookup into the (possibly transposed) cost matrix.
    return transposed ? cost[j - 1][i - 1] : cost[i - 1][j - 1];
  };

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, INF);
    std::vector<bool> used(m + 1, false);
    do {
      used[j0] = true;
      int i0 = p[j0];
      double delta = INF;
      int j1 = -1;
      for (int j = 1; j <= m; ++j) {
        if (!used[j]) {
          double cur = at(i0, j) - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  // p[j] (1-indexed column -> 1-indexed row) holds the solved assignment.
  std::vector<int> col_of_row(n, -1);
  for (int j = 1; j <= m; ++j) {
    if (p[j] > 0) {
      col_of_row[p[j] - 1] = j - 1;
    }
  }

  std::vector<int> result;
  if (!transposed) {
    result = col_of_row;  // row -> col, size == rows
    result.resize(rows, -1);
  } else {
    // We solved with detections-as-rows; invert back to tracks-as-rows.
    result.assign(rows, -1);
    for (int detIdx = 0; detIdx < n; ++detIdx) {
      int trackIdx = col_of_row[detIdx];
      if (trackIdx >= 0) {
        result[trackIdx] = detIdx;
      }
    }
  }
  return result;
}

struct AssignmentResult {
  std::vector<std::pair<int, int>> matches;  // (row, col)
  std::vector<int> unmatched_rows;
  std::vector<int> unmatched_cols;
};

// Solves assignment on `cost` (rows x cols) and discards any pair whose cost
// is >= thresh, feeding the corresponding row/col back into the unmatched
// lists. This mirrors the scipy/lap "solve then threshold-filter" pattern
// used by ByteTrack/BoT-SORT's matching.linear_assignment.
inline AssignmentResult linear_assignment(const std::vector<std::vector<double>> &cost,
                                           double thresh) {
  AssignmentResult result;
  int rows = static_cast<int>(cost.size());
  int cols = rows > 0 ? static_cast<int>(cost[0].size()) : 0;

  if (rows == 0 || cols == 0) {
    result.unmatched_rows.resize(rows);
    for (int i = 0; i < rows; ++i) result.unmatched_rows[i] = i;
    result.unmatched_cols.resize(cols);
    for (int j = 0; j < cols; ++j) result.unmatched_cols[j] = j;
    return result;
  }

  std::vector<int> assignment = hungarian_solve(cost);
  std::vector<bool> col_used(cols, false);

  for (int i = 0; i < rows; ++i) {
    int j = assignment[i];
    if (j >= 0 && cost[i][j] < thresh) {
      result.matches.emplace_back(i, j);
      col_used[j] = true;
    } else {
      result.unmatched_rows.push_back(i);
    }
  }
  for (int j = 0; j < cols; ++j) {
    if (!col_used[j]) {
      result.unmatched_cols.push_back(j);
    }
  }
  return result;
}

}  // namespace botsort
