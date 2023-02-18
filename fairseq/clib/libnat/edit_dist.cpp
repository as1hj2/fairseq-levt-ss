/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h> // @manual=//caffe2:torch_extension
#include <algorithm>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>
#include <limits>

using namespace ::std;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

vector<vector<uint32_t>> edit_distance2_with_dp(
    vector<uint32_t>& x,
    vector<uint32_t>& y) {
  uint32_t lx = x.size();
  uint32_t ly = y.size();
  vector<vector<uint32_t>> d(lx + 1, vector<uint32_t>(ly + 1));
  for (uint32_t i = 0; i < lx + 1; i++) {
    d[i][0] = i;
  }
  for (uint32_t j = 0; j < ly + 1; j++) {
    d[0][j] = j;
  }
  for (uint32_t i = 1; i < lx + 1; i++) {
    for (uint32_t j = 1; j < ly + 1; j++) {
      d[i][j] =
          min(min(d[i - 1][j], d[i][j - 1]) + 1,
              d[i - 1][j - 1] + 2 * (x.at(i - 1) == y.at(j - 1) ? 0 : 1));
    }
  }
  return d;
}

pair<vector<vector<uint32_t> >, vector<vector<float> > > pld_len_dp(
    vector<vector<float> > &P, // L-1 * K+1, because 0~K
    int32_t L,               // initial len
    int32_t M,               // obj len
    int32_t K)               // max pld len
{ 
    vector<vector<uint32_t> > predecessor(L, vector<uint32_t>(M - L + 1));
    vector<vector<float> > cost(L, vector<float>(M - L + 1, std::numeric_limits<float>::max()));  // cumulative
    pair<vector<vector<uint32_t> >, vector<vector<float> > > d(predecessor, cost);

    // row 1 can only come from (0,0)
    for (uint32_t j = MAX(0, M - L - (L - 1 - 1) * K); j <= MIN(M - L, K); j++) {
        d.first[1][j] = 0; 
        d.second[1][j] = P[0][j];
        // printf("row1: j=%u bestpre=0 bestcost=%f\n", j, P[0][j]);
    }

    for (int32_t i = 2; i < L; i++)
    {
        for (int32_t j = MAX(0, M - L - (L - 1 - i) * K); j <= MIN(M - L, i * K); j++)
        {
            uint32_t best_pre = 0;
            float best_cost = std::numeric_limits<float>::max();
            for (uint32_t pre = MAX(0, j - K); pre <= j; pre++)
            {
                if (d.second[i - 1][pre] + P[i - 1][j - pre] < best_cost)
                {
                    best_pre = pre;
                    best_cost = d.second[i - 1][pre] + P[i - 1][j - pre];
                }
            }
            d.first[i][j] = best_pre;
            d.second[i][j] = best_cost;
            // printf("row%u: j=%u bestpre=%u bestcost=%f\n", i, j, best_pre, best_cost);
        }
    }
    return d;
}

vector<vector<uint32_t>> edit_distance2_backtracking(
    vector<vector<uint32_t>>& d,
    vector<uint32_t>& x,
    vector<uint32_t>& y,
    uint32_t terminal_symbol) {
  vector<uint32_t> seq;
  vector<vector<uint32_t>> edit_seqs(x.size() + 2, vector<uint32_t>());
  /*
  edit_seqs:
  0~x.size() cell is the insertion sequences
  last cell is the delete sequence
  */

  if (x.size() == 0) {
    edit_seqs.at(0) = y;
    return edit_seqs;
  }

  uint32_t i = d.size() - 1;
  uint32_t j = d.at(0).size() - 1;

  while ((i >= 0) && (j >= 0)) {
    if ((i == 0) && (j == 0)) {
      break;
    }

    if ((j > 0) && (d.at(i).at(j - 1) < d.at(i).at(j))) {
      seq.push_back(1); // insert
      seq.push_back(y.at(j - 1));
      j--;
    } else if ((i > 0) && (d.at(i - 1).at(j) < d.at(i).at(j))) {
      seq.push_back(2); // delete
      seq.push_back(x.at(i - 1));
      i--;
    } else {
      seq.push_back(3); // keep
      seq.push_back(x.at(i - 1));
      i--;
      j--;
    }
  }

  uint32_t prev_op, op, s, word;
  prev_op = 0, s = 0;
  for (uint32_t k = 0; k < seq.size() / 2; k++) {
    op = seq.at(seq.size() - 2 * k - 2);
    word = seq.at(seq.size() - 2 * k - 1);
    if (prev_op != 1) {
      s++;
    }
    if (op == 1) // insert
    {
      edit_seqs.at(s - 1).push_back(word);
    } else if (op == 2) // delete
    {
      edit_seqs.at(x.size() + 1).push_back(1);
    } else {
      edit_seqs.at(x.size() + 1).push_back(0);
    }

    prev_op = op;
  }

  for (uint32_t k = 0; k < edit_seqs.size(); k++) {
    if (edit_seqs[k].size() == 0) {
      edit_seqs[k].push_back(terminal_symbol);
    }
  }
  return edit_seqs;
}

pair<vector<uint32_t>, vector<float> > pld_len_backtracking(
    pair<vector<vector<uint32_t> >, vector<vector<float> > > &d,
    int32_t L,
    int32_t M,
    int32_t K,
    uint32_t padded_len)
{

    // L-1 positions that need pld pred
    pair<vector<uint32_t>, vector<float> > path(vector<uint32_t>(padded_len, 0), vector<float>(padded_len, 0));

    uint32_t chosen = M - L;
    for (uint32_t i = L - 1; i > 0; i--)
    {
        path.first[i-1] = chosen - d.first[i][chosen];  // current len - predecessor
        path.second[i-1] = d.second[i][chosen];  // current cumulative cost
        chosen = d.first[i][chosen];  // move to predecessor
    }

    return path;
}

vector<vector<uint32_t>> edit_distance2_backtracking_with_delete(
    vector<vector<uint32_t>>& d,
    vector<uint32_t>& x,
    vector<uint32_t>& y,
    uint32_t terminal_symbol,
    uint32_t deletion_symbol) {
  vector<uint32_t> seq;
  vector<vector<uint32_t>> edit_seqs(x.size() + 1, vector<uint32_t>());
  /*
  edit_seqs:
  0~x.size() cell is the insertion sequences
  last cell is the delete sequence
  */

  if (x.size() == 0) {
    edit_seqs.at(0) = y;
    return edit_seqs;
  }

  uint32_t i = d.size() - 1;
  uint32_t j = d.at(0).size() - 1;

  while ((i >= 0) && (j >= 0)) {
    if ((i == 0) && (j == 0)) {
      break;
    }

    if ((j > 0) && (d.at(i).at(j - 1) < d.at(i).at(j))) {
      seq.push_back(1); // insert
      seq.push_back(y.at(j - 1));
      j--;
    } else if ((i > 0) && (d.at(i - 1).at(j) < d.at(i).at(j))) {
      seq.push_back(2); // delete
      seq.push_back(x.at(i - 1));
      i--;
    } else {
      seq.push_back(3); // keep
      seq.push_back(x.at(i - 1));
      i--;
      j--;
    }
  }

  uint32_t prev_op, op, s, word;
  prev_op = 0, s = 0;
  for (uint32_t k = 0; k < seq.size() / 2; k++) {
    op = seq.at(seq.size() - 2 * k - 2);
    word = seq.at(seq.size() - 2 * k - 1);
    if (prev_op != 1) {
      s++;
    }
    if (op == 1) // insert
    {
      edit_seqs.at(s - 1).push_back(word);
    } else if (op == 2) // delete
    {
      edit_seqs.at(s - 1).push_back(deletion_symbol);
    }

    prev_op = op;
  }

  for (uint32_t k = 0; k < edit_seqs.size(); k++) {
    if (edit_seqs.at(k).size() == 0) {
      edit_seqs.at(k).push_back(terminal_symbol);
    }
  }
  return edit_seqs;
}

vector<uint32_t> compute_ed2(
    vector<vector<uint32_t>>& xs,
    vector<vector<uint32_t>>& ys) {
  vector<uint32_t> distances(xs.size());
  for (uint32_t i = 0; i < xs.size(); i++) {
    vector<vector<uint32_t>> d = edit_distance2_with_dp(xs.at(i), ys.at(i));
    distances.at(i) = d.at(xs.at(i).size()).at(ys.at(i).size());
  }
  return distances;
}

vector<vector<vector<uint32_t>>> suggested_ed2_path(
    vector<vector<uint32_t>>& xs,
    vector<vector<uint32_t>>& ys,
    uint32_t terminal_symbol) {
  vector<vector<vector<uint32_t>>> seq(xs.size());
  for (uint32_t i = 0; i < xs.size(); i++) {
    vector<vector<uint32_t>> d = edit_distance2_with_dp(xs.at(i), ys.at(i));
    seq.at(i) =
        edit_distance2_backtracking(d, xs.at(i), ys.at(i), terminal_symbol);
  }
  return seq;
}

vector<vector<vector<uint32_t>>> suggested_ed2_path_with_delete(
    vector<vector<uint32_t>>& xs,
    vector<vector<uint32_t>>& ys,
    uint32_t terminal_symbol,
    uint32_t deletion_symbol) {
  vector<vector<vector<uint32_t>>> seq(xs.size());
  for (uint32_t i = 0; i < xs.size(); i++) {
    vector<vector<uint32_t>> d = edit_distance2_with_dp(xs.at(i), ys.at(i));
    seq.at(i) = edit_distance2_backtracking_with_delete(
        d, xs.at(i), ys.at(i), terminal_symbol, deletion_symbol);
  }
  return seq;
}

pair<vector<vector<uint32_t> >, vector<vector<float> > > suggested_pld_len(
    vector<vector<vector<float> > > &Ps,
    vector<int32_t> Ls,
    vector<int32_t> Ms,
    int32_t K )
{

    // vector<vector<uint32_t> > predecessor(Ps.size(), vector<uint32_t>(Ls - 1));
    // vector<vector<float> > cost(Ps.size(), vector<float>(Ls - 1));

    pair<vector<vector<uint32_t> >, vector<vector<float> > > seq;
    uint32_t padded_len = Ps[0].size();

    for (uint32_t i = 0; i < Ps.size(); i++)
    {
        // printf("new sample\n");
        if (Ls[i] >= Ms[i]) {
            seq.first.push_back(vector<uint32_t>(padded_len, 0));
            seq.second.push_back(vector<float>(padded_len, 0));
        }
        else {
          pair<vector<vector<uint32_t> >, vector<vector<float> > > d = pld_len_dp(Ps.at(i), Ls.at(i), Ms.at(i), K);
          pair<vector<uint32_t>, vector<float> > path = pld_len_backtracking(d, Ls.at(i), Ms.at(i), K, padded_len);
          seq.first.push_back(path.first);
          seq.second.push_back(path.second);
        }
    }
    return seq;
}

PYBIND11_MODULE(libnat, m) {
  m.def("compute_ed2", &compute_ed2, "compute_ed2");
  m.def("suggested_ed2_path", &suggested_ed2_path, "suggested_ed2_path");
  m.def(
      "suggested_ed2_path_with_delete",
      &suggested_ed2_path_with_delete,
      "suggested_ed2_path_with_delete");

  m.def("suggested_pld_len", &suggested_pld_len, "suggested_pld_len");
}
