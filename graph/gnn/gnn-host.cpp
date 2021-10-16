#include <cmath>
#include <cstdint>
#include <cstring>

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ap_int.h>
#include <tapa.h>

#include "gnn.h"

using std::clog;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

vector<Edge> NormalizeEdges(const vector<Edge>& edge_list, Vid vertex_count = 0,
                            Vid min_vid = 0) {
  vector<Vid> in_degrees;
  if (vertex_count != 0) {
    in_degrees.reserve(vertex_count);
  }
  for (auto& edge : edge_list) {
    auto idx = edge.src - min_vid;
    if (idx >= in_degrees.size()) {
      in_degrees.resize(idx + 1, 1);  // initialize to 1 because of self-edges
    }
    ++in_degrees[idx];
  }
  vector<Edge> normalized_edges;
  normalized_edges.reserve(edge_list.size() + in_degrees.size());
  for (auto& edge : edge_list) {
    normalized_edges.push_back({
        edge.src,
        edge.dst,
        edge.attr / sqrtf(in_degrees[edge.src - min_vid]) /
            sqrtf(in_degrees[edge.dst - min_vid]),
    });
  }
  for (Vid vid = min_vid; vid < min_vid + in_degrees.size(); ++vid) {
    normalized_edges.push_back({
        vid,
        vid,
        1.f / (in_degrees[vid - min_vid]),
    });
  }
  return normalized_edges;
}

vector<float> GcnFwdLayer(const vector<float>& in_feats,
                          const aligned_vector<float>& weights,
                          const vector<Edge>& edges, Vid node_count,
                          Eid edge_count, Fid in_feat_len, Fid out_feat_len) {
  vector<float> out_feats(node_count * out_feat_len);

  // feature transformation
  vector<float> xformed_feats(node_count * out_feat_len);
  for (Vid i = 0; i < node_count; ++i) {
    for (Fid j = 0; j < out_feat_len; ++j) {
      xformed_feats[i * out_feat_len + j] = 0;
      for (Fid k = 0; k < in_feat_len; ++k) {
        xformed_feats[i * out_feat_len + j] +=
            in_feats[i * tapa::round_up<float_v::length>(in_feat_len) + k] *
            weights[k * out_feat_len + j];
      }
    }
  }

  // aggregation
  std::fill(out_feats.begin(), out_feats.end(), 0);
  for (Eid i = 0; i < edge_count; ++i) {
    auto& e = edges[i];
    for (Fid j = 0; j < out_feat_len; ++j) {
      out_feats[e.src * out_feat_len + j] +=
          e.attr * xformed_feats[e.dst * out_feat_len + j];
    }
  }

  return out_feats;
}

int main(int argc, char* argv[]) {
  ifstream ifs;
  string dirname = argv[1];
  dirname += '/';
  auto open_file = [&ifs, &dirname](string filename) {
    filename = dirname + filename;
    if (ifs.is_open()) ifs.close();
    ifs.open(filename);
    if (ifs.fail()) {
      std::clog << "failed to open file `" << filename
                << "`: " << std::strerror(errno) << endl;
      exit(1);
    }
  };

  // Load node features
  vector<float> in_feats(node_count *
                         tapa::round_up<float_v::length>(in_feat_len));
  open_file("node_features.txt");
  for (Vid j = 0; j < node_count; ++j) {
    for (Fid i = 0; i < in_feat_len; ++i) {
      ifs >> in_feats[j * tapa::round_up<float_v::length>(in_feat_len) + i];
    }
  }

  // Load weight
  aligned_vector<float> weights(in_feat_len_aligned * out_feat_len);
  open_file("weight_conv1.txt");
  for (Fid j = 0; j < in_feat_len; ++j) {
    for (Fid i = 0; i < out_feat_len; ++i) {
      ifs >> weights[j * out_feat_len + i];
    }
  }

  // Load edges
  vector<Edge> raw_edge_list(edge_count);
  open_file("edge_index.txt");
  for (Eid i = 0; i < edge_count; ++i) {
    ifs >> raw_edge_list[i].dst;
    raw_edge_list[i].attr = 1.f;
  }
  for (Eid i = 0; i < edge_count; ++i) {
    ifs >> raw_edge_list[i].src;
  }

  auto normalized_edges = NormalizeEdges(raw_edge_list, node_count);

  aligned_vector<float> out_feats(node_count * out_feat_len);

  vector<float> out_feats_baseline =
      GcnFwdLayer(in_feats, weights, normalized_edges, node_count,
                  edge_count + node_count, in_feat_len, out_feat_len);
  aligned_vector<float> in_feats_dev(
      node_count * tapa::round_up<float_v::length>(in_feat_len));
  for (Fid k = 0; k < in_feat_len_aligned / float_v::length; ++k) {
    for (Vid i = 0; i < node_count; ++i) {
      for (Fid j = 0; j < float_v::length; ++j) {
        in_feats_dev[(i + k * node_count) * float_v::length + j] =
            in_feats[i * in_feat_len_aligned + k * float_v::length + j];
      }
    }
  }
  std::string bitstream;
  if (const auto bitstream_ptr = getenv("TAPAB")) {
    bitstream = bitstream_ptr;
  }
  tapa::invoke(
      Gnn, bitstream,
      tapa::read_only_mmap<float>(in_feats_dev).reinterpret<bits<float_v>>(),
      tapa::read_only_mmap<float>(weights).reinterpret<bits<float_v>>(),
      tapa::read_only_mmap<Edge>(normalized_edges),
      tapa::write_only_mmap<float>(out_feats).reinterpret<bits<float_v>>());

  int threshold = 10;
  int error_count = 0;
  for (Vid j = 0; j < node_count; ++j) {
    for (Fid i = 0; i < out_feat_len; ++i) {
      float expected = out_feats_baseline[j * out_feat_len + i];
      float actual = out_feats[j * out_feat_len + i];
      if (!(fabs(expected - actual) < 0.0001f)) {
        if (error_count < threshold) {
          clog << "expecting " << expected << ", got " << actual << endl;
        }
        ++error_count;
      }
    }
  }
  if (error_count > threshold) {
    clog << "... and " << (error_count - threshold) << " more" << endl;
  }

  if (error_count) {
    clog << "FAIL!" << endl;
    return 1;
  }

  clog << "PASS!" << endl;
  return 0;
}
