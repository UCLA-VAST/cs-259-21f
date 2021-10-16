#ifndef GCN_H_
#define GCN_H_

#include <cstdint>

#include <ap_int.h>
#include <tapa.h>

// There is a bug in Vitis HLS preventing fully pipelined read/write of struct
// via m_axi; using ap_uint can work-around this problem.
template <typename T>
using bits = ap_uint<tapa::widthof<T>()>;

using Vid = uint16_t;
using Eid = uint16_t;
using Fid = uint16_t;

constexpr int kFloatVecLen = 16;
constexpr int kFloatAccLatency = 4;

const Vid node_count = 2708;
const Eid edge_count = 10556;
const Eid norm_edge_count = edge_count + node_count;
const Fid in_feat_len = 1433;
const Fid in_feat_len_aligned =
    ((in_feat_len - 1) / kFloatVecLen + 1) * kFloatVecLen;
const Fid out_feat_len = 16;

struct Edge {
  Vid src;
  Vid dst;
  float attr;
};

using float_v = tapa::vec_t<float, kFloatVecLen>;

void Gnn(tapa::mmap<bits<float_v>> in_feats, tapa::mmap<bits<float_v>> weights,
         tapa::mmap<Edge> edges, tapa::mmap<bits<float_v>> out_feats);

#endif  // GCN_H_
