#include <ap_int.h>
#include <tapa.h>

#include "gnn.h"

const int in_feat_len_vec = in_feat_len_aligned / float_v::length;

// There is a bug in Vitis HLS preventing fully pipelined read/write of struct
// via m_axi; using ap_uint can work-around this problem.
void WeightLoader(tapa::mmap<bits<float_v>> weights,
                  tapa::ostreams<float_v, float_v::length>& weight_q) {
load_weights:
  for (Fid i = 0; i < in_feat_len_aligned; ++i) {
#pragma HLS pipeline II = 1
    weight_q[i % float_v::length].write(tapa::bit_cast<float_v>(weights[i]));
  }
}

void NodeLoader(tapa::mmap<bits<float_v>> in_feats,
                tapa::ostreams<float, float_v::length>& in_feat_q) {
load_nodes:
  for (Fid k = 0; k < in_feat_len_vec; ++k) {
    for (Vid i = 0; i < node_count; ++i) {
#pragma HLS pipeline II = 1
      const auto in_feat_v =
          tapa::bit_cast<float_v>(in_feats[k * node_count + i]);
      for (int j = 0; j < float_v::length; ++j) {
        in_feat_q[j].write(in_feat_v[j]);
      }
    }
  }
}

void MacUnit(tapa::istream<float_v>& weight_q, tapa::istream<float>& in_feat_q,
             tapa::ostream<float_v>& out_feat_q) {
  float_v weights[in_feat_len_vec];
recv_weights:
  for (Fid i = 0; i < in_feat_len_vec; ++i) {
#pragma HLS pipeline II = 1
    weights[i] = weight_q.read();
  }

gemm:
  for (Fid k = 0; k < in_feat_len_vec; ++k) {
    for (Vid i = 0; i < node_count; ++i) {
#pragma HLS pipeline II = 1
      out_feat_q.write(in_feat_q.read() * weights[k]);
    }
  }
}

void Aggregator(tapa::mmap<Edge> edges, tapa::mmap<bits<float_v>> out_feats,
                tapa::istreams<float_v, float_v::length>& out_feat_q) {
  float_v xformed_feats[node_count];
#pragma HLS bind_storage variable = xformed_feats type = RAM_S2P impl = URAM
  float_v out_feats_l[node_count];
#pragma HLS bind_storage variable = out_feats_l type = RAM_S2P impl = URAM

init_feats:
  for (Fid i = 0; i < node_count; ++i) {
#pragma HLS pipeline II = 1
    xformed_feats[i] = 0.f;
    out_feats_l[i] = 0.f;
  }

gemm:
  for (Fid k = 0; k < in_feat_len_vec; ++k) {
    for (Vid i = 0; i < node_count; ++i) {
#pragma HLS pipeline II = 1
      xformed_feats[i] += (((out_feat_q[0].read() + out_feat_q[1].read()) +
                            (out_feat_q[2].read() + out_feat_q[3].read())) +
                           ((out_feat_q[4].read() + out_feat_q[5].read()) +
                            (out_feat_q[6].read() + out_feat_q[7].read()))) +
                          (((out_feat_q[8].read() + out_feat_q[9].read()) +
                            (out_feat_q[10].read() + out_feat_q[11].read())) +
                           ((out_feat_q[12].read() + out_feat_q[13].read()) +
                            (out_feat_q[14].read() + out_feat_q[15].read())));
    }
  }

load_edges:
  for (Eid i = 0; i < norm_edge_count; ++i) {
#pragma HLS pipeline
    out_feats_l[edges[i].src] += xformed_feats[edges[i].dst] * edges[i].attr;
  }

store_nodes:
  for (Vid i = 0; i < node_count; ++i) {
#pragma HLS pipeline II = 1
    out_feats[i] = tapa::bit_cast<bits<float_v>>(out_feats_l[i]);
  }
}

void Gnn(tapa::mmap<bits<float_v>> in_feats, tapa::mmap<bits<float_v>> weights,
         tapa::mmap<Edge> edges, tapa::mmap<bits<float_v>> out_feats) {
  tapa::streams<float_v, float_v::length, 2> weight_q("weight_q");
  tapa::streams<float, float_v::length, 2> in_feat_q("in_feat_q");
  tapa::streams<float_v, float_v::length, 2> out_feat_q("out_feat_q");
  tapa::task()
      .invoke(WeightLoader, weights, weight_q)
      .invoke(NodeLoader, in_feats, in_feat_q)
      .invoke<tapa::join, float_v::length>(MacUnit, weight_q, in_feat_q,
                                           out_feat_q)
      .invoke(Aggregator, edges, out_feats, out_feat_q);
}
