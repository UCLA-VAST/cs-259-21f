#ifndef BFS_H_
#define BFS_H_

#include <cstdint>

#include <ap_int.h>
#include <tapa.h>

// There is a bug in Vitis HLS preventing fully pipelined read/write of struct
// via m_axi; using ap_uint can work-around this problem.
template <typename T>
using bits = ap_uint<tapa::widthof<T>()>;

using Vid = uint32_t;
using Eid = uint32_t;
using Pid = uint16_t;

using VertexAttr = Vid;

struct Edge {
  Vid src;
  Vid dst;
};

struct Update {
  Vid dst;
  Vid depth;
};

void Bfs(Pid num_partitions, tapa::mmap<const Vid> num_vertices,
         tapa::mmap<const Eid> num_edges, tapa::mmap<VertexAttr> vertices,
         tapa::mmap<bits<Edge>> edges, tapa::mmap<bits<Update>> updates);

#endif  // BFS_H_
