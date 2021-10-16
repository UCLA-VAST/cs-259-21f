#include <cmath>

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <tapa.h>

#include "bfs.h"
#include "nxgraph.hpp"

using std::clog;
using std::endl;
using std::runtime_error;
using std::vector;

void BfsHostBaseline(Vid base_vid, vector<VertexAttr>& vertices,
                     const vector<Edge>& edges) {
  bool has_update = true;
  while (has_update) {
    has_update = false;
    for (const auto& edge : edges) {
      if (vertices[edge.src - base_vid] < vertices[edge.dst - base_vid]) {
        vertices[edge.dst - base_vid] = vertices[edge.src - base_vid];
        has_update = true;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  const size_t partition_size = argc > 2 ? atoi(argv[2]) : 1024;
  auto partitions =
      nxgraph::LoadEdgeList<Vid, Eid, VertexAttr>(argv[1], partition_size);
  for (const auto& partition : partitions) {
    VLOG(6) << "partition";
    VLOG(6) << "num vertices: " << partition.num_vertices;
    for (Eid i = 0; i < partition.num_edges; ++i) {
      auto edge = partition.shard.get()[i];
      VLOG(6) << "src: " << edge.src << " dst: " << edge.dst;
    }
  }
  const size_t num_partitions = partitions.size();

  vector<Vid> num_vertices(num_partitions + 1);
  vector<Eid> num_edges(num_partitions);
  Vid total_num_vertices = 0;
  Eid total_num_edges = 0;
  num_vertices[0] = partitions[0].base_vid;
  const auto& base_vid = num_vertices[0];
  for (size_t i = 0; i < num_partitions; ++i) {
    num_vertices[i + 1] = partitions[i].num_vertices;
    num_edges[i] = partitions[i].num_edges;
    total_num_vertices += partitions[i].num_vertices;
    total_num_edges += partitions[i].num_edges;
  }

  vector<VertexAttr> vertices(total_num_vertices);
  vector<VertexAttr> vertices_baseline(total_num_vertices);
  for (Vid i = 0; i < total_num_vertices; ++i) {
    vertices[i] = base_vid + i;
    vertices_baseline[i] = base_vid + i;
  }

  vector<Edge> edges(total_num_edges);
  if (sizeof(Edge) != sizeof(nxgraph::Edge<Vid>)) {
    throw runtime_error("inconsistent Edge type");
  }
  auto edge_ptr = edges.data();
  for (size_t i = 0; i < num_partitions; ++i) {
    memcpy(edge_ptr, partitions[i].shard.get(), num_edges[i] * sizeof(Edge));
    edge_ptr += num_edges[i];
  }
  vector<Update> updates(total_num_edges * num_partitions);
  VLOG(10) << "num_vertices";
  for (auto n : num_vertices) {
    VLOG(10) << n;
  }
  VLOG(10) << "num_edges";
  for (auto n : num_edges) {
    VLOG(10) << n;
  }
  VLOG(10) << "vertices: ";
  for (auto v : vertices) {
    VLOG(10) << v;
  }
  VLOG(10) << "edges: ";
  for (auto e : edges) {
    VLOG(10) << e.src << " -> " << e.dst;
  }
  VLOG(10) << "updates: " << updates.size();
  std::string bitstream;
  if (const auto bitstream_ptr = getenv("TAPAB")) {
    bitstream = bitstream_ptr;
  }
  tapa::invoke(
      Bfs, bitstream, num_partitions,
      tapa::read_only_mmap<const Vid>(num_vertices),
      tapa::read_only_mmap<const Eid>(num_edges),
      tapa::read_write_mmap<VertexAttr>(vertices),
      tapa::read_only_mmap<Edge>(edges).reinterpret<bits<Edge>>(),
      tapa::placeholder_mmap<Update>(updates).reinterpret<bits<Update>>());
  BfsHostBaseline(base_vid, vertices_baseline, edges);
  VLOG(10) << "vertices: ";
  for (auto v : vertices) {
    VLOG(10) << v;
  }

  uint64_t num_errors = 0;
  const uint64_t threshold = 10;  // only report up to these errors
  for (Vid i = 0; i < total_num_vertices; ++i) {
    auto expected = vertices_baseline[i];
    auto actual = vertices[i];
    if (actual != expected) {
      if (num_errors < threshold) {
        LOG(ERROR) << "vertex #" << base_vid + i << ": "
                   << "expected: " << expected << ", actual: " << actual
                   << endl;
      } else if (num_errors == threshold) {
        LOG(ERROR) << "...";
      }
      ++num_errors;
    }
  }
  if (num_errors == 0) {
    clog << "PASS!" << endl;
  } else {
    if (num_errors > threshold) {
      clog << " (+" << (num_errors - threshold) << " more errors)" << endl;
    }
    clog << "FAIL!" << endl;
  }

  return 0;
}
