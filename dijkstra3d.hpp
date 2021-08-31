/*
 * An implementation of a Edgar Dijkstra's Shortest Path Algorithm.
 * An absolute classic.
 * 
 * E. W. Dijkstra.
 * "A Note on Two Problems in Connexion with Graphs"
 * Numerische Mathematik 1. pp. 269-271. (1959)
 *
 * Of course, I use a priority queue.
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: August 2018
 */

#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <queue>
#include <vector>
#include <iostream>

#include "./hedly.h"
#include "./libdivide.h"

#ifndef DIJKSTRA3D_HPP
#define DIJKSTRA3D_HPP

#define NHOOD_SIZE 26

namespace dijkstra {

#define sq(x) ((x) * (x))

inline float* fill(float *arr, const float value, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    arr[i] = value;
  }
  return arr;
}

void connectivity_check(int connectivity) {
  if (connectivity != 6 && connectivity != 18 && connectivity != 26) {
    throw std::runtime_error("Only 6, 18, and 26 connectivities are supported.");
  }
}

template <typename OUT = uint32_t>
inline std::vector<OUT> query_shortest_path(const OUT* parents, const OUT target) {
  std::vector<OUT> path;
  OUT loc = target;
  while (parents[loc]) {
    path.push_back(loc);
    loc = parents[loc] - 1; // offset by 1 to disambiguate the 0th index
  }
  path.push_back(loc);

  return path;
}

inline void compute_neighborhood_helper(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz,
  const int connectivity = 26) {

  const int sxy = sx * sy;

  // 6-hood
  neighborhood[0] = -1 * (x > 0); // -x
  neighborhood[1] = (x < (static_cast<int>(sx) - 1)); // +x
  neighborhood[2] = -static_cast<int>(sx) * (y > 0); // -y
  neighborhood[3] = static_cast<int>(sx) * (y < static_cast<int>(sy) - 1); // +y
  neighborhood[4] = -sxy * static_cast<int>(z > 0); // -z
  neighborhood[5] = sxy * (z < static_cast<int>(sz) - 1); // +z

  // 18-hood

  // xy diagonals
  neighborhood[6] = (connectivity > 6) * (neighborhood[0] + neighborhood[2]) * (neighborhood[0] && neighborhood[2]); // up-left
  neighborhood[7] = (connectivity > 6) * (neighborhood[0] + neighborhood[3]) * (neighborhood[0] && neighborhood[3]); // up-right
  neighborhood[8] = (connectivity > 6) * (neighborhood[1] + neighborhood[2]) * (neighborhood[1] && neighborhood[2]); // down-left
  neighborhood[9] = (connectivity > 6) * (neighborhood[1] + neighborhood[3]) * (neighborhood[1] && neighborhood[3]); // down-right

  // yz diagonals
  neighborhood[10] = (connectivity > 6) * (neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]); // up-left
  neighborhood[11] = (connectivity > 6) * (neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]); // up-right
  neighborhood[12] = (connectivity > 6) * (neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]); // down-left
  neighborhood[13] = (connectivity > 6) * (neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]); // down-right

  // xz diagonals
  neighborhood[14] = (connectivity > 6) * (neighborhood[0] + neighborhood[4]) * (neighborhood[0] && neighborhood[4]); // up-left
  neighborhood[15] = (connectivity > 6) * (neighborhood[0] + neighborhood[5]) * (neighborhood[0] && neighborhood[5]); // up-right
  neighborhood[16] = (connectivity > 6) * (neighborhood[1] + neighborhood[4]) * (neighborhood[1] && neighborhood[4]); // down-left
  neighborhood[17] = (connectivity > 6) * (neighborhood[1] + neighborhood[5]) * (neighborhood[1] && neighborhood[5]); // down-right

  // 26-hood

  // Now the eight corners of the cube
  neighborhood[18] = (connectivity > 18) * (neighborhood[0] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[19] = (connectivity > 18) * (neighborhood[1] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[20] = (connectivity > 18) * (neighborhood[0] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[21] = (connectivity > 18) * (neighborhood[0] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[22] = (connectivity > 18) * (neighborhood[1] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[23] = (connectivity > 18) * (neighborhood[1] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[24] = (connectivity > 18) * (neighborhood[0] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
  neighborhood[25] = (connectivity > 18) * (neighborhood[1] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
}

inline void compute_neighborhood(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz,
  const int connectivity = 26, const uint32_t* voxel_connectivity_graph = NULL) {

  compute_neighborhood_helper(neighborhood, x, y, z, sx, sy, sz, connectivity);

  if (voxel_connectivity_graph == NULL) {
    return;
  }

  uint64_t loc = x + sx * (y + sy * z);
  uint32_t graph = voxel_connectivity_graph[loc];

  // graph conventions are defined here:
  // https://github.com/seung-lab/connected-components-3d/blob/3.2.0/cc3d_graphs.hpp#L73-L92

  // 6-hood
  neighborhood[0] *= ((graph & 0b000010) > 0); // -x
  neighborhood[1] *= ((graph & 0b000001) > 0); // +x
  neighborhood[2] *= ((graph & 0b001000) > 0); // -y
  neighborhood[3] *= ((graph & 0b000100) > 0); // +y
  neighborhood[4] *= ((graph & 0b100000) > 0); // -z
  neighborhood[5] *= ((graph & 0b010000) > 0); // +z

  // 18-hood

  // xy diagonals
  neighborhood[6] *= ((graph & 0b1000000000) > 0); // up-left -x,-y
  neighborhood[7] *= ((graph & 0b0010000000) > 0); // up-right -x,+y
  neighborhood[8] *= ((graph & 0b0100000000) > 0); // down-left +x,-y
  neighborhood[9] *= ((graph & 0b0001000000) > 0); // down-right +x,+y

  // yz diagonals
  neighborhood[10] *= ((graph & 0b100000000000000000) > 0); // up-left -y,-z
  neighborhood[11] *= ((graph & 0b000010000000000000) > 0); // up-right -y,+z
  neighborhood[12] *= ((graph & 0b010000000000000000) > 0); // down-left +y,-z
  neighborhood[13] *= ((graph & 0b000001000000000000) > 0); // down-right +y,+z

  // xz diagonals
  neighborhood[14] *= ((graph & 0b001000000000000000) > 0); // up-left, -x,-z
  neighborhood[15] *= ((graph & 0b000000100000000000) > 0); // up-right, -x,+z
  neighborhood[16] *= ((graph & 0b000100000000000000) > 0); // down-left +x,-z
  neighborhood[17] *= ((graph & 0b000000010000000000) > 0); // down-right +x,+z

  // 26-hood

  // Now the eight corners of the cube
  neighborhood[18] *= ((graph & 0b10000000000000000000000000) > 0); // -x,-y,-z
  neighborhood[19] *= ((graph & 0b01000000000000000000000000) > 0); // +x,-y,-z
  neighborhood[20] *= ((graph & 0b00100000000000000000000000) > 0); // -x,+y,-z
  neighborhood[21] *= ((graph & 0b00001000000000000000000000) > 0); // -x,-y,+z
  neighborhood[22] *= ((graph & 0b00010000000000000000000000) > 0); // +x,+y,-z
  neighborhood[23] *= ((graph & 0b00000100000000000000000000) > 0); // +x,-y,+z
  neighborhood[24] *= ((graph & 0b00000010000000000000000000) > 0); // -x,+y,+z
  neighborhood[25] *= ((graph & 0b00000001000000000000000000) > 0); // +x,+y,+z
}


inline void compute_eucl_distance(
  float* eucl_distance, 
  const float dx, const float dy, const float dz,
  const int connectivity = 26) {

  // 6-hood
  eucl_distance[0] = dx; // -x
  eucl_distance[1] = dx; // +x
  eucl_distance[2] = dy; // -y
  eucl_distance[3] = dy; // +y
  eucl_distance[4] = dz; // -z
  eucl_distance[5] = dz; // +z

  // 18-hood

  // xy diagonals
  eucl_distance[6] = dx + dy; // up-left
  eucl_distance[7] = dx + dy; // up-right
  eucl_distance[8] = dx + dy; // down-left
  eucl_distance[9] = dx + dy; // down-right

  // yz diagonals
  eucl_distance[10] = dy + dz; // up-left
  eucl_distance[11] = dy + dz; // up-right
  eucl_distance[12] = dy + dz; // down-left
  eucl_distance[13] = dy + dz; // down-right

  // xz diagonals
  eucl_distance[14] = dx + dz; // up-left
  eucl_distance[15] = dx + dz; // up-right
  eucl_distance[16] = dx + dz; // down-left
  eucl_distance[17] = dx + dz; // down-right

  // 26-hood

  // Now the eight corners of the cube
  eucl_distance[18] = dx + dy + dz;
  eucl_distance[19] = dx + dy + dz;
  eucl_distance[20] = dx + dy + dz;
  eucl_distance[21] = dx + dy + dz;
  eucl_distance[22] = dx + dy + dz;
  eucl_distance[23] = dx + dy + dz;
  eucl_distance[24] = dx + dy + dz;
  eucl_distance[25] = dx + dy + dz;
}


template <typename T = uint32_t>
class HeapNode {
public:
  float key; 
  T value;

  HeapNode() {
    key = 0;
    value = 0;
  }

  HeapNode (float k, T val) {
    key = k;
    value = val;
  }

  HeapNode (const HeapNode<T> &h) {
    key = h.key;
    value = h.value;
  }
};

template <typename T = uint32_t>
struct HeapNodeCompare {
  bool operator()(const HeapNode<T> &t1, const HeapNode<T> &t2) const {
    return t1.key >= t2.key;
  }
};

#define DIJKSTRA_3D_PREFETCH_26WAY(field, loc) \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sxy - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sxy - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sxy + sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sxy - sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sxy + sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sxy - sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sx - 1]), 0, 1);

/* Perform dijkstra's shortest path algorithm
 * on a 3D image grid. Vertices are voxels and
 * edges are the 26 nearest neighbors (except
 * for the edges of the image where the number
 * of edges is reduced).
 *
 * For given input voxels A and B, the edge
 * weight from A to B is B and from B to A is
 * A. All weights must be non-negative (incl. 
 * negative zero).
 *
 * I take advantage of negative weights to mean
 * "visited".
 *
 * Parameters:
 *  T* field: Input weights. T can be be a floating or 
 *     signed integer type, but not an unsigned int.
 *  sx, sy, sz: size of the volume along x,y,z axes in voxels.
 *  source: 1D index of starting voxel
 *  target: 1D index of target voxel
 *
 * Returns: vector containing 1D indices of the path from
 *   source to target including source and target.
 */
template <typename T, typename OUT = uint32_t>
std::pair<std::vector<OUT>, float> dijkstra3d(
    T* field, 
    float* prob,
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target,
    const float dx, const float dy, const float dz,
    const float w_grad, const float w_eucl, const float w_prob,
    const int connectivity = 26, 
    const uint32_t* voxel_connectivity_graph = NULL
  ) {

  connectivity_check(connectivity);

  float distance_target;
  std::vector<OUT> path;

  if (source == target) {
    distance_target = 0.0;
    
    return std::make_pair(std::vector<OUT>{ static_cast<OUT>(source) }, distance_target);
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist = new float[voxels]();
  OUT *parents = new OUT[voxels]();
  fill(dist, +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE];
  float eucl_distance[NHOOD_SIZE];
  compute_eucl_distance(eucl_distance, dx, dy, dz);

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  int x, y, z;
  bool target_reached = false;

  

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();
    
    if (std::signbit(dist[loc])) {
      continue;
    }

    // As early as possible, start fetching the
    // data from RAM b/c the annotated lines below
    // have 30-50% cache miss.
    DIJKSTRA_3D_PREFETCH_26WAY(field, loc)
    DIJKSTRA_3D_PREFETCH_26WAY(prob, loc)
    DIJKSTRA_3D_PREFETCH_26WAY(dist, loc)
    

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);
    

    

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = w_grad * abs(static_cast<float>(field[neighboridx]) - static_cast<float>(field[loc])); // high cache miss
      delta += w_eucl * eucl_distance[i];
      delta += w_prob * static_cast<float>(prob[neighboridx]) ;


      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { // high cache miss
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

        // Dijkstra, Edgar. "Go To Statement Considered Harmful".
        // Communications of the ACM. Vol. 11. No. 3 March 1968. pp. 147-148
        if (neighboridx == target) {
          target_reached = true;
          distance_target = dist[target];
          goto OUTSIDE;
        }

        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  OUTSIDE:
  delete []dist;

  
  
  // if voxel graph supplied, it's possible 
  // to never reach target.
  if (target_reached) { 
    path = query_shortest_path<OUT>(parents, target);
  }
  delete [] parents;

  return std::make_pair(path, distance_target);
}

template <typename T>
float* distance_field3d(
    T* field, 
    float* prob,
    const size_t sx, const size_t sy, const size_t sz, 
    const std::vector<size_t> &sources,
    const float dx, const float dy, const float dz,
    const float w_grad, const float w_eucl, const float w_prob,
    const int connectivity=26,
    const uint32_t* voxel_connectivity_graph = NULL
  ) {

  connectivity_check(connectivity);

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist = new float[voxels]();
  fill(dist, +INFINITY, voxels);


  int neighborhood[NHOOD_SIZE];
  float eucl_distance[NHOOD_SIZE];
  compute_eucl_distance(eucl_distance, dx, dy, dz);

  std::priority_queue<HeapNode<size_t>, std::vector<HeapNode<size_t>>, HeapNodeCompare<size_t>> queue;
  
  for (size_t source : sources) {
    dist[source] = -0;
    queue.emplace(0.0, source);
  }

  size_t loc, next_loc;
  float delta;
  size_t neighboridx;

  size_t x, y, z;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();

    if (std::signbit(dist[loc])) {
      continue;
    }

    if (!queue.empty()) {
      next_loc = queue.top().value;
      if (!std::signbit(dist[next_loc])) {

        // As early as possible, start fetching the
        // data from RAM b/c the annotated lines below
        // have 30-50% cache miss.
        DIJKSTRA_3D_PREFETCH_26WAY(field, next_loc)
        DIJKSTRA_3D_PREFETCH_26WAY(prob, next_loc)
        DIJKSTRA_3D_PREFETCH_26WAY(dist, next_loc)
      }
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, 26, voxel_connectivity_graph);

    for (size_t i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = w_grad * abs(static_cast<float>(field[neighboridx]) - static_cast<float>(field[loc])); // high cache miss
      delta += w_eucl * eucl_distance[i];
      delta += w_prob * static_cast<float>(prob[neighboridx]) ;

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { 
        dist[neighboridx] = dist[loc] + delta;
        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  for (size_t i = 0; i < voxels; i++) {
    dist[i] = std::fabs(dist[i]);
  }

  return dist;
}


#undef DIJKSTRA_3D_PREFETCH_26WAY

}; // namespace dijkstra3d

#endif