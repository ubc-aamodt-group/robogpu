#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <cassert>
#include <chrono>
#include <queue>
#include <random>

#include "robot.h"
#include "sram.h"
#include "sram_cubby_task_oriented_octree.h"
#include "sram_dresser_task_oriented_octree.h"
#include "sram_merged_cubby_task_oriented_octree.h"
#include "sram_tabletop_task_oriented_octree.h"

static std::random_device rd;
static std::mt19937 gen(rd());

/**
 * Returns a random number between min and max
 **/
static inline double r(double min, double max) {
  std::uniform_real_distribution<> dis(min, max);
  return dis(gen);
}
template <typename T>
using overlap_func_t = int (*)(geometry::OBB<T> obb, geometry::AABB<T> aabb);

template <typename T>
int overlap_orig(geometry::OBB<T> obb, geometry::AABB<T> aabb) {
  return obb.overlap(aabb);
}

template <typename T>
int overlap_mod(geometry::OBB<T> obb, geometry::AABB<T> aabb) {
  auto max = geometry::Sphere<T>::create_max(obb);
  if (!max.overlap(aabb)) {
    return 0;
  }
  auto min = geometry::Sphere<T>::create_min(obb);
  if (min.overlap(aabb)) {
    return 1;
  }
  return obb.overlap(aabb);
}

template <typename T>
int traverse(int env, geometry::OBB<T> obb, int &count,
             overlap_func_t<T> func) {
  std::queue<int> q;
  q.push(0);
  count = 0;
  while (!q.empty()) {
    int addr = q.front();
    q.pop();
    auto &entry = srams[env][addr];
    // std::cout << entry.child_address << std::endl;
    int offset = 0;
    for (int i = 0; i < 8; i++) {
      switch (entry.child_status[i]) {
        case 0: {
          break;
        }
        case 1: {
          count++;
          if (func(obb, entry.children_box[i])) {
            q.push(entry.child_address + offset);
          }
          offset++;
          break;
        }
        case 2: {
          count++;
          if (func(obb, entry.children_box[i])) {
            return 1;
          }
          break;
        }
        default: {
          assert(0);
          break;
        }
      }
    }
  }
  return 0;
}

using namespace std::chrono;

void run_cpu(int option, unsigned int num_obbs) {
  int *counts = new int[num_obbs];
  int *overlaps = new int[num_obbs];

  std::vector<struct geometry::OBB<double>> links;
  for (unsigned i = 0; i < num_obbs; i++) {
    struct geometry::OBB<double> obb = obbs[i % 4];
    for (int j = 0; j < 3; j++) {
    //   obb.c[j] = r(-16.0, 16.0);
    //   obb.e[j] = r(0.0, 16.0);
      obb.c[j] += r(-4.0, 4.0);
      obb.e[j] += r(0.0, 1.0);
    }
    // eulerToMatrix(r(0.0, TWO_PI), r(0.0, TWO_PI), r(0.0, TWO_PI), obb.u);
    links.push_back(obb);
  }

  unsigned N = 10;
  unsigned long long time = 0;
  for (unsigned h = 0; h < N; h++) {
    auto start = high_resolution_clock::now();
#pragma omp parallel for
    for (unsigned i = 0; i < num_obbs; i++) {
      int count = 0;
      int overlap =
          traverse(2, links[i], count,
                   option ? overlap_mod<double> : overlap_orig<double>);
      counts[i] = count;
      overlaps[i] = overlap;
    }
    auto duration =
        duration_cast<microseconds>(high_resolution_clock::now() - start);
    time += duration.count();
  }
  std::cout << "Num obbs: " << num_obbs << " Time: " << (time / N) << "us"
            << std::endl;

  int positive = 0;
  int negative = 0;
  int final_count = 0;
  for (int i = 0; i < 4; i++) {
    final_count += counts[i];
    if (overlaps[i]) {
      negative++;
    } else {
      positive++;
    }
  }

  delete[] counts;
  delete[] overlaps;
}

void run_cpu2(unsigned int num_obbs, int sram_idx) {
  // collect voxels from sram
  std::vector<struct geometry::AABB<double>> boxes;

  // select different octrees
  auto nodes = srams[2];
  int octree_size = 220;
  if (sram_idx == 0) { // original srams
    nodes = srams[2];
    octree_size = 220;
  } else if (sram_idx == 1) {
    std::cout << "Using cubby task oriented octree" << std::endl;
    nodes = mpinet_cubby_octree[0];
    octree_size = 33823;
  } else if (sram_idx == 2) {
    std::cout << "Using dresser task oriented octree" << std::endl;
    nodes = mpinet_dresser_octree[0];
    octree_size = 37345;
  } else if (sram_idx == 3) {
    std::cout << "Using merged cubby task oriented octree" << std::endl;
    nodes = mpinet_merged_cubby_octree[0];
    octree_size = 32216;
  } else if (sram_idx == 4) {
    std::cout << "Using tabletop task oriented octree" << std::endl;
    nodes = mpinet_tabletop_octree[0];
    octree_size = 33118;
  }

  for (int v = 0; v < octree_size; v++) {
    // auto &node = srams[2][v];
    auto &node = nodes[v];
    for (int i = 0; i < 8; i++) {
      if (node.child_status[i] == 2) {
        auto &box = node.children_box[i];
        boxes.push_back(box);
      }
    }
  }

  std::vector<struct geometry::OBB<double>> links;
  for (unsigned i = 0; i < num_obbs; i++) {
    struct geometry::OBB<double> obb = obbs[i % 4];
    for (int j = 0; j < 3; j++) {
    //   obb.c[j] = r(-16.0, 16.0);
    //   obb.e[j] = r(0.0, 16.0);
      obb.c[j] += r(-4.0, 4.0);
      obb.e[j] += r(0.0, 1.0);
    }
    // eulerToMatrix(r(0.0, TWO_PI), r(0.0, TWO_PI), r(0.0, TWO_PI), obb.u);
    links.push_back(obb);
  }

  int *overlaps = new int[num_obbs * boxes.size()];

  unsigned N = 10;
  unsigned long long time = 0;
  for (unsigned h = 0; h < N; h++) {
    auto start = high_resolution_clock::now();
#pragma omp parallel for
    for (unsigned i = 0; i < num_obbs; i++) {
      for (unsigned j = 0; j < boxes.size(); j++) {
        overlaps[i * boxes.size() + j] = overlap_mod(links[i], boxes[j]);
      }
    }
    auto duration =
        duration_cast<microseconds>(high_resolution_clock::now() - start);
    time += duration.count();
  }
  std::cout << "Num obbs: " << num_obbs << " Time: " << (time / N) << "us"
            << std::endl;

  unsigned long long positive = 0, negative = 0;
  for (int i = 0; i < num_obbs; i++) {
    if (overlaps[i]) {
      negative++;
    } else {
      positive++;
    }
  }
  std::cout << "\tPositive: " << positive << " Negative: " << negative
            << std::endl;

  delete[] overlaps;
}