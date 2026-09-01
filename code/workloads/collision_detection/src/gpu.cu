#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <cassert>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <istream>
#include <string>
#include <sstream>

#include "cuda_helpers.h"
#include "robot.h"
#include "sram.h"
#include "octrees/mpinet_dresser_octree.h"
#include "octrees/mpinet_cubby_octree.h"
#include "octrees/mpinet_merged_cubby_octree.h"
#include "octrees/mpinet_tabletop_octree.h"
#include "octrees/mpnet_master_octree.h"
#include "octrees/physical_aiav_clip0_octree.h"
#include "octrees/physical_aiav_clip1_octree.h"
#include "octrees/physical_aiav_clip2_octree.h"
#include "octrees/physical_aiav_clip3_octree.h"
#include "octrees/physical_aiav_clip4_octree.h"
#include "octrees/physical_aiav_clip5_octree.h"
#include "octrees/physical_aiav_clip6_octree.h"
#include "octrees/physical_aiav_clip7_octree.h"
#include "octrees/physical_aiav_clip8_octree.h"
#include "octrees/physical_aiav_clip9_octree.h"
#include "octrees/physical_aiav_clip10_octree.h"
#include "octrees/physical_aiav_clip11_octree.h"
#include "octrees/physical_aiav_clip12_octree.h"
#include "octrees/physical_aiav_clip13_octree.h"

#include "csv_reader.h"

#define TWO_PI (2 * 3.1415926)
#define TOP_MAX_LEVEL 3

#define checkCudaErrors(call)                                 \
do {                                                        \
  cudaError_t err = call;                                   \
  if (err != cudaSuccess) {                                 \
    printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err));                        \
    exit(EXIT_FAILURE);                                     \
  }                                                         \
} while (0)


static std::random_device rd;
static std::mt19937 gen(rd());

__global__ void traverse_gpu(group_queue_t qs, unsigned int num_obbs,
                             struct entry<double> *d_srams,
                             struct geometry::OBB<double> *d_obbs,
                             int *overlaps) {
  unsigned int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx < num_obbs) {
    group_push(&qs, idx, 0);
    while (!group_empty(&qs, idx)) {
      int addr = group_pop(&qs, idx);
      auto &entry = d_srams[addr];
      int offset = 0;
      for (int i = 0; i < 8; i++) {
        switch (entry.child_status[i]) {
          case 0: {
            break;
          }
          case 1: {
            if (cudaOverlapMod(d_obbs[idx], entry.children_box[i])) {
              group_push(&qs, idx, entry.child_address + offset);
            }
            offset++;
            break;
          }
          case 2: {
            if (cudaOverlapMod(d_obbs[idx], entry.children_box[i])) {
              overlaps[idx] = 1;
              return;
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
    overlaps[idx] = 0;
  }
}

#ifdef RT_UNIT_ACCEL
__device__ __noinline__ void __TreeUnitSearch(
  TreeSearchConditions conditions,
	void* rootNode,
	TreeSearchResults* results,
	unsigned node_config,
	unsigned tri_config
) {
  printf("PLEASE RE-RUN IN GPGPUSIM\n");

  // Use all the variables so they're not optimized out
  printf("Search conditions: ");
  for (int i=0; i<12; i++) {
      printf("%d ", conditions.values[i]);
  }
  printf("\n");

  printf("Root node address: 0x%x\n", rootNode);

  printf("Search results: ");
  for (int i=0; i<8; i++) {
      printf("%d ", results->values[i]);
  }
  printf("\n");

  printf("Node intersection type: %d\n", node_config);
  printf("Leaf intersection type: %d\n", tri_config);

	return;
}
#endif

#ifdef RT_UNIT_ACCEL
__global__ void traverse_gpu_no_opt(queue_t *qs, unsigned int num_obbs,
                                    struct entry<double> *d_srams,
                                    struct geometry::OBB<double> *d_obbs,
                                    TreeSearchResults *overlaps) {
  unsigned int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx < num_obbs) {
    TreeSearchConditions conditions;
    // OBB values
    // type cast to float first
    float obb_values[15];
    obb_values[0] = (float)(d_obbs[idx].u[0][0]);
    obb_values[1] = (float)(d_obbs[idx].u[0][1]);
    obb_values[2] = (float)(d_obbs[idx].u[0][2]);
    obb_values[3] = (float)(d_obbs[idx].u[1][0]);
    obb_values[4] = (float)(d_obbs[idx].u[1][1]);
    obb_values[5] = (float)(d_obbs[idx].u[1][2]);
    obb_values[6] = (float)(d_obbs[idx].u[2][0]);
    obb_values[7] = (float)(d_obbs[idx].u[2][1]);
    obb_values[8] = (float)(d_obbs[idx].u[2][2]);
    obb_values[9] = (float)(d_obbs[idx].c[0]);
    obb_values[10] = (float)(d_obbs[idx].c[1]);
    obb_values[11] = (float)(d_obbs[idx].c[2]);
    obb_values[12] = (float)(d_obbs[idx].e[0]);
    obb_values[13] = (float)(d_obbs[idx].e[1]);
    obb_values[14] = (float)(d_obbs[idx].e[2]);
     
    // write to conditions
    conditions.values[0] = *(uint32_t*)&(obb_values[0]);
    conditions.values[1] = *(uint32_t*)&(obb_values[1]);
    conditions.values[2] = *(uint32_t*)&(obb_values[2]);
    conditions.values[3] = *(uint32_t*)&(obb_values[3]);
    conditions.values[4] = *(uint32_t*)&(obb_values[4]);
    conditions.values[5] = *(uint32_t*)&(obb_values[5]);
    conditions.values[6] = *(uint32_t*)&(obb_values[6]);
    conditions.values[7] = *(uint32_t*)&(obb_values[7]);
    conditions.values[8] = *(uint32_t*)&(obb_values[8]);
    conditions.values[9] = *(uint32_t*)&(obb_values[9]);
    conditions.values[10] = *(uint32_t*)&(obb_values[10]);
    conditions.values[11] = *(uint32_t*)&(obb_values[11]);
    conditions.values[12] = *(uint32_t*)&(obb_values[12]);
    conditions.values[13] = *(uint32_t*)&(obb_values[13]);
    conditions.values[14] = *(uint32_t*)&(obb_values[14]);

    // print the difference between the original double and the conditions.values
    // print double , float, and the difference one by one
    // if (idx == 0) {
    //   printf("u[0][0]: %f, %f, %f\n", d_obbs[idx].u[0][0], *(float*)&(conditions.values[0]), d_obbs[idx].u[0][0] - *(float*)&(conditions.values[0]));
    //   printf("u[0][1]: %f, %f, %f\n", d_obbs[idx].u[0][1], *(float*)&(conditions.values[1]), d_obbs[idx].u[0][1] - *(float*)&(conditions.values[1]));
    //   printf("u[0][2]: %f, %f, %f\n", d_obbs[idx].u[0][2], *(float*)&(conditions.values[2]), d_obbs[idx].u[0][2] - *(float*)&(conditions.values[2]));
    //   printf("u[1][0]: %f, %f, %f\n", d_obbs[idx].u[1][0], *(float*)&(conditions.values[3]), d_obbs[idx].u[1][0] - *(float*)&(conditions.values[3]));
    //   printf("u[1][1]: %f, %f, %f\n", d_obbs[idx].u[1][1], *(float*)&(conditions.values[4]), d_obbs[idx].u[1][1] - *(float*)&(conditions.values[4]));
    //   printf("u[1][2]: %f, %f, %f\n", d_obbs[idx].u[1][2], *(float*)&(conditions.values[5]), d_obbs[idx].u[1][2] - *(float*)&(conditions.values[5]));
    //   printf("u[2][0]: %f, %f, %f\n", d_obbs[idx].u[2][0], *(float*)&(conditions.values[6]), d_obbs[idx].u[2][0] - *(float*)&(conditions.values[6]));
    //   printf("u[2][1]: %f, %f, %f\n", d_obbs[idx].u[2][1], *(float*)&(conditions.values[7]), d_obbs[idx].u[2][1] - *(float*)&(conditions.values[7]));
    //   printf("u[2][2]: %f, %f, %f\n", d_obbs[idx].u[2][2], *(float*)&(conditions.values[8]), d_obbs[idx].u[2][2] - *(float*)&(conditions.values[8]));
    //   printf("c[0]: %f, %f, %f\n", d_obbs[idx].c[0], *(float*)&(conditions.values[9]), d_obbs[idx].c[0] - *(float*)&(conditions.values[9]));
    //   printf("c[1]: %f, %f, %f\n", d_obbs[idx].c[1], *(float*)&(conditions.values[10]), d_obbs[idx].c[1] - *(float*)&(conditions.values[10]));
    //   printf("c[2]: %f, %f, %f\n", d_obbs[idx].c[2], *(float*)&(conditions.values[11]), d_obbs[idx].c[2] - *(float*)&(conditions.values[11]));
    //   printf("e[0]: %f, %f, %f\n", d_obbs[idx].e[0], *(float*)&(conditions.values[12]), d_obbs[idx].e[0] - *(float*)&(conditions.values[12]));
    //   printf("e[1]: %f, %f, %f\n", d_obbs[idx].e[1], *(float*)&(conditions.values[13]), d_obbs[idx].e[1] - *(float*)&(conditions.values[13]));
    //   printf("e[2]: %f, %f, %f\n", d_obbs[idx].e[2], *(float*)&(conditions.values[14]), d_obbs[idx].e[2] - *(float*)&(conditions.values[14]));
    // }

    // TreeSearchResults results;
    volatile unsigned node_config = 8;
    volatile unsigned leaf_config = 8;
    __TreeUnitSearch(
          conditions, 
          (void*) d_srams, // root node address
          &overlaps[idx],
          // (TreeSearchResults*)&output[tid], // Temporarily use this (probably unsafe but works for now)
          node_config, 
          leaf_config);
  }
}
#else
__global__ void traverse_gpu_no_opt(queue_t *qs, unsigned int num_obbs,
                                    struct entry<double> *d_srams,
                                    struct geometry::OBB<double> *d_obbs,
                                    int *overlaps) {
  unsigned int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx < num_obbs) {
    queue_t *q = &qs[idx];
    push(q, 0);
    while (!empty(q)) {
      int addr = pop(q);
      auto &entry = d_srams[addr];
      int offset = 0;
      for (int i = 0; i < 8; i++) {
        switch (entry.child_status[i]) {
          case 0: {
            break;
          }
          case 1: {
            if (cudaOverlapMod(d_obbs[idx], entry.children_box[i])) {
              push(q, entry.child_address + offset);
            }
            offset++;
            break;
          }
          case 2: {
            if (cudaOverlapMod(d_obbs[idx], entry.children_box[i])) {
              overlaps[idx] = 1;
              return;
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
    overlaps[idx] = 0;
  }
}
#endif

__global__ void profile(group_queue_t qs, unsigned int num_obbs,
                        struct entry<double> *d_srams,
                        struct geometry::OBB<double> *d_obbs, int *matrix,
                        int start, int top_leaves) {
  unsigned int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx < num_obbs) {
    group_push(&qs, idx, 0);
    while (!group_empty(&qs, idx)) {
      int addr = group_pop(&qs, idx);
      auto &entry = d_srams[addr];
      if (entry.level == TOP_MAX_LEVEL) {
        matrix[idx * top_leaves + (addr - start)] = 1;
        continue;
      }
      int offset = 0;
      for (int i = 0; i < 8; i++) {
        switch (entry.child_status[i]) {
          case 0: {
            break;
          }
          case 1: {
            if (cudaOverlapMod(d_obbs[idx], entry.children_box[i])) {
              group_push(&qs, idx, entry.child_address + offset);
            }
            offset++;
            break;
          }
          case 2: {
            if (cudaOverlapMod(d_obbs[idx], entry.children_box[i])) {
              return;
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
  }
}

using namespace std::chrono;
void run_gpu(unsigned int num_obbs, int sram_idx, bool opt = true) {
  int CHOSEN = 2;

  // select different octrees
  auto nodes = srams[CHOSEN];
  size_t aabb_size = 40000;
  if (sram_idx == 0) { // original srams
    nodes = srams[CHOSEN];
    aabb_size = sizeof(struct entry<double>) * 220;
  } else if (sram_idx == 1) {
    std::cout << "Using cubby task oriented octree" << std::endl;
    nodes = mpinet_cubby_octree[0];
    aabb_size = sizeof(struct entry<double>) * 1557;
  } else if (sram_idx == 2) {
    std::cout << "Using dresser task oriented octree" << std::endl;
    nodes = mpinet_dresser_octree[0];
    aabb_size = sizeof(struct entry<double>) * 1557;
  } else if (sram_idx == 3) {
    std::cout << "Using merged cubby task oriented octree" << std::endl;
    nodes = mpinet_merged_cubby_octree[0];
    aabb_size = sizeof(struct entry<double>) * 1442;
  } else if (sram_idx == 4) {
    std::cout << "Using tabletop task oriented octree" << std::endl;
    nodes = mpinet_tabletop_octree[0];
    aabb_size = sizeof(struct entry<double>) * 1261;
  } else if (sram_idx == 600) {
    std::cout << "Using physical_aiav_clip0 task oriented octree" << std::endl;
    nodes = physical_aiav_clip0_octree[0];
    aabb_size = sizeof(physical_aiav_clip0_octree[0]);
  } else if (sram_idx == 601) {
    std::cout << "Using physical_aiav_clip1 task oriented octree" << std::endl;
    nodes = physical_aiav_clip1_octree[0];
    aabb_size = sizeof(physical_aiav_clip1_octree[0]);
  } else if (sram_idx == 602) {
    std::cout << "Using physical_aiav_clip2 task oriented octree" << std::endl;
    nodes = physical_aiav_clip2_octree[0];
    aabb_size = sizeof(physical_aiav_clip2_octree[0]);
  } else if (sram_idx == 603) {
    std::cout << "Using physical_aiav_clip3 task oriented octree" << std::endl;
    nodes = physical_aiav_clip3_octree[0];
    aabb_size = sizeof(physical_aiav_clip3_octree[0]);
  } else if (sram_idx == 604) {
    std::cout << "Using physical_aiav_clip4 task oriented octree" << std::endl;
    nodes = physical_aiav_clip4_octree[0];
    aabb_size = sizeof(physical_aiav_clip4_octree[0]);
  } else if (sram_idx == 605) {
    std::cout << "Using physical_aiav_clip5 task oriented octree" << std::endl;
    nodes = physical_aiav_clip5_octree[0];
    aabb_size = sizeof(physical_aiav_clip5_octree[0]);
  } else if (sram_idx == 606) {
    std::cout << "Using physical_aiav_clip6 task oriented octree" << std::endl;
    nodes = physical_aiav_clip6_octree[0];
    aabb_size = sizeof(physical_aiav_clip6_octree[0]);
  } else if (sram_idx == 607) {
    std::cout << "Using physical_aiav_clip7 task oriented octree" << std::endl;
    nodes = physical_aiav_clip7_octree[0];
    aabb_size = sizeof(physical_aiav_clip7_octree[0]);
  } else if (sram_idx == 608) {
    std::cout << "Using physical_aiav_clip8 task oriented octree" << std::endl;
    nodes = physical_aiav_clip8_octree[0];
    aabb_size = sizeof(physical_aiav_clip8_octree[0]);
  } else if (sram_idx == 609) {
    std::cout << "Using physical_aiav_clip9 task oriented octree" << std::endl;
    nodes = physical_aiav_clip9_octree[0];
    aabb_size = sizeof(physical_aiav_clip9_octree[0]);
  } else if (sram_idx == 610) {
    std::cout << "Using physical_aiav_clip10 task oriented octree" << std::endl;
    nodes = physical_aiav_clip10_octree[0];
    aabb_size = sizeof(physical_aiav_clip10_octree[0]);
  } else if (sram_idx == 611) {
    std::cout << "Using physical_aiav_clip11 task oriented octree" << std::endl;
    nodes = physical_aiav_clip11_octree[0];
    aabb_size = sizeof(physical_aiav_clip11_octree[0]);
  } else if (sram_idx == 612) {
    std::cout << "Using physical_aiav_clip12 task oriented octree" << std::endl;
    nodes = physical_aiav_clip12_octree[0];
    aabb_size = sizeof(physical_aiav_clip12_octree[0]);
  } else if (sram_idx == 613) {
    std::cout << "Using physical_aiav_clip13 task oriented octree" << std::endl;
    nodes = physical_aiav_clip13_octree[0];
    aabb_size = sizeof(physical_aiav_clip13_octree[0]);
  } else {
    std::cout << "Using octree index " << sram_idx - 5 << " from file" << std::endl;
    nodes = mpnet_master_octree[sram_idx - 5];
    aabb_size = sizeof(struct entry<double>) * 500;
  }

  // copy one sram to gpu
  struct entry<double> *d_srams;
  // size_t size = sizeof(struct entry<double>) * 220;
  cudaMalloc(&d_srams, aabb_size);
  cudaMemcpy(d_srams, &nodes[0], aabb_size, cudaMemcpyHostToDevice);

  // Read OBB csv and copy to GPU (only for scenes 1-4)
  // copy robot links to gpu
  std::vector<struct geometry::OBB<double>> links;

  if (sram_idx == 0) {
    for (int i = 0; i < num_obbs; i++) {
      struct geometry::OBB<double> obb = obbs[i % 4];
      // for (int j = 0; j < 3; j++) { // disable jittering for now
      //   //   obb.c[j] = r(-16.0, 16.0);
      //   //   obb.e[j] = r(0.0, 16.0);
      //   obb.c[j] += r(-4.0, 4.0);
      //   obb.e[j] += r(0.0, 1.0);
      // }
      // eulerToMatrix(r(0.0, TWO_PI), r(0.0, TWO_PI), r(0.0, TWO_PI), obb.u);
      links.push_back(obb);
    }
  } else if (sram_idx == 1) {
    links = load_obbs("obb_files/cubby_obbs.csv");
  } else if (sram_idx == 2) {
    links = load_obbs("obb_files/dresser_obbs.csv");
  } else if (sram_idx == 3) {
    links = load_obbs("obb_files/merged_cubby_obbs.csv");
  } else if (sram_idx == 4) {
    links = load_obbs("obb_files/tabletop_obbs.csv");
  } else if (sram_idx == 600) {
    links = load_obbs("obb_files/physicalaiav_clip0_obb.csv");
  } else if (sram_idx == 601) {
    links = load_obbs("obb_files/physicalaiav_clip1_obb.csv");
  } else if (sram_idx == 602) {
    links = load_obbs("obb_files/physicalaiav_clip2_obb.csv");
  } else if (sram_idx == 603) {
    links = load_obbs("obb_files/physicalaiav_clip3_obb.csv");
  } else if (sram_idx == 604) {
    links = load_obbs("obb_files/physicalaiav_clip4_obb.csv");
  } else if (sram_idx == 605) {
    links = load_obbs("obb_files/physicalaiav_clip5_obb.csv");
  } else if (sram_idx == 606) {
    links = load_obbs("obb_files/physicalaiav_clip6_obb.csv");
  } else if (sram_idx == 607) {
    links = load_obbs("obb_files/physicalaiav_clip7_obb.csv");
  } else if (sram_idx == 608) {
    links = load_obbs("obb_files/physicalaiav_clip8_obb.csv");
  } else if (sram_idx == 609) {
    links = load_obbs("obb_files/physicalaiav_clip9_obb.csv");
  } else if (sram_idx == 610) {
    links = load_obbs("obb_files/physicalaiav_clip10_obb.csv");
  } else if (sram_idx == 611) {
    links = load_obbs("obb_files/physicalaiav_clip11_obb.csv");
  } else if (sram_idx == 612) {
    links = load_obbs("obb_files/physicalaiav_clip12_obb.csv");
  } else if (sram_idx == 613) {
    links = load_obbs("obb_files/physicalaiav_clip13_obb.csv");
  } else {
    links = load_obbs("obb_files/mpnet_master_obb.csv");
  }

  printf("Loaded %lu obbs...\n", links.size());

  size_t size = sizeof(struct geometry::OBB<double>) * std::min((size_t)num_obbs, std::min((size_t)num_obbs, links.size()));
  struct geometry::OBB<double> *d_obbs;
  cudaMalloc(&d_obbs, size);
  cudaMemcpy(d_obbs, links.data(), size, cudaMemcpyHostToDevice);

  #ifdef RT_UNIT_ACCEL
    TreeSearchResults *d_overlaps;
    size = sizeof(TreeSearchResults) * std::min((size_t)num_obbs, links.size());
    cudaMalloc(&d_overlaps, size);
    cudaMemset(d_overlaps, 0, size);
  #else
    int *d_overlaps;
    size = sizeof(int) * std::min((size_t)num_obbs, links.size())*2;
    cudaMalloc(&d_overlaps, size);
    cudaMemset(d_overlaps, 0, size);
  #endif

  if (opt) {
    // prepare some FIFO queue and move to the GPU
    group_queue_t qs;
    group_init(&qs, std::min((size_t)num_obbs, links.size()), std::min((size_t)num_obbs, links.size())*2, true); // was 1024

    int state = 0;
    int s = -1;
    int top_leaves = 0;
    for (int i = 0; i < 220; i++) {
      // int l = srams[CHOSEN][i].level;
      int l = nodes[i].level;
      if (state == 0) {
        if (l == (sram_idx > 0 ? TOP_MAX_LEVEL-1 : TOP_MAX_LEVEL)) {
          s = i;
          state = 1;
        }
      } else if (state == 1) {
        if (l != (sram_idx > 0 ? TOP_MAX_LEVEL-1 : TOP_MAX_LEVEL)) {
          top_leaves = i - s;
          state = 2;
        }
      } else {
        assert(l != (sram_idx > 0 ? TOP_MAX_LEVEL-1 : TOP_MAX_LEVEL));
      }
    }

    int *d_matrix;
    size = sizeof(int) * std::min((size_t)num_obbs, links.size()) * top_leaves;
    cudaMalloc(&d_matrix, size);
    cudaMemset(d_matrix, 0, size);

    profile<<<std::ceil(std::min((size_t)num_obbs, links.size()) / 256.0), 256>>>(qs, std::min((size_t)num_obbs, links.size()), d_srams, d_obbs,
                                                  d_matrix, s, top_leaves);
    cudaDeviceSynchronize();

    size = sizeof(int) * std::min((size_t)num_obbs, links.size()) * top_leaves;
    int *matrix = (int *)malloc(size);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    std::vector<int> sandpile;
    for (int o = 0; o < std::min((size_t)num_obbs, links.size()); o++) {
      sandpile.push_back(o);
    }
    for (int n = 0; n < top_leaves; n++) {
      std::vector<int> taken, untaken;
      for (int o = 0; o < std::min((size_t)num_obbs, links.size()); o++) {
        int oid = sandpile[o];
        if (matrix[oid * top_leaves + n]) {
          taken.push_back(oid);
        } else {
          untaken.push_back(oid);
        }
      }
      for (auto u : untaken) {
        taken.push_back(u);
      }
      sandpile = taken;
    }
    std::vector<struct geometry::OBB<double>> links2;
    for (int i = 0; i < sandpile.size(); i++) {
      links2.push_back(links[sandpile[i]]);
    }

    cudaFree(d_obbs);
    size = sizeof(struct geometry::OBB<double>) * std::min((size_t)num_obbs, links.size());
    cudaMalloc(&d_obbs, size);
    cudaMemcpy(d_obbs, links2.data(), size, cudaMemcpyHostToDevice);

    cudaFree(d_matrix);
    free(matrix);

    // std::cerr << "start running with " << std::min((size_t)num_obbs, links.size()) << std::endl;
    auto start = high_resolution_clock::now();
    int N = 1; // just need to run one time, no need to avg
    unsigned long long count = 0;
    for (int i = 0; i < N; i++) {
      #ifdef RT_UNIT_ACCEL
        // TODO
      #else
        traverse_gpu<<<std::ceil(std::min((size_t)num_obbs, links.size()) / 256.0), 256>>>(qs, std::min((size_t)num_obbs, links.size()), d_srams,
                                                          d_obbs, d_overlaps);
      #endif
      cudaDeviceSynchronize();
      auto duration =
          duration_cast<microseconds>(high_resolution_clock::now() - start);
      count += duration.count();
    }
    std::cout << "Num obbs: " << std::min((size_t)num_obbs, links.size())
              << " Kernel execution: " << (count / N) << "us" << std::endl;

    group_destroy(&qs, true);
  } else {
    size = sizeof(queue_t) * std::max((size_t)num_obbs, links.size()) * 2;
    queue_t *qs = (queue_t *)malloc(size);
    for (int i = 0; i < std::max((size_t)num_obbs, links.size()); i++) {
      init(&qs[i], std::max((size_t)num_obbs, links.size())*2, true); // was 1024
    }
    queue_t *d_qs;
    cudaMalloc(&d_qs, size);
    cudaMemcpy(d_qs, qs, size, cudaMemcpyHostToDevice);

    // std::cerr << "start running with " << std::min((size_t)num_obbs, links.size()) << std::endl;
    auto start = high_resolution_clock::now();
    int N = 1; // only need to run 1 time
    unsigned long long count = 0;
    for (int i = 0; i < N; i++) {
      traverse_gpu_no_opt<<<std::ceil(std::min((size_t)num_obbs, links.size()) / 256.0), 256>>>( 
          d_qs, std::min((size_t)num_obbs, links.size()), d_srams, d_obbs, d_overlaps);
      cudaDeviceSynchronize();
      auto duration =
          duration_cast<microseconds>(high_resolution_clock::now() - start);
      count += duration.count();
    }
    std::cout << "Num obbs: " << std::min((size_t)num_obbs, links.size())
              << " Kernel execution: " << (count / N) << "us" << std::endl;

    cudaFree(d_qs);
    for (int i = 0; i < std::min((size_t)num_obbs, links.size()); i++) {
      destroy(&qs[i], true);
    }
    free(qs);
  }

  #ifdef RT_UNIT_ACCEL
    TreeSearchResults *overlaps = (TreeSearchResults*)calloc(std::min((size_t)num_obbs, links.size()), sizeof(TreeSearchResults));
    cudaMemcpy(overlaps, d_overlaps, sizeof(TreeSearchResults) * std::min((size_t)num_obbs, links.size()),
               cudaMemcpyDeviceToHost);
  #else
    int *overlaps = (int *)malloc(sizeof(int) * std::min((size_t)num_obbs, links.size()));
    cudaMemcpy(overlaps, d_overlaps, sizeof(int) * std::min((size_t)num_obbs, links.size()),
               cudaMemcpyDeviceToHost);
  #endif
  
  #ifdef RT_UNIT_ACCEL
    unsigned long long positive = 0, negative = 0;
    for (int i = 0; i < std::min((size_t)num_obbs, links.size()); i++) {
      if (overlaps[i].values[0]) {
        negative++;
      } else {
        positive++;
      }
    }
  #else
    unsigned long long positive = 0, negative = 0;
    for (int i = 0; i < std::min((size_t)num_obbs, links.size()); i++) {
      if (overlaps[i]) {
        negative++;
      } else {
        positive++;
      }
    }
  #endif

  free(overlaps);
  std::cout << "\tPositive: " << positive << " Negative: " << negative
            << std::endl;
  cudaFree(d_srams);
  cudaFree(d_obbs);
  cudaFree(d_overlaps);
}

__global__ void traverse_gpu2(struct geometry::AABB<double> *boxes,
                              int num_boxes,
                              struct geometry::OBB<double> *d_obbs,
                              unsigned int num_obbs, int *overlaps) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int box_idx = tid / num_obbs;
  int obb_idx = tid % num_obbs;
  if (box_idx < num_boxes) {
    if (cudaOverlapMod(d_obbs[obb_idx], boxes[box_idx])) {
      overlaps[obb_idx * num_boxes + box_idx] = 1;
    } else {
      overlaps[obb_idx * num_boxes + box_idx] = 0;
    }
  }
  
  // int link = blockIdx.x % num_obbs;
  // int idx = threadIdx.x;
  // if (idx < num_boxes) {
  //   if (cudaOverlapMod(d_obbs[link], boxes[idx])) {
  //     overlaps[link * num_boxes + idx] = 1;
  //   } else {
  //     overlaps[link * num_boxes + idx] = 0;
  //   }
  // }
}

void run_gpu2(unsigned int num_obbs, int sram_idx) {
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
    octree_size = 1557;
  } else if (sram_idx == 2) {
    std::cout << "Using dresser task oriented octree" << std::endl;
    nodes = mpinet_dresser_octree[0];
    octree_size = 1557;
  } else if (sram_idx == 3) {
    std::cout << "Using merged cubby task oriented octree" << std::endl;
    nodes = mpinet_merged_cubby_octree[0];
    octree_size = 1442;
  } else if (sram_idx == 4) {
    std::cout << "Using tabletop task oriented octree" << std::endl;
    nodes = mpinet_tabletop_octree[0];
    octree_size = 1261;
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
  std::cout << "Number of boxes: " << boxes.size() << std::endl;

  // copy boxes to gpu
  int size = sizeof(struct geometry::AABB<double>) * boxes.size();
  struct geometry::AABB<double> *d_boxes;
  cudaMalloc(&d_boxes, size);
  cudaMemcpy(d_boxes, boxes.data(), size, cudaMemcpyHostToDevice);

  // copy robot links to gpu
  std::vector<struct geometry::OBB<double>> links;
  if (sram_idx == 0) {
    for (int i = 0; i < num_obbs; i++) {
      struct geometry::OBB<double> obb = obbs[i % 4];
      // for (int j = 0; j < 3; j++) { // disable jittering for now
      //   //   obb.c[j] = r(-16.0, 16.0);
      //   //   obb.e[j] = r(0.0, 16.0);
      //   obb.c[j] += r(-4.0, 4.0);
      //   obb.e[j] += r(0.0, 1.0);
      // }
      // eulerToMatrix(r(0.0, TWO_PI), r(0.0, TWO_PI), r(0.0, TWO_PI), obb.u);
      links.push_back(obb);
    }
  } else if (sram_idx == 1) {
    links = load_obbs("obb_files/cubby_obbs.csv");
  } else if (sram_idx == 2) {
    links = load_obbs("obb_files/dresser_obbs.csv");
  } else if (sram_idx == 3) {
    links = load_obbs("obb_files/merged_cubby_obbs.csv");
  } else if (sram_idx == 4) {
    links = load_obbs("obb_files/tabletop_obbs.csv");
  }

  size = sizeof(struct geometry::OBB<double>) * std::min((size_t)num_obbs, links.size());
  struct geometry::OBB<double> *d_obbs;
  cudaMalloc(&d_obbs, size);
  cudaMemcpy(d_obbs, links.data(), size, cudaMemcpyHostToDevice);

  // allocate memory for overlap results
  int *d_overlaps;
  cudaMalloc(&d_overlaps, sizeof(int) * std::min((size_t)num_obbs, links.size()) * boxes.size());

  // run the kernel
  // std::cerr << "start running with " << std::min((size_t)num_obbs, links.size()) << std::endl;
  int N = 1; // tommy: used to be 10, but dont need to average anymore
  unsigned long long count = 0;
  for (int i = 0; i < N; i++) {
    auto start = high_resolution_clock::now();
    // traverse_gpu2<<<num_obbs, boxes.size()>>>(d_boxes, boxes.size(), d_obbs,
    //                                           num_obbs, d_overlaps);
    traverse_gpu2<<<std::ceil(std::min((size_t)num_obbs, links.size()) * boxes.size() / 256), 256>>>(d_boxes, boxes.size(), d_obbs,
                                              std::min((size_t)num_obbs, links.size()), d_overlaps);
    // checkCudaErrors( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    auto duration =
        duration_cast<microseconds>(high_resolution_clock::now() - start);
    count += duration.count();
  }
  std::cout << "Num obbs: " << std::min((size_t)num_obbs, links.size()) << " Kernel execution: " << (count / N)
            << "us" << std::endl;

  int *overlaps = (int *)malloc(sizeof(int) * std::min((size_t)num_obbs, links.size()) * boxes.size());
  checkCudaErrors(cudaMemcpy(overlaps, d_overlaps, sizeof(int) * std::min((size_t)num_obbs, links.size()) * boxes.size(),
             cudaMemcpyDeviceToHost));

  unsigned long long negative = 0, positive = 0;
  for (int i = 0; i < std::min((size_t)num_obbs, links.size()); i++) {
    bool ok = true;
    for (int j = 0; j < boxes.size(); j++) {
      if (overlaps[i * boxes.size() + j]) {
        ok = false;
      }
    }
    if (ok) {
      positive++;
    } else {
      negative++;
    }
  }
  free(overlaps);
  std::cout << "\tPositive: " << positive << " Negative: " << negative
            << std::endl;

  cudaFree(d_boxes);
  cudaFree(d_obbs);
  cudaFree(d_overlaps);
}