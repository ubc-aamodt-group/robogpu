// Copyright (c) 2022, Mohammadreza Saed, Yuan Hsi Chou, Lufei Liu, Tor M. Aamodt,
// The University of British Columbia
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "vulkan_ray_tracing.h"
#include "vulkan_rt_thread_data.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include <assert.h>
#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#define __CUDA_RUNTIME_API_H__
// clang-format off
#include "host_defines.h"
#include "builtin_types.h"
#include "driver_types.h"
#include "../../libcuda/cuda_api.h"
#include "cudaProfiler.h"
// clang-format on
#if (CUDART_VERSION < 8000)
#include "__cudaFatFormat.h"
#endif

#include "../../libcuda/gpgpu_context.h"
#include "../../libcuda/cuda_api_object.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../cuda-sim/ptx_loader.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx_ir.h"
#include "../cuda-sim/ptx_parser.h"
#include "../gpgpusim_entrypoint.h"
#include "../stream_manager.h"
#include "../abstract_hardware_model.h"
#include "vulkan_acceleration_structure_util.h"
#include "../gpgpu-sim/vector-math.h"

#if defined(MESA_USE_LVPIPE_DRIVER)
#include "lvp_private.h"
#endif 
//#include "intel_image_util.h"
#include "astc_decomp.h"

// #define HAVE_PTHREAD
// #define UTIL_ARCH_LITTLE_ENDIAN 1
// #define UTIL_ARCH_BIG_ENDIAN 0
// #define signbit signbit

// #define UINT_MAX 65535
// #define GLuint MESA_GLuint
// // #include "isl/isl.h"
// // #include "isl/isl_tiled_memcpy.c"
// #include "vulkan/anv_private.h"
// #undef GLuint

// #undef HAVE_PTHREAD
// #undef UTIL_ARCH_LITTLE_ENDIAN
// #undef UTIL_ARCH_BIG_ENDIAN
// #undef signbit

// #include "vulkan/anv_public.h"

#if defined(MESA_USE_INTEL_DRIVER)
#include "intel_image.h"
#elif defined(MESA_USE_LVPIPE_DRIVER)
// #include "lvp_image.h"
#endif

#define RCP_ERROR 0.0003662109375
// #define INJECT_RCP_ERROR
#define TWO_PI (6.283185308)

// #include "anv_include.h"

VkRayTracingPipelineCreateInfoKHR* VulkanRayTracing::pCreateInfos = NULL;
VkAccelerationStructureGeometryKHR* VulkanRayTracing::pGeometries = NULL;
uint32_t VulkanRayTracing::geometryCount = 0;
VkAccelerationStructureKHR VulkanRayTracing::topLevelAS = NULL;
std::vector<std::vector<Descriptor> > VulkanRayTracing::descriptors;
std::ofstream VulkanRayTracing::imageFile;
std::map<std::string, std::string> outputImages;
bool VulkanRayTracing::firstTime = true;
std::vector<shader_stage_info> VulkanRayTracing::shaders;
// RayDebugGPUData VulkanRayTracing::rayDebugGPUData[2000][2000] = {0};
struct DESCRIPTOR_SET_STRUCT* VulkanRayTracing::descriptorSet = NULL;
void* VulkanRayTracing::launcher_descriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS] = {NULL};
void* VulkanRayTracing::launcher_deviceDescriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS] = {NULL};
std::vector<void*> VulkanRayTracing::child_addrs_from_driver;
std::map<void*, void*> VulkanRayTracing::blas_addr_map;
void* VulkanRayTracing::tlas_addr;

bool VulkanRayTracing::dumped = false;

const bool dump_trace = false;

bool VulkanRayTracing::_init_ = false;
warp_intersection_table *** VulkanRayTracing::intersection_table;
warp_intersection_table *** VulkanRayTracing::anyhit_table;
IntersectionTableType VulkanRayTracing::intersectionTableType = IntersectionTableType::Baseline;

float get_norm(float4 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}
float get_norm(float3 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float4 normalized(float4 v)
{
    float norm = get_norm(v);
    return {v.x / norm, v.y / norm, v.z / norm, v.w / norm};
}
float3 normalized(float3 v)
{
    float norm = get_norm(v);
    return {v.x / norm, v.y / norm, v.z / norm};
}

Ray make_transformed_ray(Ray &ray, float4x4 matrix, float *worldToObject_tMultiplier)
{
    Ray transformedRay;
    float4 transformedOrigin4 = matrix * float4({ray.get_origin().x, ray.get_origin().y, ray.get_origin().z, 1});
    float4 transformedDirection4 = matrix * float4({ray.get_direction().x, ray.get_direction().y, ray.get_direction().z, 0});

    float3 transformedOrigin = {transformedOrigin4.x / transformedOrigin4.w, transformedOrigin4.y / transformedOrigin4.w, transformedOrigin4.z / transformedOrigin4.w};
    float3 transformedDirection = {transformedDirection4.x, transformedDirection4.y, transformedDirection4.z};
    *worldToObject_tMultiplier = get_norm(transformedDirection);
    transformedDirection = normalized(transformedDirection);

    transformedRay.make_ray(transformedOrigin, transformedDirection, ray.get_tmin() * (*worldToObject_tMultiplier), ray.get_tmax() * (*worldToObject_tMultiplier));
    return transformedRay;
}

float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MIN_MAX(a0, a1, d);
	float t2 = MIN_MAX(b0, b1, t1);
	float t3 = MIN_MAX(c0, c1, t2);
	return t3;
}

float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MAX_MIN(a0, a1, d);
	float t2 = MAX_MIN(b0, b1, t1);
	float t3 = MAX_MIN(c0, c1, t2);
	return t3;
}

float3 get_t_bound(float3 box, float3 origin, float3 idirection)
{
    // // Avoid div by zero, returns 1/2^80, an extremely small number
    // const float ooeps = exp2f(-80.0f);

    // // Calculate inverse direction
    // float3 idir;
    // idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    // idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    // idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));

    // Calculate bounds
    float3 result;
    result.x = (box.x - origin.x) * idirection.x;
    result.y = (box.y - origin.y) * idirection.y;
    result.z = (box.z - origin.z) * idirection.z;

    // Return
    return result;
}

float3 calculate_idir(float3 direction) {
    // Avoid div by zero, returns 1/2^80, an extremely small number
    const float ooeps = exp2f(-80.0f);

    // Calculate inverse direction
    float3 idir;
    // TODO: is this wrong?
    idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));

    // idir.x = fabsf(direction.x) > ooeps ? 1.0f / direction.x : copysignf(ooeps, direction.x);
    // idir.y = fabsf(direction.y) > ooeps ? 1.0f / direction.y : copysignf(ooeps, direction.y);
    // idir.z = fabsf(direction.z) > ooeps ? 1.0f / direction.z : copysignf(ooeps, direction.z);

    // Inject error
    // random number between -error and error
#ifdef INJECT_RCP_ERROR
    float injection = (float)rand() / RAND_MAX * RCP_ERROR * 2 - RCP_ERROR + 1;
    assert(injection >= (1 - RCP_ERROR) && injection <= (1 + RCP_ERROR));
    idir = idir * injection;
#endif

    return idir;
}

bool ray_box_test(float3 low, float3 high, float3 idirection, float3 origin, float tmin, float tmax, float& thit)
{
	// const float3 lo = Low * InvDir - Ood;
	// const float3 hi = High * InvDir - Ood;
    float3 lo = get_t_bound(low, origin, idirection);
    float3 hi = get_t_bound(high, origin, idirection);

    // QUESTION: max value does not match rtao benchmark, rtao benchmark converts float to int with __float_as_int
    // i.e. __float_as_int: -110.704826 => -1025677090, -24.690834 => -1044019502

	// const float slabMin = tMinFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMin);
	// const float slabMax = tMaxFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMax);
    float min = magic_max7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmin);
    float max = magic_min7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmax);

	// OutIntersectionDist = slabMin;
    thit = min;

	// return slabMin <= slabMax;
    return (min <= max);
}

geometry::octree_node read_octree_node(memory_space *mem, void* node_addr) {
    RT_DPRINTF("Process node 0x%x ", node_addr);
    geometry::octree_node current_node;
    mem->read(node_addr, sizeof(int), &current_node.child_address);
    node_addr += sizeof(int);

    for (int i=0; i<8; i++) {
        mem->read(node_addr, sizeof(int), &current_node.child_status[i]);
        RT_DPRINTF("Reading address 0x%x for child_status[%d], value %d\n", node_addr, i, current_node.child_status[i]);
        node_addr += sizeof(int);
    }
    node_addr += sizeof(int); // Skip padding since doubles are aligned to 8 bytes

    for (int i=0; i<8; i++) {
        for (int j=0; j<3; j++) {
            void* ptr = malloc(sizeof(double));
            mem->read(node_addr, sizeof(double), ptr);
            RT_DPRINTF("Reading address 0x%x for children_box[%d]c[%d], value %f\n", node_addr, i, j, *(double*)ptr);
            current_node.children_box[i].c[j] = *(double*)ptr;
            node_addr += sizeof(double);
            free(ptr);
        }
        for (int j=0; j<3; j++) {
            void* ptr = malloc(sizeof(double));
            mem->read(node_addr, sizeof(double), ptr);
            RT_DPRINTF("Reading address 0x%x for children_box[%d]e[%d], value %f\n", node_addr, i, j, *(double*)ptr);
            current_node.children_box[i].e[j] = *(double*)ptr;
            node_addr += sizeof(double);
            free(ptr);
        }
    }

    mem->read(node_addr, sizeof(int), &current_node.level);
    node_addr += 2 * sizeof(int); // 2 additional padding, total size of geometry::octree_node is 432 bytes

    // print out the node
    RT_DPRINTF("Node: %d\n", current_node.child_address);
    RT_DPRINTF("Children Status: ");
    for (int i=0; i<8; i++) {
        RT_DPRINTF("%d ", current_node.child_status[i]);
    }
    RT_DPRINTF("\n");
    RT_DPRINTF("Children Box:\n");
    for (int i=0; i<8; i++) {
        RT_DPRINTF("Centers: [%f %f %f], Extents: [%f %f %f]\n", current_node.children_box[i].c[0], current_node.children_box[i].c[1], current_node.children_box[i].c[2], current_node.children_box[i].e[0], current_node.children_box[i].e[1], current_node.children_box[i].e[2]);
    }
    RT_DPRINTF("%d\n", current_node.level);

    return current_node;
}

// TODO: Figure out how to configure robot
const int robotLength = 10, robotWidth = 4;

void calculate_next_node_xy(int direction, int& nx, int& ny, int X, int Y, int dX, int dY) {
    const int NX[4] = { X, X, X + dX, X - dX };
    const int NY[4] = { Y, Y, Y + dY, Y - dY };

    assert(direction >= 0 && direction < 4);
    nx = NX[direction];
    ny = NY[direction];
}

//  The collision index should be "reversed" to actually match the baseline
#define REVERSE_COLLISION_IDX true
void calculate_direction_index(int collision_index, int& direction, int& i, int& j) {
    // Direction -> [0-3]
    // i -> [0, d_robotLength] inclusive
    // j -> [0, d_robotWidth] inclusive
    // collision_index = [direction][i][j] = direction * (robotLength + 1) * (robotWidth + 1) + i * (robotWidth + 1) + j

    direction = collision_index / ((robotLength + 1) * (robotWidth + 1));
    int remainder = collision_index % ((robotLength + 1) * (robotWidth + 1));

    if (REVERSE_COLLISION_IDX) {
        int max_direction = 3;
        direction = max_direction - direction; // Reverse the direction to match baseline

        int max_remainder = (robotLength + 1) * (robotWidth + 1) - 1;
        remainder = max_remainder - remainder; // Reverse the remainder to match baseline
    }

    i = remainder / (robotWidth + 1);
    j = remainder % (robotWidth + 1);
    assert(direction >= 0 && direction < 4);
    assert(i >= 0 && i <= robotLength);
    assert(j >= 0 && j <= robotWidth);
}

void track_collision_transaction(std::vector<MemoryTransactionRecord> &transactions, void* last_node_addr, CollisionResult result) {
    bool allow_return = GPGPU_Context()->func_sim->g_rt_allow_return;

    if (!allow_return) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;
    } else if (result == CollisionResult::HIT) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;
    } else if (result == CollisionResult::AXIS1_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET1)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET1)]++;
    } else if (result == CollisionResult::AXIS2_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET2)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET2)]++;
    } else if (result == CollisionResult::AXIS3_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET3)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET3)]++;
    } else if (result == CollisionResult::AXIS4_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET4)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET4)]++;
    } else if (result == CollisionResult::AXIS5_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET5)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET5)]++;
    } else if (result == CollisionResult::AXIS6_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET6)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET6)]++;
    } else if (result == CollisionResult::AXIS7_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET7)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET7)]++;
    } else if (result == CollisionResult::AXIS8_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET8)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET8)]++;
    } else if (result == CollisionResult::AXIS9_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET9)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET9)]++;
    } else if (result == CollisionResult::AXIS10_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET10)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET10)]++;
    } else if (result == CollisionResult::AXIS11_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET11)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET11)]++;
    } else if (result == CollisionResult::AXIS12_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET12)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET12)]++;
    } else if (result == CollisionResult::AXIS13_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET13)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET13)]++;
    } else if (result == CollisionResult::AXIS14_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_RET14)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_RET14)]++;
    } else if (result == CollisionResult::AXIS15_CLEAR) {
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;
    } else if (result == CollisionResult::OUTER_SPHERE_MISS) {
        // Outer sphere
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_OUTSPHERE)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_OUTSPHERE)]++;
    } else if (result == CollisionResult::INNER_SPHERE_HIT) {
        // Inner sphere
        transactions.push_back(MemoryTransactionRecord(
            last_node_addr,
            128,
            TransactionType::BVH_INTERNAL_NODE_INSPHERE)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE_INSPHERE)]++;
    } else {
        printf("gpgpusim: ERROR! Unrecognized return value %d.\n", result);
    }
}

struct BoundingBox {
    int x_min, x_max, y_min, y_max; // Boundaries of the box in the grid
};
  
typedef struct quadtree_node {
    BoundingBox bbox;          // Boundaries of this node
    int32_t children[4];  // 4 children: top-left, top-right, bottom-left, bottom-right
    double occData; // Occupancy data within the bounding box
    bool isLeaf;               // Flag indicating if it's a leaf node (no further subdivision)
} quadtree_node;

quadtree_node read_quadtree_node(memory_space *mem, void* node_addr) {
    RT_DPRINTF("Process node 0x%x ", node_addr);
    quadtree_node current_node;
    mem->read(node_addr, sizeof(quadtree_node), &current_node);
    
    RT_DPRINTF("(%d %d %d %d) ", current_node.bbox.x_min, current_node.bbox.x_max, current_node.bbox.y_min, current_node.bbox.y_max);
    RT_DPRINTF("(%d %d %d %d) ", current_node.children[0], current_node.children[1], current_node.children[2], current_node.children[3]);
    RT_DPRINTF("(%f) ", current_node.occData);
    RT_DPRINTF("(%d) ", current_node.isLeaf);
    RT_DPRINTF("\n");
    return current_node;
}

bool ray_box_test_2d(quadtree_node node, double ray_orig_x, double ray_orig_y, double ray_dir_x, double ray_dir_y, double& thit) {
    RT_DPRINTF("RayBoxTest: (%d, %d, %d, %d)\n", node.bbox.x_min, node.bbox.x_max, node.bbox.y_min, node.bbox.y_max);
    // Check if the ray intersects with the bounding box of the node
    double txmin = (node.bbox.x_min - ray_orig_x) / ray_dir_x;
    double txmax = (node.bbox.x_max - ray_orig_x) / ray_dir_x;

    double tymin = (node.bbox.y_min - ray_orig_y) / ray_dir_y;
    double tymax = (node.bbox.y_max - ray_orig_y) / ray_dir_y;

    double t_entry = std::max(std::min(txmin, txmax), std::min(tymin, tymax));
    double t_exit  = std::min(std::max(txmin, txmax), std::max(tymin, tymax));

    bool hit = t_entry <= t_exit && t_exit >= 0.0f;

    if (hit) {
        // Update the hit distance
        thit = std::min(t_entry, t_exit) >= 0.0f ? std::min(t_entry, t_exit) : 0.0;
    }

    RT_DPRINTF("Hit: %d, t_entry: %.2f, t_exit: %.2f, thit = %.2f\n", hit, t_entry, t_exit, thit);
    return hit;
}
typedef struct btree_node {
    bool is_leaf;
    uint32_t n_children;
    int* keys;
    int* child_indices;
} btree_node;

btree_node read_btree_node(memory_space *mem, void* node_addr) {
    RT_DPRINTF("Process node 0x%x ", node_addr);
    btree_node current_node;
    mem->read(node_addr, sizeof(uint32_t), &current_node.is_leaf);
    node_addr += sizeof(uint32_t);
    mem->read(node_addr, sizeof(uint32_t), &current_node.n_children);
    node_addr += sizeof(uint32_t);
    RT_DPRINTF("(%d children) %d: ", current_node.n_children, current_node.is_leaf);

    assert(current_node.n_children <= 9);

    current_node.keys = (int *)calloc(current_node.n_children, sizeof(int));
    current_node.child_indices = (int *)calloc(current_node.n_children + 1, sizeof(int));
    
    for (uint32_t i=0; i<current_node.n_children; i++) {
        mem->read(node_addr + i * sizeof(uint32_t), sizeof(uint32_t), &current_node.keys[i]);
        mem->read(node_addr + current_node.n_children * sizeof(uint32_t) + i * sizeof(uint32_t), sizeof(uint32_t), &current_node.child_indices[i]);
    }
    mem->read(node_addr + 2 * current_node.n_children * sizeof(uint32_t), sizeof(uint32_t), &current_node.child_indices[current_node.n_children]);

    for (uint32_t i=0; i<current_node.n_children; i++) {
        RT_DPRINTF("[%6d %6d] ", current_node.keys[i], current_node.child_indices[i]);
    }
    RT_DPRINTF("[%6d] ", current_node.child_indices[current_node.n_children]);
    RT_DPRINTF("\n");
    return current_node;
}

void reorder_child_nodes(child_node* children, std::map<float, child_node>& reordered_children, int key) {
    for (int i=0; i<6; i++) {
        // Only add childs that are hit
        if (children[i].hit) {
            // No reordering
            if (key == 0) {
                reordered_children[(float)i] = children[i];
            }

            // Reorder by surface area
            else if (key == 1) {
                // Duplicate SA? 
                while (reordered_children.find(children[i].surface_area) != reordered_children.end()) {
                    children[i].surface_area *= 1.0001;
                }
                reordered_children[children[i].surface_area] = children[i];
            }

            // Reorder by thit
            else if (key == 2) {
                while (reordered_children.find(1.0f/children[i].thit) != reordered_children.end()) {
                    children[i].thit *= 1.0001;
                }
                reordered_children[1.0f/children[i].thit] = children[i];
            }

            // Random
            else if (key == 3) {
                int val = rand() % 100;
                while (reordered_children.find(val) != reordered_children.end()) {
                    val = rand() % 100;
                }
                reordered_children[float(val)] = children[i];
            }

            else {
                assert(0);
            }
        }
    }
}

typedef struct StackEntry {
    uint8_t* addr;
    bool topLevel;
    bool leaf;
    float depth;
    StackEntry(uint8_t* addr, bool topLevel, bool leaf): addr(addr), topLevel(topLevel), leaf(leaf), depth(0) {}
    StackEntry(uint8_t* addr, bool topLevel, bool leaf, float depth): addr(addr), topLevel(topLevel), leaf(leaf), depth(depth) {}
} StackEntry;

// Search Conditions (Ray - 48 bytes)
typedef struct TreeSearchConditions {
    uint32_t values[16]; // was 12
    // RTAO:
    //  [0-3] float4 origin_tmin;
    //  [4-7] float4 dir_tmax;
    //  [8]   void* triangle_addr;
    //  [9]   bool anyhit;
} TreeSearchConditions;

// Search results (HitPayload - 32 bytes)
typedef struct TreeSearchResults {
    uint32_t values[8];
    // RTAO:
    // [0]   bool valid;
    // [1]   float thit;
    // [2-4] float3 bary;
    // [5]   int tri_id;
} TreeSearchResults;

void read_traverse_tree_args(const ptx_instruction *pI, ptx_thread_info *thread, TreeSearchConditions& conditions, addr_t& TreeRootAddr, unsigned& node_processing_configuration, unsigned& leaf_processing_configuration, TreeSearchResults** results_payload_addr) {

    // Read input arguments
    int arg = 0;

    // Tree search conditions
    const operand_info &param_op1 = pI->operand_lookup(arg + 1);    
    addr_t from_addr = param_op1.get_symbol()->get_address();
    unsigned size = sizeof(TreeSearchConditions);
    thread->m_local_mem->read(from_addr, size, &conditions);

    // Tree root
    arg++;
    const operand_info &param_op2 = pI->operand_lookup(arg + 1);    
    from_addr = param_op2.get_symbol()->get_address();
    size = sizeof(void *);
    thread->m_local_mem->read(from_addr, size, &TreeRootAddr);

    // Tree search results
    arg++;
    const operand_info &param_op5 = pI->operand_lookup(arg + 1);    
    from_addr = param_op5.get_symbol()->get_address();
    size = sizeof(TreeSearchResults *);
    thread->m_local_mem->read(from_addr, size, results_payload_addr);
    
    // Node processing
    arg++;
    const operand_info &param_op3 = pI->operand_lookup(arg + 1);    
    from_addr = param_op3.get_symbol()->get_address();
    size = sizeof(unsigned);
    thread->m_local_mem->read(from_addr, size, &node_processing_configuration);

    // Leaf processing
    arg++;
    const operand_info &param_op4 = pI->operand_lookup(arg + 1);    
    from_addr = param_op4.get_symbol()->get_address();
    size = sizeof(unsigned);
    thread->m_local_mem->read(from_addr, size, &leaf_processing_configuration);

}

bool rt_decode_node(unsigned node_type, memory_space *mem, StackEntry node, addr_t root_node_addr, TreeSearchConditions conditions) {
    void* node_addr = node.addr;
    switch (node_type) {

        // CUDA magic ray tracing
        case 0: {
            addr_t tri_addr = conditions.values[8];
            int index = (int)node_addr - (int)tri_addr;
            return index >= 0;
        }

        // B-tree
        case 1: {
            btree_node node = read_btree_node(mem, node_addr);
            if (node.is_leaf == 1) {
                return true;
            }
            else {
                uint32_t search_key = conditions.values[0];
                for (uint32_t i=0; i<node.n_children; i++) {
                    if (node.keys[i] == search_key) {
                        RT_DPRINTF("Not leaf, but %d matches %d\n", node.keys[i], search_key);
                        return true;
                    }
                }
                return false;
            }
            break;
        }
        
        // B+ tree
        case 2: {
            btree_node node = read_btree_node(mem, node_addr);
            if (node.is_leaf == 1) {
                return true;
            }
            return false;
            break;
        }

        // N body
        case 3: 
        case 7: {
            // TODO: Fix this
            if (node.leaf) {
                RT_DPRINTF("Leaf node\n")
                return true;
            }
            else {
                RT_DPRINTF("Inner node\n")
                return false;
            }
            break;
        }

        // 3D N body
        case 4: {
            // TODO: Fix this
            if (node.leaf) {
                RT_DPRINTF("Leaf node\n")
                return true;
            }
            else {
                RT_DPRINTF("Inner node\n")
                return false;
            }
            break;
        }

        // Collision detection
        case 8: {
            return false; // every node is an inner node, they all perform OBB vs AABB intersection tests, treat them all the same
            break;
        }

        case 9: {
            quadtree_node node = read_quadtree_node(mem, node_addr);
            return node.isLeaf;
            break;
        }

        case 10: {
            // Root node is never leaf
            if (node_addr == root_node_addr) {
                RT_DPRINTF("Root node.\n");
                return false;
            }
            else if (node.leaf) {
                RT_DPRINTF("Out of bounds termination leaf node.\n");
                return true;
            }

            double occ;
            double minProbability = 0.35;

            mem->read(node_addr, sizeof(double), &occ);
            RT_DPRINTF("Occupancy: %5.3f\n", occ);
            // Occupied; Leaf node.
            if (occ == -1 || occ >= minProbability) {
                return true;
            }
            return false;
            break;
        }

        case 12: {
            int collision_index = conditions.values[14];
            assert(collision_index >= 0);

            if (collision_index == 0) {
                RT_DPRINTF("Collision index is 0, this is a leaf node.\n");
                return true;
            }
            else {
                RT_DPRINTF("Collision index is %d, this is an inner node.\n", collision_index);
                return false;
            }
            break;
        }

        default:
            printf("gpgpusim: ERROR! Unrecognized node type.\n");
            abort();
    }
}

std::list<StackEntry> rt_process_inner_node(unsigned node_type, memory_space *mem, StackEntry node, addr_t root_node_addr, TreeSearchConditions& conditions, std::vector<MemoryTransactionRecord> &transactions, bool &terminate) {
    void* node_addr = node.addr;
    std::list<StackEntry> next_nodes;
    switch (node_type) {

        // CUDA magic ray tracing
        case 0: {

            transactions.push_back(MemoryTransactionRecord(
                node_addr,
                64, 
                TransactionType::BVH_INTERNAL_NODE)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;


            // Read node
            float4 n0xy, n1xy, n01z;
            mem->read(node_addr, sizeof(float4), &n0xy);
            mem->read(node_addr + sizeof(float4), sizeof(float4), &n1xy);
            mem->read(node_addr + 2*sizeof(float4), sizeof(float4), &n01z);

            // Reorganize
            float3 n0lo, n0hi, n1lo, n1hi;
            n0lo = {n0xy.x, n0xy.z, n01z.x};
            n0hi = {n0xy.y, n0xy.w, n01z.y};
            n1lo = {n1xy.x, n1xy.z, n01z.z};
            n1hi = {n1xy.y, n1xy.w, n01z.w};

            float thit0, thit1;
            float tmin = *(float*)&conditions.values[3];
            float tmax = *(float*)&conditions.values[7];
            float3 origin = {*(float*)&conditions.values[0], *(float*)&conditions.values[1], *(float*)&conditions.values[2]};
            float3 direction = {*(float*)&conditions.values[4], *(float*)&conditions.values[5], *(float*)&conditions.values[6]};

            float3 idir = calculate_idir(direction);
            bool child0_hit = ray_box_test(n0lo, n0hi, idir, origin, tmin, tmax, thit0);
            bool child1_hit = ray_box_test(n1lo, n1hi, idir, origin, tmin, tmax, thit1);

            addr_t child0_addr, child1_addr;
            mem->read(node_addr + 3*sizeof(float4), sizeof(int), &child0_addr);
            mem->read(node_addr + 3*sizeof(float4) + sizeof(int), sizeof(int), &child1_addr);

            // Traverse depth first
            addr_t next_addr;
            if (child0_hit) {
                if ((int)child0_addr < 0) {
                    addr_t tri_base_addr = (addr_t)conditions.values[8];
                    next_addr = ~child0_addr;
                    next_addr *= 0x10;
                    next_addr += tri_base_addr;
                }
                else {
                    next_addr = root_node_addr + (int)child0_addr*0x10;
                }
                next_nodes.push_front(StackEntry(next_addr, true, (int)child0_addr < 0));
            }
            if (child1_hit) {
                if ((int)child1_addr < 0) {
                    addr_t tri_base_addr = (addr_t)conditions.values[8];
                    next_addr = ~child1_addr;
                    next_addr *= 0x10;
                    next_addr += tri_base_addr;
                }
                else {
                    next_addr = root_node_addr + (int)child1_addr*0x10;
                }
                next_nodes.push_front(StackEntry(next_addr, true, (int)child1_addr < 0));
            }

            break;
        }

        // B-tree and B+ tree
        case 1: 
        case 2: {
            uint32_t search_key = conditions.values[0];
            btree_node node = read_btree_node(mem, node_addr);

            // First transaction to figure out how many children this node has
            transactions.push_back(MemoryTransactionRecord(
                node_addr,
                sizeof(uint32_t) * 2, // for decoding, read is_leaf and n_children
                TransactionType::BVH_STRUCTURE)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_STRUCTURE)]++;

            // Add transaction record for accessing keys and child indices
            transactions.push_back(MemoryTransactionRecord(
                node_addr + 2 * sizeof(uint32_t), // starts at base addr + 2
                sizeof(uint32_t) * (node.n_children) * 2 + sizeof(uint32_t), // n keys + (n+1) child indices
                TransactionType::BVH_INTERNAL_NODE)
            );

            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;

            uint32_t i_child;
            for (i_child = 0; i_child < node.n_children; i_child++) {
                if (node.keys[i_child] == search_key) {
                    // This should not happen for B-tree (decode should classify this as leaf)
                    if (node_type == 1) {
                        printf("ERROR: Leaf node classified as inner node!\n");
                        return;
                    }
                    // In B+ tree, node_key == search_key is treated the same as node_key < search_key (do not traverse)
                }
                else if (node.keys[i_child] > search_key) {
                    RT_DPRINTF("Continuing traversal for key %d at 0x%x (%d)\n", 
                        node.keys[i_child], 
                        root_node_addr + node.child_indices[i_child] * sizeof(uint32_t), 
                        node.child_indices[i_child]);

                    next_nodes.push_front(StackEntry(
                        root_node_addr + node.child_indices[i_child] * sizeof(uint32_t), 
                        true, 0));
                    break;
                }
            }

            // search key is larger than all the children keys
            if (i_child == node.n_children) {
                RT_DPRINTF("Continuing traversal at 0x%x (%d)\n", 
                    root_node_addr + node.child_indices[node.n_children] * sizeof(uint32_t), 
                    node.child_indices[node.n_children]);

                next_nodes.push_front(StackEntry(
                    root_node_addr + node.child_indices[node.n_children] * sizeof(uint32_t), 
                    true, 0));
            }
            break;
        }

        // N body
        case 3: {
            unsigned n_children = 4;
            unsigned n = conditions.values[6];

            // Special case for root node
            if (node_addr == root_node_addr) {
                transactions.push_back(MemoryTransactionRecord(
                    node_addr,
                    sizeof(int) * n_children,
                    TransactionType::BVH_INTERNAL_NODE) 
                );
                GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;

                float depth = *(float*)&conditions.values[5];

                for (unsigned i_child = 0; i_child < n_children; i_child++) {
                    int child_node;
                    mem->read(node_addr + i_child * sizeof(int), sizeof(int), &child_node);
                    RT_DPRINTF("Read child %i: %d\n", i_child, child_node);

                    if (child_node >= 0) {
                        int index = child_node * n_children;
                        addr_t child_addr = root_node_addr + index * sizeof(int);

                        // All inner nodes
                        RT_DPRINTF("Pushing child %d 0x%x (index %d) top node depth %f\n", i_child, child_addr, index, depth);
                        next_nodes.push_front(StackEntry(child_addr, true, false, depth));
                    }
                    else {
                        RT_DPRINTF("Child %d invalid.\n", i_child);
                    }
                }
            }

            else {
                // Unconditionally push children (for now)
                for (unsigned i_child = 0; i_child < n_children; i_child++) {
                    int child_node;
                    mem->read(node_addr + i_child * sizeof(int), sizeof(int), &child_node);
                    RT_DPRINTF("Read child %i: %d\n", i_child, child_node);

                    if (child_node >= 0) {
                        if (child_node < n) {
                            addr_t child_addr = node_addr + i_child * sizeof(int);
                            int index = (child_addr - root_node_addr) / sizeof(int);
                            RT_DPRINTF("Pushing child %d 0x%x (index %d) is leaf\n", i_child, child_addr, index);
                            next_nodes.push_front(StackEntry(child_addr, true, true));
                        }
                        else {
                            // Prune distant nodes
                            const float eps2 = 0.025;
                            float node_depth = node.depth * 0.25;

                            // Read x, y position data and mass data addresses
                            addr_t x_addr = conditions.values[2];
                            addr_t y_addr = conditions.values[3];
                            addr_t mass_addr = conditions.values[4];

                            // Read x, y position data and mass data
                            float x, y, mass;
                            mem->read(x_addr + child_node * sizeof(float), sizeof(float), &x);
                            mem->read(y_addr + child_node * sizeof(float), sizeof(float), &y);
                            mem->read(mass_addr + child_node * sizeof(float), sizeof(float), &mass);

                            float pos_x = *(float*)&conditions.values[0];
                            float pos_y = *(float*)&conditions.values[1];

                            float dx = x - pos_x;
                            float dy = y - pos_y;
                            float dist = dx*dx + dy*dy + eps2;
                            RT_DPRINTF("Node depth %f, dist %f\n", node_depth, dist);

                            // Treat as leaf if distance is small enough
                            if (node_depth <= dist) {
                                addr_t child_addr = node_addr + i_child * sizeof(int);
                                int index = (child_addr - root_node_addr) / sizeof(int);
                                RT_DPRINTF("Pushing child %d 0x%x (index %d) distant leaf\n", i_child, child_addr, index);
                                next_nodes.push_front(StackEntry(child_addr, true, true));
                            }
                            else {
                                addr_t child_addr = root_node_addr + child_node * n_children * sizeof(int);
                                RT_DPRINTF("Pushing child %d 0x%x (index %d) not leaf\n", i_child, child_addr, child_node);
                                next_nodes.push_front(StackEntry(child_addr, true, false, node_depth));
                            }
                        }
                    }
                    else {
                        RT_DPRINTF("Child %d invalid.\n", i_child);
                    }
                }
            }
            break;
        }

        // 3D N body
        case 4: {
            unsigned n_children = 8;
            unsigned n = conditions.values[8];

            // Special case for root node
            if (node_addr == root_node_addr) {
                transactions.push_back(MemoryTransactionRecord(
                    node_addr,
                    sizeof(int) * n_children,
                    TransactionType::BVH_STRUCTURE) // Use BVH_STRUCTURE since this is just a decoding operation
                );
                GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;

                float depth = *(float*)&conditions.values[7];

                for (unsigned i_child = 0; i_child < n_children; i_child++) {
                    int child_node;
                    mem->read(node_addr + i_child * sizeof(int), sizeof(int), &child_node);
                    RT_DPRINTF("Read child %i: %d\n", i_child, child_node);

                    if (child_node >= 0) {
                        int index = child_node * n_children;
                        addr_t child_addr = root_node_addr + index * sizeof(int);

                        // All inner nodes
                        RT_DPRINTF("Pushing child %d 0x%x (index %d) top node depth %f\n", i_child, child_addr, index, depth);
                        next_nodes.push_front(StackEntry(child_addr, true, false, depth));
                    }
                    else {
                        RT_DPRINTF("Child %d invalid.\n", i_child);
                    }
                }
            }

            else {
                // Unconditionally push children (for now)
                for (unsigned i_child = 0; i_child < n_children; i_child++) {
                    int child_node;
                    mem->read(node_addr + i_child * sizeof(int), sizeof(int), &child_node);
                    RT_DPRINTF("Read child %i: %d\n", i_child, child_node);

                    if (child_node >= 0) {

                        // Read x, y, z position data and mass data addresses
                        addr_t x_addr = conditions.values[3];
                        addr_t y_addr = conditions.values[4];
                        addr_t z_addr = conditions.values[5];
                        addr_t mass_addr = conditions.values[6];

                        if (child_node < n) {
                            addr_t child_addr = node_addr + i_child * sizeof(int);
                            int index = (child_addr - root_node_addr) / sizeof(int);
                            RT_DPRINTF("Pushing child %d 0x%x (index %d) is leaf\n", i_child, child_addr, index);
                            next_nodes.push_front(StackEntry(child_addr, true, true));

                            // Assume "dist" part of the calculation is done here and ray-tri only handles force computation
                            transactions.push_back(MemoryTransactionRecord(
                                mass_addr + child_node * sizeof(float4),
                                sizeof(float4), 
                                TransactionType::BVH_INSTANCE_LEAF) // borrow BVH instance leaf because this transaction is neither internal nor leaf
                            );
                        }
                        else {
                            // Prune distant nodes
                            const float eps2 = 0.025;
                            float node_depth = node.depth * 0.125;

                            // Read x, y, z position data and mass data
                            float x, y, z, mass;
                            mem->read(x_addr + child_node * sizeof(float), sizeof(float), &x);
                            mem->read(y_addr + child_node * sizeof(float), sizeof(float), &y);
                            mem->read(z_addr + child_node * sizeof(float), sizeof(float), &z);
                            mem->read(mass_addr + child_node * sizeof(float), sizeof(float), &mass);
                            // TODO: (FIX) currently pretending the data structure is better and x, y, mass is stored together as an array of struct instead of struct of arrays
                            transactions.push_back(MemoryTransactionRecord(
                                mass_addr + child_node * sizeof(float4),
                                sizeof(float4), 
                                TransactionType::BVH_INSTANCE_LEAF) // borrow BVH instance leaf because this transaction is neither internal nor leaf
                            );

                            // Calculations
                            float pos_x = *(float*)&conditions.values[0];
                            float pos_y = *(float*)&conditions.values[1];
                            float pos_z = *(float*)&conditions.values[2];

                            float dx = x - pos_x;
                            float dy = y - pos_y;
                            float dz = z - pos_z;
                            float dist = dx*dx + dy*dy + dz*dz + eps2;
                            RT_DPRINTF("Node depth %f, dist %f\n", node_depth, dist);

                            // Treat as leaf if distance is small enough
                            if (node_depth <= dist) {
                                addr_t child_addr = node_addr + i_child * sizeof(int);
                                int index = (child_addr - root_node_addr) / sizeof(int);
                                RT_DPRINTF("Pushing child %d 0x%x (index %d) distant leaf\n", i_child, child_addr, index);
                                next_nodes.push_front(StackEntry(child_addr, true, true));
                            }
                            else {
                                addr_t child_addr = root_node_addr + child_node * n_children * sizeof(int);
                                RT_DPRINTF("Pushing child %d 0x%x (index %d) not leaf\n", i_child, child_addr, child_node);
                                next_nodes.push_front(StackEntry(child_addr, true, false, node_depth));
                            }
                        }
                    }
                    else {
                        RT_DPRINTF("Child %d invalid.\n", i_child);
                    }
                }
            }

            break;
        }

        // 2D N-Body on TTA
        case 7: {
            unsigned n_children = 4;
            unsigned n = conditions.values[6];

            // Special case for root node
            if (node_addr == root_node_addr) {
                transactions.push_back(MemoryTransactionRecord(
                    node_addr,
                    sizeof(int) * n_children,
                    TransactionType::BVH_INTERNAL_NODE)
                );
                GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;

                float depth = *(float*)&conditions.values[5];

                for (unsigned i_child = 0; i_child < n_children; i_child++) {
                    int child_node;
                    mem->read(node_addr + i_child * sizeof(int), sizeof(int), &child_node);
                    RT_DPRINTF("Read child %i: %d\n", i_child, child_node);

                    if (child_node >= 0) {
                        int index = child_node * n_children;
                        addr_t child_addr = root_node_addr + index * sizeof(int);

                        // All inner nodes
                        RT_DPRINTF("Pushing child %d 0x%x (index %d) top node depth %f\n", i_child, child_addr, index, depth);
                        next_nodes.push_front(StackEntry(child_addr, true, false, depth));
                    }
                    else {
                        RT_DPRINTF("Child %d invalid.\n", i_child);
                    }
                }
            }

            else {

                // Unconditionally push children (for now)
                for (unsigned i_child = 0; i_child < n_children; i_child++) {
                    int child_node;
                    mem->read(node_addr + i_child * sizeof(int), sizeof(int), &child_node);
                    RT_DPRINTF("Read child %i: %d\n", i_child, child_node);

                    if (child_node >= 0) {
                        // Read x, y position data and mass data addresses
                        addr_t x_addr = conditions.values[2];
                        addr_t y_addr = conditions.values[3];
                        addr_t mass_addr = conditions.values[4];

                        if (child_node < n) {
                            addr_t child_addr = node_addr + i_child * sizeof(int);
                            int index = (child_addr - root_node_addr) / sizeof(int);
                            RT_DPRINTF("Pushing child %d 0x%x (index %d) is leaf\n", i_child, child_addr, index);
                            next_nodes.push_front(StackEntry(child_addr, true, true));
                            transactions.push_back(MemoryTransactionRecord(
                                mass_addr + child_node * sizeof(float3),
                                sizeof(float3),
                                TransactionType::BVH_INSTANCE_LEAF) // borrow BVH instance leaf because this transaction is neither internal nor leaf
                            );
                        }
                        else {
                            // Prune distant nodes
                            const float eps2 = 0.025;
                            float node_depth = node.depth * 0.25;

                            // Read x, y position data and mass data
                            float x, y, mass;
                            mem->read(x_addr + child_node * sizeof(float), sizeof(float), &x);
                            mem->read(y_addr + child_node * sizeof(float), sizeof(float), &y);
                            mem->read(mass_addr + child_node * sizeof(float), sizeof(float), &mass);

                            // Calculations
                            float pos_x = *(float*)&conditions.values[0];
                            float pos_y = *(float*)&conditions.values[1];

                            float dx = x - pos_x;
                            float dy = y - pos_y;
                            float dist = dx*dx + dy*dy + eps2;
                            RT_DPRINTF("Node depth %f, dist %f\n", node_depth, dist);

                            // Treat as leaf if distance is small enough
                            if (node_depth <= dist) {
                                addr_t child_addr = node_addr + i_child * sizeof(int);
                                int index = (child_addr - root_node_addr) / sizeof(int);
                                RT_DPRINTF("Pushing child %d 0x%x (index %d) distant leaf\n", i_child, child_addr, index);
                                next_nodes.push_front(StackEntry(child_addr, true, true));
                                transactions.push_back(MemoryTransactionRecord(
                                    mass_addr + child_node * sizeof(float3),
                                    sizeof(float3),
                                    TransactionType::BVH_INSTANCE_LEAF) // borrow BVH instance leaf because this transaction is neither internal nor leaf
                                );
                            }
                            else {
                                addr_t child_addr = root_node_addr + child_node * n_children * sizeof(int);
                                RT_DPRINTF("Pushing child %d 0x%x (index %d) not leaf\n", i_child, child_addr, child_node);
                                next_nodes.push_front(StackEntry(child_addr, true, false, node_depth));
                            }
                        }
                    }
                    else {
                        RT_DPRINTF("Child %d invalid.\n", i_child);
                    }
                }
            }
            break;
        }

        // Collision detection
        case 8:{
            // Put conditions in an OBB struct
            geometry::OBB<float> obb;
            obb.u[0][0] = *(float*)&(conditions.values[0]);
            obb.u[0][1] = *(float*)&(conditions.values[1]);
            obb.u[0][2] = *(float*)&(conditions.values[2]);
            obb.u[1][0] = *(float*)&(conditions.values[3]);
            obb.u[1][1] = *(float*)&(conditions.values[4]);
            obb.u[1][2] = *(float*)&(conditions.values[5]);
            obb.u[2][0] = *(float*)&(conditions.values[6]);
            obb.u[2][1] = *(float*)&(conditions.values[7]);
            obb.u[2][2] = *(float*)&(conditions.values[8]);
            obb.c[0] = *(float*)&(conditions.values[9]);
            obb.c[1] = *(float*)&(conditions.values[10]);
            obb.c[2] = *(float*)&(conditions.values[11]);
            obb.e[0] = *(float*)&(conditions.values[12]);
            obb.e[1] = *(float*)&(conditions.values[13]);
            obb.e[2] = *(float*)&(conditions.values[14]);

            // Read octree node
            geometry::octree_node current_node = read_octree_node(mem, node_addr);
            
            // Transaction for reading the whole node
            // manually break this up into max 128B transactions
            // Track last part with specific return transaction type instead
            void* last_node_addr = node_addr;
            for (int i = 0; i < ((int)sizeof(geometry::octree_node) - 128); i += 128) {
                transactions.push_back(MemoryTransactionRecord(
                    node_addr + i,
                    std::min(128, (int)sizeof(geometry::octree_node) - i),
                    TransactionType::BVH_STRUCTURE)
                );
                last_node_addr = node_addr + i;
            }

            // Cast float OBB to double just for the sake of the geometry library
            geometry::OBB<double> obb_double;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    obb_double.u[i][j] = static_cast<double>(obb.u[i][j]);
                }
                obb_double.c[i] = static_cast<double>(obb.c[i]);
                obb_double.e[i] = static_cast<double>(obb.e[i]);
            }

            // Go through each children and perform AABB and OBB intersections, and push intersected childred to next_nodes
            int offset = 0;
            for (int i = 0; i < 8; i++) {
                switch (current_node.child_status[i]) {
                    case 0: { // No child
                        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_STRUCTURE)]++;
                        RT_DPRINTF("Child %d empty\n", i);
                        break;
                    }
                    case 1: {
                        CollisionResult result = geometry::collisionOverlap(obb_double, current_node.children_box[i]);

                        RT_DPRINTF("Child %d collision result with ", i);
                        RT_DPRINTF("c(%5.3f, %5.3f, %5.3f) ", current_node.children_box[i].c[0], current_node.children_box[i].c[1], current_node.children_box[i].c[2]);
                        RT_DPRINTF("e(%5.3f, %5.3f, %5.3f) ", current_node.children_box[i].e[0], current_node.children_box[i].e[1], current_node.children_box[i].e[2]);
                        RT_DPRINTF("result: %d\n", result);


                        if (result == CollisionResult::HIT || result == CollisionResult::INNER_SPHERE_HIT) { // if intersect, then traverse deeper
                            // calc child address and push to next_nodes
                            next_nodes.push_front(StackEntry(
                                root_node_addr + (current_node.child_address + offset) * sizeof(geometry::octree_node), 
                                true, 0));
                        }
                        offset++;

                        track_collision_transaction(transactions, last_node_addr, result);

                        break;
                    }
                    case 2: {
                        bool sphere_only = GPGPU_Context()->func_sim->g_rt_sphere_only;

                        CollisionResult result = geometry::collisionOverlap(obb_double, current_node.children_box[i]);
                        
                        RT_DPRINTF("Child %d collision result with ", i);
                        RT_DPRINTF("c(%5.3f, %5.3f, %5.3f) ", current_node.children_box[i].c[0], current_node.children_box[i].c[1], current_node.children_box[i].c[2]);
                        RT_DPRINTF("e(%5.3f, %5.3f, %5.3f) ", current_node.children_box[i].e[0], current_node.children_box[i].e[1], current_node.children_box[i].e[2]);
                        RT_DPRINTF("result: %d\n", result);

                        track_collision_transaction(transactions, last_node_addr, result);

                        if (result == CollisionResult::HIT || result == CollisionResult::INNER_SPHERE_HIT) { // if intersect, then we done
                            terminate = true;
                            return;
                        }
                        else if (sphere_only && result == CollisionResult::OUTER_SPHERE_MISS) { // if only sphere intersect, treat it as a hit for sphere-only mode
                            terminate = true;
                            return;
                        }
                        break;
                    }
                }
            }

            break;
        }

        case 9: {
            quadtree_node currentNode = read_quadtree_node(mem, node_addr);
            transactions.push_back(MemoryTransactionRecord(
                node_addr,
                sizeof(quadtree_node), // Might need to change this to represent bbox size of child nodes instead
                TransactionType::BVH_STRUCTURE)
            );

            double ray_orig_x, ray_orig_y, ray_dir_x, ray_dir_y;
            double max_range, resolution;
            double closest_hit;
    
            ray_orig_x = *(double*)&conditions.values[0];
            ray_orig_y = *(double*)&conditions.values[2];
            ray_dir_x = *(double*)&conditions.values[4];
            ray_dir_y = *(double*)&conditions.values[6];
            max_range = *(double*)&conditions.values[8];
            resolution = *(double*)&conditions.values[10];
            closest_hit = *(double*)&conditions.values[12];

            double t_hit = max_range;

            for (int i = 3; i >= 0; i--) {
                int32_t childNodeIdx = currentNode.children[i];
                if (childNodeIdx > 0) {
                    // Get the child node
                    quadtree_node childNode = read_quadtree_node(mem, root_node_addr + childNodeIdx * sizeof(quadtree_node));
                    if (ray_box_test_2d(childNode, ray_orig_x, ray_orig_y, ray_dir_x, ray_dir_y, t_hit)) {
                        if (t_hit < closest_hit) {
                            RT_DPRINTF("Hit child %d (index %d) at 0x%x (leaf %d), push to stack. Closest hit: %5.3f\n", i, childNodeIdx, root_node_addr + childNodeIdx * sizeof(quadtree_node), childNode.isLeaf, closest_hit);
                            // Currently push_back for debugging; Probably can change this to push_front and iteration from 0-4. 
                            next_nodes.push_back(StackEntry(
                                root_node_addr + childNodeIdx * sizeof(quadtree_node), 
                                true, childNode.isLeaf));
                        }
                    }
                }
            }

            break;
        }

        case 10: {
            double occ;
            double minProbability = 0.35;
            double maxRange = 1000.0;

            if (node_addr == root_node_addr) {
                RT_DPRINTF("Root node\n");
            }
            else {
                mem->read(node_addr, sizeof(double), &occ);
                transactions.push_back(MemoryTransactionRecord(
                    node_addr,
                    sizeof(double), 
                    TransactionType::BVH_INTERNAL_NODE)
                );
                GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;

                RT_DPRINTF("Reading occupancy: %5.3f\n", occ);
                // Leaf nodes should be filtered out.
                assert(occ != -1 && occ < minProbability);
            }

            // Figure out the next index
            double ray_orig_x, ray_orig_y, ray_dir_x, ray_dir_y;
            double dist, resolution;
            int mapSizeX, mapSizeY;

            ray_orig_x = *(double*)&conditions.values[0];
            ray_orig_y = *(double*)&conditions.values[2];
            ray_dir_x = *(double*)&conditions.values[4];
            ray_dir_y = *(double*)&conditions.values[6];
            dist = *(double*)&conditions.values[8];
            resolution = *(double*)&conditions.values[10];
            mapSizeX = *(int*)&conditions.values[14];
            mapSizeY = *(int*)&conditions.values[15];

            dist += resolution;
            ray_orig_x += ray_dir_x;
            ray_orig_y += ray_dir_y;

            // Write back
            uint32_t* base;
            base = (uint32_t *)&ray_orig_x;
            conditions.values[0] = base[0];
            conditions.values[1] = base[1];
            base = (uint32_t *)&ray_orig_y;
            conditions.values[2] = base[0];
            conditions.values[3] = base[1];
            base = (uint32_t *)&dist;
            conditions.values[8] = base[0];
            conditions.values[9] = base[1];

            int xIdx = static_cast<int>(ray_orig_x / resolution);
            int yIdx = static_cast<int>(ray_orig_y / resolution);
            int gIdx = xIdx * mapSizeY + yIdx;

            RT_DPRINTF("Ray xy: (%5.3f, %5.3f), ray dir: (%5.3f, %5.3f), dist: %5.3f, resolution: %5.3f -> occ[%d, %d] \n", ray_orig_x, ray_orig_y, ray_dir_x, ray_dir_y, dist, resolution, xIdx, yIdx);

            // Check for bounds
            if (dist >= maxRange || xIdx >= mapSizeX || yIdx >= mapSizeY ||
                xIdx < 0 || yIdx < 0) {
                RT_DPRINTF("Out of bounds [%d, %d] - [%d, %d] (dist: %5.3f), terminate and push leaf node\n", xIdx, yIdx, mapSizeX, mapSizeY, dist);
                next_nodes.push_back(StackEntry(
                    root_node_addr + gIdx * sizeof(double), 
                    true, 1));
            }
            else {
                RT_DPRINTF("Pushing node %d - [%d, %d] at 0x%x\n", gIdx, xIdx, yIdx, root_node_addr + gIdx * sizeof(double));
                next_nodes.push_back(StackEntry(
                    root_node_addr + gIdx * sizeof(double), 
                    true, 0));
            }

            break;
        }

        case 12: {
            int X = conditions.values[0];
            int Y = conditions.values[1];
            int dX = conditions.values[2];
            int dY = conditions.values[3];
            int collision_index = conditions.values[14];

            int direction, i, j;
            calculate_direction_index(collision_index, direction, i, j);
            int nx, ny;
            calculate_next_node_xy(direction, nx, ny, X, Y, dX, dY);

            float sine = *(float*)&conditions.values[4 + 2*direction];
            float cosine = *(float*)&conditions.values[5 + 2*direction];
            
            int mapX = *(int*)&conditions.values[12];
            int mapY = *(int*)&conditions.values[13];
            int xbar = nx + round(i * cosine);
            int ybar = ny + round(j * sine);
            
            bool in_range = (0 <= xbar && xbar < mapX && 0 <= ybar && ybar < mapY);

            int thetaIdx = (conditions.values[15] >> (4 * direction)) & 0xF; // 4 bits per direction, 16 directions total
            int gidx = (xbar * mapY + ybar) * 8 + thetaIdx;

            RT_DPRINTF("Collision index %d ([%d][%d][%d]) (%d, %d) (%4.2f, %4.2f) -> next node (%d, %d, %d)\n", collision_index, direction, i, j, nx, ny, sine, cosine, xbar, ybar, gidx);

            uint8_t occ;
            mem->read(root_node_addr + gidx*sizeof(uint8_t), sizeof(uint8_t), &occ);
            transactions.push_back(MemoryTransactionRecord(
                root_node_addr + gidx*sizeof(uint8_t),
                sizeof(uint8_t), 
                TransactionType::BVH_INTERNAL_NODE)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;

            bool direction_done;
            if (REVERSE_COLLISION_IDX) {
                direction_done = collision_index % ((robotLength + 1) * (robotWidth + 1)) == ((robotLength + 1) * (robotWidth + 1) - 1);
            }
            else {
                direction_done = collision_index % ((robotLength + 1) * (robotWidth + 1)) == 1;
            }

            if (occ == 0 || !in_range) {
                if (!in_range) {
                    RT_DPRINTF("Out of bounds (%d, %d) - (%d, %d)\n", xbar, ybar, mapX, mapY);
                }
                else {
                    RT_DPRINTF("Collision detected\n");
                }
                conditions.values[15] |= (0xF << (4 * direction)); // Mark this direction as terminated
                RT_DPRINTF("Marking direction %d as terminated: 0x%x\n", direction, conditions.values[15]);
                // Terminate this direction
                if (REVERSE_COLLISION_IDX) {
                    if (direction < 3) {
                        direction = 3 - direction;

                        collision_index = direction * (robotLength + 1) * (robotWidth + 1);
                        collision_index--;
                    }
                    else {
                        collision_index = 0;
                    }
                }
                else {
                    if (direction > 0) {
                        collision_index = direction * (robotLength + 1) * (robotWidth + 1);
                        collision_index--;
                    }
                    else {
                        collision_index = 0;
                    }  
                }
                conditions.values[14] = collision_index; // Update collision index
                RT_DPRINTF("Collision index updated to %d\n", collision_index);
            }
            else if (direction_done) {
                // If we are at the last node in this direction
                conditions.values[15] &= ~(0xF << (4 * direction)); // Mark this direction as terminated
                RT_DPRINTF("No hits in direction %d: 0x%x\n", direction, conditions.values[15]);
                conditions.values[14] = collision_index - 1; // Decrement collision index
            }
            else {
                conditions.values[14] = collision_index - 1; // Decrement collision index
            }

            // Push dummy node
            next_nodes.push_back(StackEntry(
                root_node_addr, 
                true, 0));

            break;
        }

        default:
            printf("gpgpusim: ERROR! Unrecognized node type %d.\n", node_type);
            abort();
    }

    return next_nodes;
}

TreeSearchResults rt_process_leaf_node(unsigned leaf_type, memory_space *mem, void* leaf_addr, addr_t root_node_addr, TreeSearchConditions conditions, std::vector<MemoryTransactionRecord> &transactions) {
    TreeSearchResults results;
    switch (leaf_type) {

        // CUDA magic ray tracing (pre-processed geometry)
        case 0: {

            transactions.push_back(MemoryTransactionRecord(
                leaf_addr,
                48, 
                TransactionType::BVH_QUAD_LEAF)
            );

            float tmin = *(float*)&conditions.values[3];
            float tmax = *(float*)&conditions.values[7];
            float3 origin = {*(float*)&conditions.values[0], *(float*)&conditions.values[1], *(float*)&conditions.values[2]};
            float3 direction = {*(float*)&conditions.values[4], *(float*)&conditions.values[5], *(float*)&conditions.values[6]};

            Ray ray;
            ray.make_ray(origin, direction, tmin, tmax);
            float thit = tmax;
            results.values[0] = 0;

            float4 p0, p1, p2;
            leaf_addr = (unsigned)leaf_addr & 0xffffffff;
            addr_t tri_base_addr = (addr_t)conditions.values[8];

            mem->read(leaf_addr, sizeof(float4), &p0);
            mem->read(leaf_addr + sizeof(float4), sizeof(float4), &p1);
            mem->read(leaf_addr + 2*sizeof(float4), sizeof(float4), &p2);

            while (true) {

                addr_t tri_addr = leaf_addr - tri_base_addr;
                RT_DPRINTF("triangle %d (%5.3f %5.3f %5.3f %5.3f), (%5.3f, %5.3f, %5.3f %5.3f), (%5.3f, %5.3f, %5.3f %5.3f) ", 
                    tri_addr >> 4,
                    p0.x, p0.y, p0.z, p0.w,
                    p1.x, p1.y, p1.z, p1.w,
                    p2.x, p2.y, p2.z, p2.w);

                float3 bary;
                bool hit = VulkanRayTracing::rtao_ray_triangle_test(p0, p1, p2, ray, &thit, &bary);

                if (hit) {
                    RT_DPRINTF("hit\n");
                    results.values[0] = 1;
                    results.values[1] = *(uint32_t*)&thit;
                    results.values[2] = *(uint32_t*)&bary.x;
                    results.values[3] = *(uint32_t*)&bary.y;
                    results.values[4] = *(uint32_t*)&bary.z;
                    results.values[5] = (uint32_t)(tri_addr >> 4);

                    // If anyhit ray, don't need to continue
                    if (conditions.values[9]) {
                        break;
                    }
                }
                else {
                    RT_DPRINTF("miss\n");
                }

                // Go to next triangle
                leaf_addr += 0x30;

                mem->read(leaf_addr, sizeof(float4), &p0);
                mem->read(leaf_addr + sizeof(float4), sizeof(float4), &p1);
                mem->read(leaf_addr + 2*sizeof(float4), sizeof(float4), &p2);

                // Marks an invalid triangle; leave while loop
                if (*(int*)&p0.x ==  0x80000000) {
                    break;
                }

                transactions.push_back(MemoryTransactionRecord(
                    leaf_addr,
                    48, 
                    TransactionType::BVH_QUAD_LEAF)
                );
            }
            break;
        }
        case 1: {
            uint32_t search_key = conditions.values[0];
            results.values[0] = 0;
            btree_node node = read_btree_node(mem, leaf_addr);

            transactions.push_back(MemoryTransactionRecord(
                leaf_addr,
                sizeof(uint32_t) * 2, // read is_leaf and n_children
                TransactionType::BVH_STRUCTURE)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_STRUCTURE)]++;

            transactions.push_back(MemoryTransactionRecord(
                leaf_addr + sizeof(uint32_t) * 2,
                sizeof(uint32_t) * (node.n_children), // read leaf keys
                TransactionType::BVH_QUAD_LEAF)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF)]++;

            for (uint32_t i_child = 0; i_child < node.n_children; i_child++) {
                if (node.keys[i_child] == search_key) {
                    RT_DPRINTF("Key found!\n");
                    results.values[0] = 1;
                }
            }
            break;
        }

        // N body
        case 3: {
            const float eps2 = 0.025;

            // Read x, y position data and mass data addresses
            addr_t x_addr = conditions.values[2];
            addr_t y_addr = conditions.values[3];
            addr_t mass_addr = conditions.values[4];

            // Read child node
            int child_node;
            mem->read(leaf_addr, sizeof(int), &child_node);
            RT_DPRINTF("Read leaf node: 0x%x -> (%d)\n", leaf_addr, child_node);
            transactions.push_back(MemoryTransactionRecord(
                leaf_addr,
                sizeof(int), 
                TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR)]++;

            // Read x, y position data and mass data
            float x, y, mass;
            mem->read(x_addr + child_node * sizeof(float), sizeof(float), &x);
            mem->read(y_addr + child_node * sizeof(float), sizeof(float), &y);
            mem->read(mass_addr + child_node * sizeof(float), sizeof(float), &mass);
            RT_DPRINTF("Leaf data: x=%7.3f, y=%7.3f, mass=%7.3f\n", x, y, mass);
            
            // TODO: (FIX) currently pretending the data structure is better and x, y, mass is stored together as an array of struct instead of struct of arrays
            transactions.push_back(MemoryTransactionRecord(
                mass_addr + child_node * sizeof(float3),
                sizeof(float3), 
                TransactionType::BVH_QUAD_LEAF)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF)]++;

            // Calculations
            float pos_x = *(float*)&conditions.values[0];
            float pos_y = *(float*)&conditions.values[1];

            float dx = x - pos_x;
            float dy = y - pos_y;
            float dist = 1 / sqrt(dx*dx + dy*dy + eps2);
            RT_DPRINTF("dx=%7.3f, dy=%7.3f, dist=%7.3f\n", dx, dy, dist);

            float accel_f = mass * dist * dist * dist;
            RT_DPRINTF("accel_f=%7.3f\n", accel_f);

            // Save results
            float accel_x = accel_f * dx;
            float accel_y = accel_f * dy;
            RT_DPRINTF("accel_x=%7.3f, accel_y=%7.3f\n", accel_x, accel_y);

            results.values[0] = 1; // Mark as valid
            results.values[1] = *(uint32_t*)&accel_x;
            results.values[2] = *(uint32_t*)&accel_y;

            break;
        }

        case 4: {
            const float eps2 = 0.025;

            // Read x, y position data and mass data addresses
            addr_t x_addr = conditions.values[3];
            addr_t y_addr = conditions.values[4];
            addr_t z_addr = conditions.values[5];
            addr_t mass_addr = conditions.values[6];

            // Read child node
            int child_node;
            mem->read(leaf_addr, sizeof(int), &child_node);
            RT_DPRINTF("Read leaf node: 0x%x -> (%d)\n", leaf_addr, child_node);
            transactions.push_back(MemoryTransactionRecord(
                leaf_addr,
                sizeof(int), 
                TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR)]++;

            // Read x, y position data and mass data
            float x, y, z, mass;
            mem->read(x_addr + child_node * sizeof(float), sizeof(float), &x);
            mem->read(y_addr + child_node * sizeof(float), sizeof(float), &y);
            mem->read(z_addr + child_node * sizeof(float), sizeof(float), &z);
            mem->read(mass_addr + child_node * sizeof(float), sizeof(float), &mass);
            RT_DPRINTF("Leaf data: x=%7.3f, y=%7.3f, z=%7.3f, mass=%7.3f\n", x, y, z, mass);
            
            // TODO: (FIX) currently pretending the data structure is better and x, y, mass is stored together as an array of struct instead of struct of arrays
            transactions.push_back(MemoryTransactionRecord(
                mass_addr + child_node * sizeof(float4),
                sizeof(float4), 
                TransactionType::BVH_QUAD_LEAF)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF)]++;

            // Calculations
            float pos_x = *(float*)&conditions.values[0];
            float pos_y = *(float*)&conditions.values[1];
            float pos_z = *(float*)&conditions.values[2];

            float dx = x - pos_x;
            float dy = y - pos_y;
            float dz = z - pos_z;
            float dist = 1 / sqrt(dx*dx + dy*dy + dz*dz + eps2);
            RT_DPRINTF("dx=%7.3f, dy=%7.3f, dz=%7.3f, dist=%7.3f\n", dx, dy, dz, dist);

            float accel_f = mass * dist * dist * dist;
            RT_DPRINTF("accel_f=%7.3f\n", accel_f);

            // Save results
            float accel_x = accel_f * dx;
            float accel_y = accel_f * dy;
            float accel_z = accel_f * dz;
            RT_DPRINTF("accel_x=%7.3f, accel_y=%7.3f, accel_z=%7.3f\n", accel_x, accel_y, accel_z);

            results.values[0] = 1; // Mark as valid
            results.values[1] = *(uint32_t*)&accel_x;
            results.values[2] = *(uint32_t*)&accel_y;
            results.values[3] = *(uint32_t*)&accel_z;

            break;
        }

        case 8: {
            printf("gpgpusim: ERROR! Collision detection leaf node not implemented.\n");
            abort();
        }

        case 9: {

            quadtree_node currentNode = read_quadtree_node(mem, leaf_addr);
            transactions.push_back(MemoryTransactionRecord(
                leaf_addr,
                sizeof(quadtree_node), // Might need to change this to represent bbox size of child nodes instead
                TransactionType::BVH_QUAD_LEAF)
            );

            double ray_orig_x, ray_orig_y, ray_dir_x, ray_dir_y;
            double resolution;
            double closest_hit;
            double zkt;
    
            ray_orig_x = *(double*)&conditions.values[0];
            ray_orig_y = *(double*)&conditions.values[2];
            ray_dir_x = *(double*)&conditions.values[4];
            ray_dir_y = *(double*)&conditions.values[6];
            zkt = *(double*)&conditions.values[8];
            resolution = *(double*)&conditions.values[10];
            closest_hit = *(double*)&conditions.values[12];

            double t_hit = closest_hit;

            double zHit = 10;
            double zShort = 0.01;
            double zMax = 0.1;
            double zRand = 10;
            double sigmaHit = 50.0;
            double lambdaShort = 0.1;
            double minProbability = 0.35;
            double maxRange = 1000.0;
            double sensorOffset = 25.0;

            RT_DPRINTF("Occupancy data: %5.3f\n", currentNode.occData);

            // Check occupancy
            if (currentNode.occData == -1 || currentNode.occData >= minProbability) {
                // Should be a hit. Also this assert will get the correct t_hit value.
                assert(ray_box_test_2d(currentNode, ray_orig_x, ray_orig_y, ray_dir_x, ray_dir_y, t_hit));
                RT_DPRINTF("Hit leaf node at 0x%x. t_hit = %5.3f, closest_hit = %5.3f, occData = %5.3f\n", leaf_addr, t_hit, closest_hit, currentNode.occData);
                if (t_hit < closest_hit) {
                    uint32_t *base = (uint32_t *)&t_hit;
                    results.values[4] = base[0];
                    results.values[5] = base[1];

                    // Calculate probability assuming this is the closest hit
                    double probability; 
                    double stepSize = std::sqrt(ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y);

                    // Calculate dist (only needed for probability, can discard)
                    t_hit /= stepSize;
                    t_hit++;
                    t_hit = std::ceil(t_hit);
                    t_hit *= resolution;

                    double zktStar = t_hit;
                    double pRand = (zkt >= 0 && zkt < maxRange) ? 1.0 / maxRange : 0.0;
                    double pMax = (zkt >= maxRange) ? 1.0 : 0.0;
                    double pShort = (zkt >= 0 && zkt <= zktStar)
                                        ? (1.0 / (1 - exp(-lambdaShort * zktStar))) *
                                            lambdaShort * exp(-lambdaShort * zkt)
                                        : 0.0;
                    double pHit = (zkt >= 0 && zkt <= maxRange)
                                    ? exp(-0.5 * (zkt - zktStar) * (zkt - zktStar) /
                                            (sigmaHit * sigmaHit)) /
                                            sqrt(TWO_PI * sigmaHit * sigmaHit)
                                    : 0.0;
                    probability =
                        zHit * pHit + zShort * pShort + zMax * pMax + zRand * pRand;

                    RT_DPRINTF("Results: dist = %5.3f, closest_hit = %5.3f, occData = %5.3f, probability = %5.3f\n", t_hit, closest_hit, currentNode.occData, probability);

                    // Save results
                    results.values[0] = 1; // Mark as valid
                    base = (uint32_t *)&probability;
                    results.values[2] = base[0];
                    results.values[3] = base[1];
                }
            }

            // Otherwise ignore

            break;
        }

        case 10: {
            // Don't actually need a mem fetch, but need TransactionType for processing
            transactions.push_back(MemoryTransactionRecord(
                leaf_addr,
                sizeof(double),
                TransactionType::BVH_QUAD_LEAF)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF)]++;
            double zHit = 10;
            double zShort = 0.01;
            double zMax = 0.1;
            double zRand = 10;
            double sigmaHit = 50.0;
            double lambdaShort = 0.1;
            double minProbability = 0.35;
            double maxRange = 1000.0;
            double sensorOffset = 25.0;

            double dist = *(double*)&conditions.values[8];
            double zkt = *(double*)&conditions.values[12];

            double zktStar = dist;
            double pRand = (zkt >= 0 && zkt < maxRange) ? 1.0 / maxRange : 0.0;
            double pMax = (zkt >= maxRange) ? 1.0 : 0.0;
            double pShort = (zkt >= 0 && zkt <= zktStar)
                                ? (1.0 / (1 - exp(-lambdaShort * zktStar))) *
                                    lambdaShort * exp(-lambdaShort * zkt)
                                : 0.0;
            double pHit = (zkt >= 0 && zkt <= maxRange)
                            ? exp(-0.5 * (zkt - zktStar) * (zkt - zktStar) /
                                    (sigmaHit * sigmaHit)) /
                                    sqrt(TWO_PI * sigmaHit * sigmaHit)
                            : 0.0;
            double probability =
                zHit * pHit + zShort * pShort + zMax * pMax + zRand * pRand;

            RT_DPRINTF("Results: dist = %5.3f, probability = %5.3f\n", dist, probability);

            // Save results
            results.values[0] = 1; // Mark as valid
            uint32_t *base = (uint32_t *)&probability;
            results.values[2] = base[0];
            results.values[3] = base[1];
            break;
        }

        case 12: {
            int X = conditions.values[0];
            int Y = conditions.values[1];
            int dX = conditions.values[2];
            int dY = conditions.values[3];
            int collision_index = conditions.values[14];

            int direction, i, j;
            calculate_direction_index(collision_index, direction, i, j);
            int nx, ny;
            calculate_next_node_xy(direction, nx, ny, X, Y, dX, dY);

            float sine = *(float*)&conditions.values[4 + 2*direction];
            float cosine = *(float*)&conditions.values[5 + 2*direction];

            RT_DPRINTF("Collision index %d ([%d][%d][%d]) -> next node (%d, %d) (%4.2f, %4.2f)\n", collision_index, direction, i, j, nx, ny, sine, cosine);

            int mapX = *(int*)&conditions.values[12];
            int mapY = *(int*)&conditions.values[13];
            int xbar = nx + round(i * cosine);
            int ybar = ny + round(j * sine);

            bool in_range = (0 <= xbar && xbar < mapX && 0 <= ybar && ybar < mapY);

            int thetaIdx = (conditions.values[15] >> (4 * direction)) & 0xF; // 4 bits per direction, 16 directions total
            int gidx = (xbar * mapY + ybar) * 8 + thetaIdx;

            uint8_t occ;
            mem->read(root_node_addr + gidx*sizeof(uint8_t), sizeof(uint8_t), &occ);
            transactions.push_back(MemoryTransactionRecord(
                root_node_addr + gidx*sizeof(uint8_t),
                sizeof(uint8_t), 
                TransactionType::BVH_INTERNAL_NODE)
            );
            GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;

            if (occ == 0 || !in_range) {
                if (!in_range) {
                    RT_DPRINTF("Out of bounds (%d, %d) - (%d, %d)\n", xbar, ybar, mapX, mapY);
                }
                else {
                    RT_DPRINTF("Collision detected\n");
                }
                conditions.values[15] |= (0xF << (4 * direction)); // Mark this direction as terminated
                RT_DPRINTF("Marking direction %d as terminated: 0x%x\n", direction, conditions.values[15]);
            }

            RT_DPRINTF("Bits: 0x%x\n", conditions.values[15]);
            results.values[0] = 1; // Mark as valid

            results.values[1] = !(conditions.values[15] & 0x1);
            results.values[2] = !(conditions.values[15] & 0x10);
            results.values[3] = !(conditions.values[15] & 0x100);
            results.values[4] = !(conditions.values[15] & 0x1000);
            RT_DPRINTF("Results: %d %d %d %d\n", results.values[1], results.values[2], results.values[3], results.values[4]);
            break;
        }

        default:
            printf("gpgpusim: ERROR! Unrecognized leaf type %d.\n", leaf_type);
            abort();
    }

    return results;
}

bool rt_accept_criteria(unsigned leaf_type, TreeSearchResults result, TreeSearchResults& existing_result, TreeSearchConditions& conditions) {
    switch (leaf_type) {
        case 0: {
            if (conditions.values[9]) {
                existing_result = result;
                // Return false to terminate traversal
                return false;
            }
            else if (!existing_result.values[0]) {
                // Update unconditionally if no existing results
                existing_result = result;
                return true;
            }
            else {
                // Only update if closer than existing closest hit
                float t_existing = *(float*)&existing_result.values[1];
                float t_new = *(float*)&result.values[1];
                if (t_new < t_existing) {
                    existing_result = result;
                }
                return true;
            }
            break;
        }
        case 1: {
            if (result.values[0] == 1) {
                existing_result = result;
                return false;
            }
            break;
        }
        case 3: 
        case 7: {
            // Always accept valid results
            if (result.values[0] == 1) {
                if (existing_result.values[0]) {
                    float current_accel_x = *(float*)&existing_result.values[1];
                    float current_accel_y = *(float*)&existing_result.values[2];
                    RT_DPRINTF("Update existing: current_accel_x=%7.3f, current_accel_y=%7.3f\n", current_accel_x, current_accel_y);

                    float new_accel_x = *(float*)&result.values[1];
                    float new_accel_y = *(float*)&result.values[2];

                    float accel_x = current_accel_x + new_accel_x;
                    float accel_y = current_accel_y + new_accel_y;
                    RT_DPRINTF("accel_x=%7.3f, accel_y=%7.3f\n", accel_x, accel_y);

                    existing_result.values[1] = *(uint32_t*)&accel_x;
                    existing_result.values[2] = *(uint32_t*)&accel_y;
                }
                else {
                    RT_DPRINTF("Create new results\n");
                    float new_accel_x = *(float*)&result.values[1];
                    float new_accel_y = *(float*)&result.values[2];
                    RT_DPRINTF("accel_x=%7.3f, accel_y=%7.3f\n", new_accel_x, new_accel_y);

                    existing_result = result;
                }
            }

            // No early termination
            return true;
            break;
        }
        case 4: {
            // Always accept valid results
            if (result.values[0] == 1) {
                if (existing_result.values[0]) {
                    float current_accel_x = *(float*)&existing_result.values[1];
                    float current_accel_y = *(float*)&existing_result.values[2];
                    float current_accel_z = *(float*)&existing_result.values[3];
                    RT_DPRINTF("Update existing: current_accel_x=%7.3f, current_accel_y=%7.3f, current_accel_z=%7.3f\n", current_accel_x, current_accel_y, current_accel_z);

                    float new_accel_x = *(float*)&result.values[1];
                    float new_accel_y = *(float*)&result.values[2];
                    float new_accel_z = *(float*)&result.values[3];

                    float accel_x = current_accel_x + new_accel_x;
                    float accel_y = current_accel_y + new_accel_y;
                    float accel_z = current_accel_z + new_accel_z;
                    RT_DPRINTF("accel_x=%7.3f, accel_y=%7.3f, accel_z=%7.3f\n", accel_x, accel_y, accel_y);

                    existing_result.values[1] = *(uint32_t*)&accel_x;
                    existing_result.values[2] = *(uint32_t*)&accel_y;
                    existing_result.values[3] = *(uint32_t*)&accel_z;
                }
                else {
                    RT_DPRINTF("Create new results\n");
                    float new_accel_x = *(float*)&result.values[1];
                    float new_accel_y = *(float*)&result.values[2];
                    float new_accel_z = *(float*)&result.values[3];
                    RT_DPRINTF("accel_x=%7.3f, accel_y=%7.3f, accel_z=%7.3f\n", new_accel_x, new_accel_y, new_accel_z);

                    existing_result = result;
                }
            }

            // No early termination
            return true;
            break;
        }

        // Delibot
        case 9: {
            // Always accept valid results
            if (result.values[0] == 1) {
                existing_result = result;

                // Update closest hit
                conditions.values[12] = result.values[4];
                conditions.values[13] = result.values[5];

                double closest_hit = *(double*)&conditions.values[12];

                RT_DPRINTF("Results updated! Probability = %5.3f, Closest hit = %5.3f\n", *(double*)&existing_result.values[2], closest_hit);

            }
            return true; // No early termination
            break;
        }

        case 10: {
            if (result.values[0] == 1) {
                RT_DPRINTF("Accepting valid results\n");
                existing_result = result;
                return false;
            }
            break;
        }
        case 12: {
            RT_DPRINTF("Accept results\n");
            existing_result.values[0] = result.values[1];
            existing_result.values[1] = result.values[2];
            existing_result.values[2] = result.values[3];
            existing_result.values[3] = result.values[4];
            assert(conditions.values[14] == 0); // Should be 0 after processing
            return false; // Done
            break;
        }

        default:
            printf("gpgpusim: ERROR! Unrecognized leaf type %d.\n", leaf_type);
            abort();
    }
}

void set_op_sequence(unsigned node_processing_configuration, unsigned leaf_processing_configuration, ptx_thread_info *thread)
{
    int func_type = GPGPU_Context()->func_sim->g_rt_func_type;
    // printf("Setting op sequence for %s\n", func_type ? "individual RTAx units" : "fixed-function RTA");
    bool allow_return = GPGPU_Context()->func_sim->g_rt_allow_return;
    bool predicate_return = GPGPU_Context()->func_sim->g_rt_predicate_return;
    int optimize = GPGPU_Context()->func_sim->g_rt_optimize;
    bool run_parallel = GPGPU_Context()->func_sim->g_rt_run_parallel;
    bool sphere_only = GPGPU_Context()->func_sim->g_rt_sphere_only;

    std::map<int, std::queue<RTFuncInsnType> > &rt_op_sequences = GPGPU_Context()->func_sim->g_rt_op_seq;

    if (rt_op_sequences.size() == 0) {

        printf("DEBUGGING MESSAGE: Setting TTA+ op sequences with node_processing_configuration %d\n", node_processing_configuration);

        // RTAx
        if (func_type == 1) {
            switch (node_processing_configuration) {
                // Normal ray tracing
                case 0: {
                    std::queue<RTFuncInsnType> ray_box_ops;
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_RCP);
                    ray_box_ops.push(RTFuncInsnType::RT_RCP);
                    ray_box_ops.push(RTFuncInsnType::RT_RCP);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                    ray_box_ops.push(RTFuncInsnType::RT_MINMAX);
                    ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                    ray_box_ops.push(RTFuncInsnType::RT_MINMAX);
                    ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                    ray_box_ops.push(RTFuncInsnType::RT_MINMAX);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_OR);

                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = ray_box_ops;

                    std::queue<RTFuncInsnType> ray_tri_ops;
                    ray_tri_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_tri_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_tri_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_tri_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_tri_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_tri_ops.push(RTFuncInsnType::RT_DOT);
                    ray_tri_ops.push(RTFuncInsnType::RT_DOT);
                    ray_tri_ops.push(RTFuncInsnType::RT_DOT);
                    ray_tri_ops.push(RTFuncInsnType::RT_RCP);
                    ray_tri_ops.push(RTFuncInsnType::RT_MUL);
                    ray_tri_ops.push(RTFuncInsnType::RT_MUL);
                    ray_tri_ops.push(RTFuncInsnType::RT_DOT);
                    ray_tri_ops.push(RTFuncInsnType::RT_VEC_CMP);
                    ray_tri_ops.push(RTFuncInsnType::RT_VEC_CMP);
                    ray_tri_ops.push(RTFuncInsnType::RT_MUL);
                    ray_tri_ops.push(RTFuncInsnType::RT_VEC_OR);
                    ray_tri_ops.push(RTFuncInsnType::RT_VEC_OR);

                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF] = ray_tri_ops;
                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF_HIT] = ray_tri_ops;

                    std::queue<RTFuncInsnType> ray_xform_ops;
                    ray_xform_ops.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP);
                    rt_op_sequences[(int)TransactionType::BVH_INSTANCE_LEAF] = ray_xform_ops;


                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);
                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PROCEDURAL_LEAF] = ray_decode_ops;
                    break;
                }

                // B tree
                case 1:
                case 2: {
                    std::queue<RTFuncInsnType> btree_op;
                    btree_op.push(RTFuncInsnType::RT_MINMAX);
                    btree_op.push(RTFuncInsnType::RT_MAXMIN);
                    btree_op.push(RTFuncInsnType::RT_VEC_CMP);
                    btree_op.push(RTFuncInsnType::RT_VEC_OR);
                    btree_op.push(RTFuncInsnType::RT_MINMAX);
                    btree_op.push(RTFuncInsnType::RT_MAXMIN);
                    btree_op.push(RTFuncInsnType::RT_VEC_CMP);
                    btree_op.push(RTFuncInsnType::RT_VEC_OR);
                    btree_op.push(RTFuncInsnType::RT_MINMAX);
                    btree_op.push(RTFuncInsnType::RT_MAXMIN);
                    btree_op.push(RTFuncInsnType::RT_VEC_CMP);
                    btree_op.push(RTFuncInsnType::RT_VEC_OR);

                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = btree_op;
                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF] = btree_op;

                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);

                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    break;
                }

                // N body 2D
                case 3: 
                case 7: {
                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);

                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR] = ray_decode_ops;

                    // DIST calculation
                    std::queue<RTFuncInsnType> nbody_box_ops;
                    nbody_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    nbody_box_ops.push(RTFuncInsnType::RT_DOT);
                    nbody_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = nbody_box_ops;
                    rt_op_sequences[(int)TransactionType::BVH_INSTANCE_LEAF] = nbody_box_ops;


                    // FORCE calculation
                    std::queue<RTFuncInsnType> nbody_ops;
                    // dist = sqrt(dist)
                    nbody_ops.push(RTFuncInsnType::RT_SQRT);
                    // dist = dist * dist
                    nbody_ops.push(RTFuncInsnType::RT_MUL);
                    nbody_ops.push(RTFuncInsnType::RT_MUL);
                    // accel_f = mass * dist
                    nbody_ops.push(RTFuncInsnType::RT_MUL);
                    // accel_x = accel_f * dx, accel_y = accel_f * dy, accel_z = accel_f * dz
                    nbody_ops.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP);

                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF] = nbody_ops;
                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF_HIT] = nbody_ops;
                    break;
                }
                // N body 3D
                case 4: {
                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);

                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR] = ray_decode_ops;
                    // Internal node is used to fetch children nodes; computation for pruning is performed per "instance leaf" (i.e. child of the node)

                    std::queue<RTFuncInsnType> nbody_box_ops;
                    // <dx, dy, dz> = <x1, y1, z1> - <x0, y0, z0>
                    nbody_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    // 	float dist = dx*dx + dy*dy + dz*dz + eps2;
                    nbody_box_ops.push(RTFuncInsnType::RT_DOT);
                    // check if dist < node_depth
                    nbody_box_ops.push(RTFuncInsnType::RT_VEC_CMP);
                    
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = nbody_box_ops;
                    rt_op_sequences[(int)TransactionType::BVH_INSTANCE_LEAF] = nbody_box_ops;

                    std::queue<RTFuncInsnType> nbody_ops;
                    // dist = sqrt(dist)
                    nbody_ops.push(RTFuncInsnType::RT_SQRT);
                    // dist = dist * dist
                    nbody_ops.push(RTFuncInsnType::RT_MUL);
                    nbody_ops.push(RTFuncInsnType::RT_MUL);
                    // accel_f = mass * dist
                    nbody_ops.push(RTFuncInsnType::RT_MUL);
                    // accel_x = accel_f * dx, accel_y = accel_f * dy, accel_z = accel_f * dz
                    nbody_ops.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP);

                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF] = nbody_ops;
                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF_HIT] = nbody_ops;
                    break;
                }


                // RTNN
                case 5: {
                    std::queue<RTFuncInsnType> rtnn_ops;
                    rtnn_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    rtnn_ops.push(RTFuncInsnType::RT_DOT);
                    rtnn_ops.push(RTFuncInsnType::RT_MUL);
                    rtnn_ops.push(RTFuncInsnType::RT_VEC_CMP);
                    rtnn_ops.push(RTFuncInsnType::RT_VEC_OR);

                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF] = rtnn_ops;
                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF_HIT] = rtnn_ops;

                    std::queue<RTFuncInsnType> ray_xform_ops;
                    ray_xform_ops.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP);

                    rt_op_sequences[(int)TransactionType::BVH_INSTANCE_LEAF] = ray_xform_ops;
                    
                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);

                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PROCEDURAL_LEAF] = ray_decode_ops;

                    std::queue<RTFuncInsnType> ray_box_ops;
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_RCP);
                    ray_box_ops.push(RTFuncInsnType::RT_RCP);
                    ray_box_ops.push(RTFuncInsnType::RT_RCP);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                    ray_box_ops.push(RTFuncInsnType::RT_MINMAX);
                    ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                    ray_box_ops.push(RTFuncInsnType::RT_MINMAX);
                    ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                    ray_box_ops.push(RTFuncInsnType::RT_MINMAX);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_OR);

                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = ray_box_ops;
                    break;
                }


                // Ray sphere
                case 6: {
                    std::queue<RTFuncInsnType> ray_sphere_ops;
                    ray_sphere_ops.push(RTFuncInsnType::RT_DOT);
                    ray_sphere_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_sphere_ops.push(RTFuncInsnType::RT_DOT);
                    ray_sphere_ops.push(RTFuncInsnType::RT_DOT);
                    ray_sphere_ops.push(RTFuncInsnType::RT_MUL);
                    ray_sphere_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_sphere_ops.push(RTFuncInsnType::RT_MUL);
                    ray_sphere_ops.push(RTFuncInsnType::RT_MUL);
                    ray_sphere_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_sphere_ops.push(RTFuncInsnType::RT_SQRT);
                    ray_sphere_ops.push(RTFuncInsnType::RT_RCP);
                    ray_sphere_ops.push(RTFuncInsnType::RT_VEC_CMP);
                    ray_sphere_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_sphere_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_sphere_ops.push(RTFuncInsnType::RT_MUL);
                    ray_sphere_ops.push(RTFuncInsnType::RT_MUL);
                    ray_sphere_ops.push(RTFuncInsnType::RT_VEC_CMP);
                    ray_sphere_ops.push(RTFuncInsnType::RT_VEC_OR);

                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF] = ray_sphere_ops;
                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF_HIT] = ray_sphere_ops;

                    std::queue<RTFuncInsnType> ray_xform_ops;
                    ray_xform_ops.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP);

                    rt_op_sequences[(int)TransactionType::BVH_INSTANCE_LEAF] = ray_xform_ops;

                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);

                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PROCEDURAL_LEAF] = ray_decode_ops;

                    std::queue<RTFuncInsnType> ray_box_ops;
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_RCP);
                    ray_box_ops.push(RTFuncInsnType::RT_RCP);
                    ray_box_ops.push(RTFuncInsnType::RT_RCP);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MUL);
                    ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                    ray_box_ops.push(RTFuncInsnType::RT_MINMAX);
                    ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                    ray_box_ops.push(RTFuncInsnType::RT_MINMAX);
                    ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                    ray_box_ops.push(RTFuncInsnType::RT_MINMAX);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_OR);

                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = ray_box_ops;
                    break;
                }

                case 8: {
                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);
                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    
                    std::queue<RTFuncInsnType> ray_box_ops;

                    // Bounding and inscribing sphere tests
                    if (optimize && allow_return) {

                        /* UPDATED VERSION FOR SPHERE-ONLY TEST (MICRO REBUTTAL)*/
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                        ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                        ray_box_ops.push(RTFuncInsnType::RT_MAXMIN);
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);
                        
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);

                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_OUTSPHERE] = ray_box_ops;

                        if (sphere_only) {
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET1] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET2] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET3] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET4] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET5] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET6] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET7] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET8] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET9] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET10] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET11] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET12] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET13] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET14] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_OUTSPHERE] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_INSPHERE] = ray_box_ops;
                            break;
                        }

                        // create min sphere
                        ray_box_ops.push(RTFuncInsnType::RT_MINMAX); // MIN only
                        ray_box_ops.push(RTFuncInsnType::RT_MINMAX); // MIN only
                        ray_box_ops.push(RTFuncInsnType::RT_MUL);

                        // circle intersection code
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB); 

                        // for loop 3 times
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP); 
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB); 
                        ray_box_ops.push(RTFuncInsnType::RT_MUL); 
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB); 

                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP); 
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB); 
                        ray_box_ops.push(RTFuncInsnType::RT_MUL); 
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB); 

                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP); 
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB); 
                        ray_box_ops.push(RTFuncInsnType::RT_MUL); 
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB); 

                        // if intersect with min circle, return 1
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP); 
                        // predicated return op
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_INSPHERE] = ray_box_ops;
                    }


                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP);

                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);

                    // // return 1
                    ray_box_ops.push(RTFuncInsnType::RT_DOT);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET1] = ray_box_ops;
                    }


                    // // return 2
                    ray_box_ops.push(RTFuncInsnType::RT_DOT);
                    ray_box_ops.push(RTFuncInsnType::RT_DOT);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET2] = ray_box_ops;
                    }


                    // // return 3
                    ray_box_ops.push(RTFuncInsnType::RT_DOT);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET3] = ray_box_ops;
                    }


                    // // return 4
                    ray_box_ops.push(RTFuncInsnType::RT_DOT);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET4] = ray_box_ops;
                    }


                    // // return 5
                    ray_box_ops.push(RTFuncInsnType::RT_DOT);
                    ray_box_ops.push(RTFuncInsnType::RT_DOT);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET5] = ray_box_ops;
                    }


                    // // return 6
                    ray_box_ops.push(RTFuncInsnType::RT_DOT);
                    ray_box_ops.push(RTFuncInsnType::RT_DOT);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET6] = ray_box_ops;
                    }


                    // // return 7
                    ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET7] = ray_box_ops;
                    }


                    // // return 8
                    ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET8] = ray_box_ops;
                    }


                    // // return 9
                    ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET9] = ray_box_ops;
                    }


                    // // return 10
                    ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET10] = ray_box_ops;
                    }


                    // // return 11
                    ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET11] = ray_box_ops;
                    }


                    // // return 12
                    ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET12] = ray_box_ops;
                    }

                    
                    // // return 13
                    ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET13] = ray_box_ops;
                    }


                    // // return 14
                    ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    if (allow_return) {
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET14] = ray_box_ops;
                    }


                    // // return 15
                    ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                    ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = ray_box_ops;

                    if (!allow_return) {
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET1] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET2] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET3] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET4] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET5] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET6] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET7] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET8] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET9] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET10] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET11] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET12] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET13] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET14] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_OUTSPHERE] = ray_box_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_INSPHERE] = ray_box_ops;
                    }
                    else if (predicate_return) {
                        // ray_box_ops has the full set of ops -> copy into std::deque
                        std::queue<RTFuncInsnType> ray_box_ops_copy = ray_box_ops;
                        std::deque<RTFuncInsnType> full_op_sequence;
                        while (!ray_box_ops_copy.empty()) {
                            full_op_sequence.push_back(ray_box_ops_copy.front());
                            ray_box_ops_copy.pop();
                        }
                        for (int ret_type = (int)TransactionType::BVH_INTERNAL_NODE_RET1; ret_type <= (int)TransactionType::BVH_INTERNAL_NODE_RET14; ret_type++) {
                            // Find the position of the RT_RETURN instruction based on length
                            unsigned ret_idx = rt_op_sequences[ret_type].size() - 1;

                            // Start from ray_box_ops[ret_idx] and loop to the end to copy the remaining ops
                            for (unsigned i = ret_idx; i < ray_box_ops.size(); i++) {
                                rt_op_sequences[ret_type].push(full_op_sequence[i]);
                            }
                        }
                    }
                    break;
                }

                case 9: {
                    // TODO (THIS IS A PLACEHOLDER)
                    std::queue<RTFuncInsnType> raybox2d;
                    raybox2d.push(RTFuncInsnType::RT_MINMAX); 

                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = raybox2d;
                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF] = raybox2d;

                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);

                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    break;
                }

                // DeliBot
                case 10: {
                    std::queue<RTFuncInsnType> bound_check;
                    // Step forward dist += resolution, ray_orig_x += ray_dir_x, ray_orig_y += ray_dir_y
                    bound_check.push(RTFuncInsnType::RT_VEC_SUB);
                    // Calculate index xIdx = static_cast<int>(xRay / resolution), gIdx = xIdx * mapSizeY + yIdx
                    bound_check.push(RTFuncInsnType::RT_RCP); 
                    bound_check.push(RTFuncInsnType::RT_DOT); 
                    // CMP (dist >= maxRange || xIdx >= mapSizeX || yIdx >= mapSizeY || xIdx < 0 || yIdx < 0)
                    bound_check.push(RTFuncInsnType::RT_VEC_CMP); 
                    bound_check.push(RTFuncInsnType::RT_VEC_OR); 
                    bound_check.push(RTFuncInsnType::RT_VEC_CMP); 
                    bound_check.push(RTFuncInsnType::RT_VEC_OR); 

                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = bound_check;

                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);

                    std::queue<RTFuncInsnType> prob_calc;
                    // pRand and pMax
                    prob_calc.push(RTFuncInsnType::RT_VEC_CMP); 
                    prob_calc.push(RTFuncInsnType::RT_RCP); 
                    // exp(-lambdaShort * zktStar)
                    prob_calc.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP); 
                    prob_calc.push(RTFuncInsnType::RT_OP_CLUSTER_B); // Choose B ~20 cycles
                    prob_calc.push(RTFuncInsnType::RT_RCP); 
                    prob_calc.push(RTFuncInsnType::RT_VEC_SUB);
                    prob_calc.push(RTFuncInsnType::RT_MUL); 
                    prob_calc.push(RTFuncInsnType::RT_MUL); 
                    // exp(-lambdaShort * zkt)
                    prob_calc.push(RTFuncInsnType::RT_VEC_SUB);
                    prob_calc.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP); 
                    prob_calc.push(RTFuncInsnType::RT_OP_CLUSTER_B); // Choose B ~20 cycles
                    prob_calc.push(RTFuncInsnType::RT_MUL); 
                    // probability = zHit * pHit + zShort * pShort + zMax * pMax + zRand * pRand;
                    prob_calc.push(RTFuncInsnType::RT_DOT); 
                    prob_calc.push(RTFuncInsnType::RT_DOT); 

                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF] = prob_calc;

                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    break;
                }

                default: {
                    printf("gpgpusim: ERROR! Unrecognized node type %d.\n", node_processing_configuration);
                    abort();
                }
            }
        }

        // RTA fixed-function
        else if (func_type == 0){
            switch (node_processing_configuration) {
                case 0: // Regular ray tracing
                case 1: // RTA+
                case 2: // RTA+
                case 3: // RTA+
                case 4: // RTA+
                case 5: // RTA+ (RTNN)
                case 7: // RTA+
                case 9: // RTA+
                {
                    std::queue<RTFuncInsnType> ray_box_ops;
                    ray_box_ops.push(RTFuncInsnType::RT_RAY_BOX_FUNC_OP);

                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = ray_box_ops;

                    std::queue<RTFuncInsnType> ray_tri_ops;
                    ray_tri_ops.push(RTFuncInsnType::RT_RAY_TRI_FUNC_OP);

                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF] = ray_tri_ops;
                    rt_op_sequences[(int)TransactionType::BVH_QUAD_LEAF_HIT] = ray_tri_ops;

                    std::queue<RTFuncInsnType> ray_xform_ops;
                    ray_xform_ops.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP);

                    rt_op_sequences[(int)TransactionType::BVH_INSTANCE_LEAF] = ray_xform_ops;

                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);

                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR] = ray_decode_ops;
                    rt_op_sequences[(int)TransactionType::BVH_PROCEDURAL_LEAF] = ray_decode_ops;

                    break;
                }
                case 8: // Staged Collision Core
                {                    
                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);

                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    std::queue<RTFuncInsnType> collision_ops;

                    // Pre-Processing
                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_C);

                    if (run_parallel) {
                        // Assume RT_OP_CLUSTER_C represents all 15 axis running in parallel. Latency will be max of all axes
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET1] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET2] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET3] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET4] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET5] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET6] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET7] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET8] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET9] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET10] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET11] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET12] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET13] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET14] = collision_ops;
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = collision_ops;

                        break;
                    }


                    // RET1-14 op types will only exist if allow_return is true (as set by track_collision_transaction)
                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET1] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET2] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET3] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET4] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET5] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET6] = collision_ops;

                    // Axis 7+ uses cluster B
                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET7] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET8] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET9] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET10] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET11] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET12] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET13] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET14] = collision_ops;

                    collision_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                    rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = collision_ops;

                    if (predicate_return) {
                        for (int ret_type = (int)TransactionType::BVH_INTERNAL_NODE_RET1; ret_type <= (int)TransactionType::BVH_INTERNAL_NODE_RET14; ret_type++) {
                            rt_op_sequences[ret_type].push(RTFuncInsnType::RT_RETURN);
                            // Push the remaining cluster ops
                            for (int clusterA=0; clusterA<((int)TransactionType::BVH_INTERNAL_NODE_RET6 - ret_type); clusterA++) {
                                rt_op_sequences[ret_type].push(RTFuncInsnType::RT_OP_CLUSTER_A);
                            }
                            for (int clusterB=0; clusterB<((int)TransactionType::BVH_INTERNAL_NODE_RET14 + 1 - std::max(ret_type, (int)TransactionType::BVH_INTERNAL_NODE_RET6)); clusterB++) {
                                rt_op_sequences[ret_type].push(RTFuncInsnType::RT_OP_CLUSTER_B);
                            }
                        }
                    }
                    else {
                        for (int ret_type = (int)TransactionType::BVH_INTERNAL_NODE_RET1; ret_type <= (int)TransactionType::BVH_INTERNAL_NODE_RET14; ret_type++) {
                            rt_op_sequences[ret_type].push(RTFuncInsnType::RT_RETURN);
                        }
                    }

                    break;
                }

                default: {
                    printf("gpgpusim: ERROR! Unrecognized node type %d.\n", node_processing_configuration);
                    abort();
                }
            }
        }

        // Clustered version
        else if (func_type == 2) {
            switch (node_processing_configuration) {

                case 8: {
                    std::queue<RTFuncInsnType> ray_box_ops;

                    if (optimize && allow_return) {
                        // create max circle
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);

                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_C); 

                        // if no intersection with max circle, return 0
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_OUTSPHERE] = ray_box_ops;
                        
                        // create min sphere
                        ray_box_ops.push(RTFuncInsnType::RT_MINMAX); // MIN only
                        ray_box_ops.push(RTFuncInsnType::RT_MINMAX); // MIN only
                        ray_box_ops.push(RTFuncInsnType::RT_MUL);

                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_C); 

                        // if intersect with min circle, return 1
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);
                        ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_INSPHERE] = ray_box_ops;


                        // obbOverlap
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP);

                        // // obbDisjoint
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);

                        // // return 1
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET1] = ray_box_ops;
                        }


                        // // return 2
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET2] = ray_box_ops;
                        }


                        // // return 3
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET3] = ray_box_ops;
                        }


                        // // return 4
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET4] = ray_box_ops;
                        }


                        // // return 5
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET5] = ray_box_ops;
                        }


                        // // return 6
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET6] = ray_box_ops;
                        }


                        // // return 7
                        ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET7] = ray_box_ops;
                        }


                        // // return 8
                        ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET8] = ray_box_ops;
                        }


                        // // return 9
                        ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET9] = ray_box_ops;
                        }


                        // // return 10
                        ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET10] = ray_box_ops;
                        }


                        // // return 11
                        ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET11] = ray_box_ops;
                        }


                        // // return 12
                        ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET12] = ray_box_ops;
                        }

                        
                        // // return 13
                        ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET13] = ray_box_ops;
                        }


                        // // return 14
                        ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET14] = ray_box_ops;
                        }


                        // // return 15
                        ray_box_ops.push(RTFuncInsnType::RT_CROSS);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_CMP);

                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = ray_box_ops;

                    }

                    else { 

                        // obbOverlap
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_RAY_XFORM_FUNC_OP);

                        // // obbDisjoint
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);
                        ray_box_ops.push(RTFuncInsnType::RT_VEC_SUB);

                        // // return 1
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET1] = ray_box_ops;
                        }

                        // // return 2
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET2] = ray_box_ops;
                        }

                        // // return 3
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET3] = ray_box_ops;
                        }

                        // // return 4
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET4] = ray_box_ops;
                        }

                        // // return 5
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET5] = ray_box_ops;
                        }

                        // // return 6
                        ray_box_ops.push(RTFuncInsnType::RT_DOT);
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_A);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET6] = ray_box_ops;
                        }

                        // // return 7
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET7] = ray_box_ops;
                        }

                        // // return 8
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET8] = ray_box_ops;
                        }

                        // // return 9
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET9] = ray_box_ops;
                        }

                        // // return 10
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET10] = ray_box_ops;
                        }

                        // // return 11
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET11] = ray_box_ops;
                        }

                        // // return 12
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET12] = ray_box_ops;
                        }
                        
                        // // return 13
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET13] = ray_box_ops;
                        }

                        // // return 14
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);
                        if (allow_return) {
                            ray_box_ops.push(RTFuncInsnType::RT_RETURN);
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET14] = ray_box_ops;
                        }

                        // // return 15
                        ray_box_ops.push(RTFuncInsnType::RT_OP_CLUSTER_B);

                        rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE] = ray_box_ops;

                        if (!allow_return) {
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET1] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET2] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET3] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET4] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET5] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET6] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET7] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET8] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET9] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET10] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET11] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET12] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET13] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_RET14] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_OUTSPHERE] = ray_box_ops;
                            rt_op_sequences[(int)TransactionType::BVH_INTERNAL_NODE_INSPHERE] = ray_box_ops;
                        }
                    }

                    std::queue<RTFuncInsnType> ray_decode_ops;
                    ray_decode_ops.push(RTFuncInsnType::RT_DECODE);
                    rt_op_sequences[(int)TransactionType::BVH_STRUCTURE] = ray_decode_ops;
                    break;
                }

                default: {
                    printf("gpgpusim: ERROR! Unrecognized node type %d.\n", node_processing_configuration);
                    abort();
                }
            }
        }
        else {
            printf("gpgpusim: ERROR! Unrecognized function type %d.\n", func_type);
        }
    }

    return;
}

// General purpose tree traversal
void rt_traverse_tree(const ptx_instruction *pI, ptx_thread_info *thread)
{
    TreeSearchConditions conditions;
    addr_t TreeRootAddr;
    unsigned node_processing_configuration;
    unsigned leaf_processing_configuration;
    TreeSearchResults* results_payload_addr;
    
    read_traverse_tree_args(pI, thread, conditions, TreeRootAddr, node_processing_configuration, leaf_processing_configuration, &results_payload_addr);

    RT_DPRINTF("[0x%x]: result buffer 0x%x\n", thread->get_uid(), results_payload_addr);
    RT_DPRINTF("Node Configuration: %d, Leaf Configuration: %d\n", node_processing_configuration, leaf_processing_configuration);

    std::vector<MemoryTransactionRecord> transactions;
    std::list<StackEntry> stack;
    TreeSearchResults results_payload;
    results_payload.values[0] = 0;

    memory_space *mem = NULL;
    mem = thread->get_global_memory();
    gpgpu_context *ctx = GPGPU_Context();

    // Track each traversal as a closest hit ray
    ctx->func_sim->g_n_closesthit_rays++;

    unsigned total_nodes_accessed = 0;

    // Initialize traversal stack
    StackEntry current_node = StackEntry(TreeRootAddr, true, false);
    stack.push_back(current_node);

    if (node_processing_configuration == 0) {
        RT_DPRINTF("[0x%x]: tree_traversal initialized with ray ", thread->get_uid());
        RT_DPRINTF("orig %7.3f %7.3f %7.3f : dir %7.3f %7.3f %7.3f t [%5.3f %5.3f]\n", 
            *(float*)&conditions.values[0],
            *(float*)&conditions.values[1],
            *(float*)&conditions.values[2],
            *(float*)&conditions.values[4],
            *(float*)&conditions.values[5],
            *(float*)&conditions.values[6],
            *(float*)&conditions.values[3],
            *(float*)&conditions.values[7]
        );
    }
    else if (node_processing_configuration == 1) {
        RT_DPRINTF("[0x%x]: tree_traversal initialized with key %d \n", 
            thread->get_uid(),
            *(int *)&conditions.values[0]
        );
    }
    else if (node_processing_configuration == 3) {
        RT_DPRINTF("[0x%x]: tree_traversal initialized with ", thread->get_uid());
        RT_DPRINTF("position (%7.3f %7.3f), starting depth (%7.3f), x: 0x%x, y: 0x%x, mass: 0x%x, n: %d\n", 
            *(float*)&conditions.values[0],
            *(float*)&conditions.values[1],
            *(float*)&conditions.values[5],
            conditions.values[2],
            conditions.values[3],
            conditions.values[4],
            conditions.values[6]
        );
    }
    else if (node_processing_configuration == 8) {
        RT_DPRINTF("[0x%x]: tree_traversal initialized with ", thread->get_uid());

        // Put conditions in an OBB struct
        geometry::OBB<float> obb;
        obb.u[0][0] = *(float*)&(conditions.values[0]);
        obb.u[0][1] = *(float*)&(conditions.values[1]);
        obb.u[0][2] = *(float*)&(conditions.values[2]);
        obb.u[1][0] = *(float*)&(conditions.values[3]);
        obb.u[1][1] = *(float*)&(conditions.values[4]);
        obb.u[1][2] = *(float*)&(conditions.values[5]);
        obb.u[2][0] = *(float*)&(conditions.values[6]);
        obb.u[2][1] = *(float*)&(conditions.values[7]);
        obb.u[2][2] = *(float*)&(conditions.values[8]);
        obb.c[0] = *(float*)&(conditions.values[9]);
        obb.c[1] = *(float*)&(conditions.values[10]);
        obb.c[2] = *(float*)&(conditions.values[11]);
        obb.e[0] = *(float*)&(conditions.values[12]);
        obb.e[1] = *(float*)&(conditions.values[13]);
        obb.e[2] = *(float*)&(conditions.values[14]);

        RT_DPRINTF("OBB: u0 (%5.3f %5.3f %5.3f), u1 (%5.3f %5.3f %5.3f), u2 (%5.3f %5.3f %5.3f), c (%5.3f %5.3f %5.3f), e (%5.3f %5.3f %5.3f)\n",
            obb.u[0][0], obb.u[0][1], obb.u[0][2],
            obb.u[1][0], obb.u[1][1], obb.u[1][2],
            obb.u[2][0], obb.u[2][1], obb.u[2][2],
            obb.c[0], obb.c[1], obb.c[2],
            obb.e[0], obb.e[1], obb.e[2]
        );
    }
    else if (node_processing_configuration == 9) {
        RT_DPRINTF("[0x%x]: tree_traversal initialized with ", thread->get_uid());

        double ray_orig_x, ray_orig_y, ray_dir_x, ray_dir_y;
        double zkt, resolution;
        double closest_hit;

        ray_orig_x = *(double*)&conditions.values[0];
        ray_orig_y = *(double*)&conditions.values[2];
        ray_dir_x = *(double*)&conditions.values[4];
        ray_dir_y = *(double*)&conditions.values[6];
        zkt = *(double*)&conditions.values[8];
        resolution = *(double*)&conditions.values[10];
        closest_hit = *(double*)&conditions.values[12];

        RT_DPRINTF("ray_orig (%5.3f %5.3f), ray_dir (%5.3f %5.3f), zkt (%5.3f), resolution (%5.3f), closesthit (%5.3f)\n",
            ray_orig_x, ray_orig_y,
            ray_dir_x, ray_dir_y,
            zkt, resolution, closest_hit
        );
    }
    else if (node_processing_configuration == 10) {
        RT_DPRINTF("[0x%x]: tree_traversal initialized with ", thread->get_uid());

        double ray_orig_x, ray_orig_y, ray_dir_x, ray_dir_y;
        double dist, resolution, zkt;
        int mapX, mapY;

        ray_orig_x = *(double*)&conditions.values[0];
        ray_orig_y = *(double*)&conditions.values[2];
        ray_dir_x = *(double*)&conditions.values[4];
        ray_dir_y = *(double*)&conditions.values[6];
        dist = *(double*)&conditions.values[8];
        resolution = *(double*)&conditions.values[10];
        zkt = *(double*)&conditions.values[12];
        mapX = *(int*)&conditions.values[14];
        mapY = *(int*)&conditions.values[15];

        RT_DPRINTF("ray_orig (%5.3f %5.3f), ray_dir (%5.3f %5.3f), dist (%5.3f), resolution (%5.3f), zkt (%5.3f) in map [%d, %d]\n",
            ray_orig_x, ray_orig_y,
            ray_dir_x, ray_dir_y,
            zkt, resolution, mapX, mapY
        );
    }

    else if (node_processing_configuration == 12) {
        RT_DPRINTF("[0x%x]: tree_traversal initialized with ", thread->get_uid());
        int X = *(int*)&conditions.values[0];
        int Y = *(int*)&conditions.values[1];
        int dX = *(int*)&conditions.values[2];
        int dY = *(int*)&conditions.values[3];

        RT_DPRINTF("X: %d, Y: %d, dX: %d, dY: %d ->", X, Y, dX, dY);

        // const int NX[4] = { x, x, x + dX, x - dX };
        // const int NY[4] = { y, y, y + dY, y - dY };
        RT_DPRINTF(" (%d, %d), (%d, %d), (%d, %d), (%d, %d) ; ", 
            X, Y,
            X, Y,
            X + dX, Y + dY,
            X - dX, Y - dY
        );

        RT_DPRINTF("(%4.2f, %4.2f), (%4.2f, %4.2f), (%4.2f, %4.2f), (%4.2f, %4.2f)\n", 
            *(float*)&conditions.values[4], *(float*)&conditions.values[5],
            *(float*)&conditions.values[6], *(float*)&conditions.values[7],
            *(float*)&conditions.values[8], *(float*)&conditions.values[9],
            *(float*)&conditions.values[10], *(float*)&conditions.values[11]
        );
        RT_DPRINTF("d_mapX: %d, d_mapY: %d, ", *(int*)&conditions.values[12], *(int*)&conditions.values[13]);
        RT_DPRINTF("index: %d, ", conditions.values[14]);

        uint32_t thetaIdx = conditions.values[15];
        for (unsigned i = 0; i < 4; i++) {
            RT_DPRINTF("thetaIdx[%d]: %d ", i, (thetaIdx >> (i * 4)) & 0xF);
        }
        RT_DPRINTF("\n");
    }


    bool use_dfs;
    switch (node_processing_configuration) {
        case 0:
        case 1:
        case 2: {
            use_dfs = true; // B tree doesn't matter
            break;
        }
        case 3: 
        case 7: {
            use_dfs = true;
            break;
        }
        case 4: {
            use_dfs = true;
            break;
        }
        case 8: {
            use_dfs = false;
            break;
        }
        case 9: {
            use_dfs = true;
            break;
        }
        case 10: {
            use_dfs = true;
            break;
        }
        case 12: {
            use_dfs = true; // RTA+ doesn't matter
            break;
        }
        default: {
            printf("gpgpusim: ERROR! Unrecognized node type %d.\n", node_processing_configuration);
            abort();
        }
    }

    // While-While loop
    while (!stack.empty()) {
        // Pop next node
        current_node = stack.back();
        stack.pop_back();
        RT_DPRINTF("[0x%x]: pop next node 0x%x from stack, %d nodes remaining\n", thread->get_uid(), current_node.addr, stack.size());

        // Decode node
        bool isLeaf = rt_decode_node(node_processing_configuration, mem, current_node, TreeRootAddr, conditions);

        // Process inner node
        if (!isLeaf) {
            RT_DPRINTF("[0x%x]: traversing inner node 0x%x\n", thread->get_uid(), current_node.addr);
            bool terminate = false;

            std::list<StackEntry> next_nodes = rt_process_inner_node(node_processing_configuration, mem, current_node, TreeRootAddr, conditions, transactions, terminate);
            total_nodes_accessed++;

            // Push to stack
            if (next_nodes.size() > 0) {
                RT_DPRINTF("[0x%x]: inner node 0x%x hit, pushing %d nodes to stack, %d nodes total\n", thread->get_uid(), current_node.addr, next_nodes.size(), stack.size());

                // BFS / DFS
                if (use_dfs) {
                    stack.splice(stack.end(), next_nodes);
                }
                else {
                    stack.splice(stack.begin(), next_nodes);
                }
            }
            else {
                RT_DPRINTF("[0x%x]: inner node 0x%x miss\n", thread->get_uid(), current_node.addr);
            }

            // Early termination for collision detection
            if (node_processing_configuration == 8 && terminate) {
                RT_DPRINTF("[0x%x]: Collision detected, terminating traversal\n", thread->get_uid());
                results_payload.values[0] = 1; // report collision
                break;
            }
        }

        // Process leaf node
        else {
            RT_DPRINTF("[0x%x]: traversing leaf node 0x%x\n", thread->get_uid(), current_node.addr);

            // Requires an intersection shader
            if (leaf_processing_configuration == 5) {
                // Track how many times the intersection shader is called
                if (results_payload.values[0] == 1) {
                    results_payload.values[1] += 1;
                }
                else {
                    results_payload.values[0] = 1;
                    results_payload.values[1] = 1;
                }
            }

            else {
                TreeSearchResults results = rt_process_leaf_node(leaf_processing_configuration, mem, current_node.addr, TreeRootAddr, conditions, transactions);
                total_nodes_accessed++;

                if (results.values[0]) {
                    // RT_DPRINTF("[0x%x]: hit tri %d, bary [%5.3f, %5.3f, %5.3f], thit %5.3f\n", thread->get_uid(),
                    //     results.values[5], 
                    //     *(float*)&results.values[2], 
                    //     *(float*)&results.values[3], 
                    //     *(float*)&results.values[4], 
                    //     *(float*)&results.values[1]
                    // );

                    // Save results; early terminate
                    if (rt_accept_criteria(leaf_processing_configuration, results, results_payload, conditions)) {
                        RT_DPRINTF("[0x%x]: leaf node 0x%x hit, continuing traversal\n", thread->get_uid(), current_node.addr);
                    }
                    else {
                        RT_DPRINTF("[0x%x]: leaf node 0x%x hit, terminating traversal\n", thread->get_uid(), current_node.addr);
                        break;
                    }
                }
            }
        }
    }
    // Write result
    if (results_payload.values[0] == 1) {
        RT_DPRINTF("[0x%x]: writing hit to payload ", thread->get_uid());
        // RT_DPRINTF("(tri %d, [%5.3f, %5.3f, %5.3f], %5.3f)", 
        //     results_payload.values[5], 
        //     *(float*)&results_payload.values[2], 
        //     *(float*)&results_payload.values[3], 
        //     *(float*)&results_payload.values[4], 
        //     *(float*)&results_payload.values[1]
        // );

        // RT_DPRINTF("[0x%x]: Valid %d, Closest hit %5.2f, Probability %5.2f\n", 
        //     thread->get_uid(),
        //     *(uint32_t*)&results_payload.values[0], 
        //     *(double *)&results_payload.values[4],
        //     *(double *)&results_payload.values[2]
        // );

        RT_DPRINTF("\n");
        
        // Count number of hits
        ctx->func_sim->g_rt_num_hits++;

        transactions.push_back(MemoryTransactionRecord(
            results_payload_addr,
            32, // 32 byte result
            TransactionType::WRITE_TRAVERSAL_RESULT)
        );
        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::WRITE_TRAVERSAL_RESULT)]++;
    }

    else {
        RT_DPRINTF("[0x%x]: traversal done, miss\n", thread->get_uid());
    }

    // if (node_processing_configuration == 3) {
    //     printf("[0x%x]: (%7.3f, %7.3f)\n", thread->get_uid(), *(float*)&results_payload.values[1], *(float*)&results_payload.values[2]);
    // }

    // Special case for delibot to only write the probability (saves memory)
    if (node_processing_configuration == 10) {
        double probability = *(double *)&results_payload.values[2];
        RT_DPRINTF("[0x%x]: Result = %f\n", thread->get_uid(), probability);
        RT_DPRINTF("[0x%x]: Writing results to payload 0x%x\n", thread->get_uid(), results_payload_addr);
        memory_space *shared_mem = NULL;
        shared_mem = thread->m_shared_mem;

        // Write to both mem spaces
        mem->write(results_payload_addr, 8, &probability, NULL, NULL);

        // Convert to shared memory address
        new_addr_type sh_addr = generic_to_shared(thread->get_hw_sid(), results_payload_addr);
        shared_mem->write(sh_addr, 8, &probability, NULL, NULL);
    }

    else if (node_processing_configuration == 12) {
        // Write the result to the payload
        RT_DPRINTF("[0x%x]: Writing results (%d %d %d %d) to payload 0x%x\n", thread->get_uid(), results_payload.values[0], results_payload.values[1], results_payload.values[2], results_payload.values[3], results_payload_addr);
        mem->write(results_payload_addr, sizeof(TreeSearchResults), &results_payload, NULL, NULL);

        // Also write to shared memory
        memory_space *shared_mem = NULL;
        shared_mem = thread->m_shared_mem;
        new_addr_type sh_addr = generic_to_shared(thread->get_hw_sid(), results_payload_addr);
        shared_mem->write(sh_addr, sizeof(TreeSearchResults), &results_payload, NULL, NULL);
    }

    else {
        RT_DPRINTF("[0x%x]: Writing results to payload 0x%x\n", thread->get_uid(), results_payload_addr);
        mem->write(results_payload_addr, sizeof(TreeSearchResults), &results_payload, NULL, NULL);
    }

    RT_DPRINTF("[0x%x]: %d memory transactions recorded\n", thread->get_uid(), transactions.size());
    thread->set_rt_transactions(transactions);

    if (total_nodes_accessed > GPGPU_Context()->func_sim->g_max_nodes_per_ray) {
        GPGPU_Context()->func_sim->g_max_nodes_per_ray = total_nodes_accessed;
    }
    GPGPU_Context()->func_sim->g_tot_nodes_per_ray += total_nodes_accessed;

    set_op_sequence(node_processing_configuration, leaf_processing_configuration, thread);
}

bool isInRange(int x, int y) {
    int d_mapX = 1024; // Example value, replace with actual map width
    int d_mapY = 1024; // Example value, replace with actual map height
    return 0 <= x && x < d_mapX && 0 <= y && y < d_mapY;
}

int xythetaToID(int x, int y, int thetaIdx)
{
    int d_mapY = 1024; // Example value, replace with actual map height
    return (x * d_mapY + y) * 8 + thetaIdx;
}

void rt_tree_unit_magic_func(const ptx_instruction *pI, ptx_thread_info *thread) {
    // __TreeUnitMagicFunc(&isStateFree, nx, ny, sine, cosine, g_graph);

    // Read input arguments
    int arg = 0;

    // &isStateFree
    void* result_addr;
    const operand_info &param_op1 = pI->operand_lookup(arg + 1);    
    addr_t from_addr = param_op1.get_symbol()->get_address();
    unsigned size = sizeof(void *);
    thread->m_local_mem->read(from_addr, size, &result_addr);

    // nx
    int nx, ny;
    arg++;
    const operand_info &param_op2 = pI->operand_lookup(arg + 1);    
    from_addr = param_op2.get_symbol()->get_address();
    size = sizeof(int);
    thread->m_local_mem->read(from_addr, size, &nx);

    // ny
    arg++;
    const operand_info &param_op5 = pI->operand_lookup(arg + 1);    
    from_addr = param_op5.get_symbol()->get_address();
    size = sizeof(int);
    thread->m_local_mem->read(from_addr, size, &ny);

    // nthetaIdx
    int nthetaIdx;
    arg++;
    const operand_info &param_op7 = pI->operand_lookup(arg + 1);
    from_addr = param_op7.get_symbol()->get_address();
    size = sizeof(int);
    thread->m_local_mem->read(from_addr, size, &nthetaIdx);
    
    // sine
    double sine, cosine;
    arg++;
    const operand_info &param_op3 = pI->operand_lookup(arg + 1);    
    from_addr = param_op3.get_symbol()->get_address();
    size = sizeof(double);
    thread->m_local_mem->read(from_addr, size, &sine);

    // cosine
    arg++;
    const operand_info &param_op4 = pI->operand_lookup(arg + 1);    
    from_addr = param_op4.get_symbol()->get_address();
    size = sizeof(double);
    thread->m_local_mem->read(from_addr, size, &cosine);

    // g_graph
    addr_t g_graph_addr;
    arg++;
    const operand_info &param_op6 = pI->operand_lookup(arg + 1);    
    from_addr = param_op6.get_symbol()->get_address();
    size = sizeof(addr_t);
    thread->m_local_mem->read(from_addr, size, &g_graph_addr);

    printf("[%03d]: TreeUnitMagicFunc called with isStateFree = %p, nx = %d, ny = %d, nthetaIdx = %d, sine = %f, cosine = %f, g_graph = 0x%lx\n", 
        thread->get_uid(), result_addr, nx, ny, nthetaIdx, sine, cosine, g_graph_addr);

    int d_robotLength = 10; 
    int d_robotWidth = 5; 
    bool isStateFree = true;
    for (int __i = 0; __i <= d_robotLength; __i++) {

        for (int __j = 0; __j <= d_robotWidth; __j++) {
            int xbar = nx + round(__i * cosine);
            int ybar = ny + round(__j * sine);

            if (!isInRange(xbar, ybar)) {
                isStateFree = false;
                break;
            }

            int gIdx = xythetaToID(xbar, ybar, nthetaIdx);
            memory_space *mem = thread->get_global_memory();
            unsigned size = sizeof(uint8_t);
            uint8_t g_graph_value;
            mem->read(g_graph_addr + gIdx * size, size, &g_graph_value);
            if (g_graph_value == 0) {
                isStateFree = false;
                break;
            }

        }

        if (!isStateFree) break;
    }

    // Convert result_addr to local memory address
    new_addr_type result_addr_local = generic_to_local(thread->get_hw_sid(), thread->get_hw_tid(), result_addr);
    printf("[%03d]: TreeUnitMagicFunc finished with isStateFree = %d; writing to %p\n", thread->get_uid(), isStateFree, result_addr_local);
    thread->m_local_mem->write(result_addr_local, sizeof(int), &isStateFree, NULL, NULL);

}

void rt_ray_box_intersect(const ptx_instruction *pI, ptx_thread_info *thread) {

    // Read input arguments
    int arg = 0;
    assert(0 && "rt_ray_box_intersect not implemented yet\n");

}

void rt_ray_triangle_intersect(const ptx_instruction *pI, ptx_thread_info *thread) {

    // Read input arguments
    int arg = 0;
    assert(0 && "rt_ray_triangle_intersect not implemented yet\n");

}


std::ofstream print_tree;
void traverse_tree(volatile uint8_t* address, bool isTopLevel = true, bool isLeaf = false, bool isRoot = true)
{
    if(isRoot)
    {
        GEN_RT_BVH topBVH;
        GEN_RT_BVH_unpack(&topBVH, (uint8_t*)address);

        uint8_t* topRootAddr = (uint8_t*)address + topBVH.RootNodeOffset;

        if (print_tree.is_open())
        {
            print_tree << "traversing bvh , isTopLevel = " << isTopLevel << (void *)(address) << ", RootNodeOffset = (" << topBVH.RootNodeOffset << std::endl;
        }

        traverse_tree(topRootAddr, isTopLevel, false, false);
    }
    
    else if(!isLeaf) // internal nodes
    {
        struct GEN_RT_BVH_INTERNAL_NODE node;
        GEN_RT_BVH_INTERNAL_NODE_unpack(&node, address);

        if (print_tree.is_open())
        {
            uint8_t *child_addrs[6];
            child_addrs[0] = address + (node.ChildOffset * 64);
            for(int i = 0; i < 5; i++)
                child_addrs[i + 1] = child_addrs[i] + node.ChildSize[i] * 64;
            
            print_tree << "traversing internal node " << (void *)address;
            print_tree << ", isTopLevel = " << isTopLevel << ", child offset = " << node.ChildOffset << ", node type = " << node.NodeType;
            print_tree << ", child size = (" << node.ChildSize[0] << ", " << node.ChildSize[1] << ", " << node.ChildSize[2] << ", " << node.ChildSize[3] << ", " << node.ChildSize[4] << ", " << node.ChildSize[5] << ")";
            print_tree << ", child type = (" << node.ChildType[0] << ", " << node.ChildType[1] << ", " << node.ChildType[2] << ", " << node.ChildType[3] << ", " << node.ChildType[4] << ", " << node.ChildType[5] << ")";
            print_tree << ", child addresses = (" << (void*)(child_addrs[0]) << ", " << (void*)(child_addrs[1]) << ", " << (void*)(child_addrs[2]) << ", " << (void*)(child_addrs[3]) << ", " << (void*)(child_addrs[4]) << ", " << (void*)(child_addrs[5]) << ")";
            print_tree << std::endl;
        }

        uint8_t *child_addr = address + (node.ChildOffset * 64);
        for(int i = 0; i < 6; i++)
        {
            if(node.ChildSize[i] > 0)
            {
                if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                    isLeaf = true;
                else
                    isLeaf = false;

                traverse_tree(child_addr, isTopLevel, isLeaf, false);
            }

            child_addr += node.ChildSize[i] * 64;
        }
    }

    else // leaf nodes
    {
        if(isTopLevel)
        {
            GEN_RT_BVH_INSTANCE_LEAF instanceLeaf;
            GEN_RT_BVH_INSTANCE_LEAF_unpack(&instanceLeaf, address);

            float4x4 worldToObjectMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.WorldToObjectm00);
            float4x4 objectToWorldMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.ObjectToWorldm00);

            assert(instanceLeaf.BVHAddress != NULL);

            if (print_tree.is_open())
            {
                print_tree << "traversing top level leaf node " << (void *)address << ", instanceID = " << instanceLeaf.InstanceID << ", BVHAddress = " << instanceLeaf.BVHAddress << ", ShaderIndex = " << instanceLeaf.ShaderIndex << std::endl;
            }

            traverse_tree(address + instanceLeaf.BVHAddress, false, false, true);
        }
        else
        {
            struct GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR leaf_descriptor;
            GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_unpack(&leaf_descriptor, address);
            
            if (leaf_descriptor.LeafType == TYPE_QUAD)
            {
                struct GEN_RT_BVH_QUAD_LEAF leaf;
                GEN_RT_BVH_QUAD_LEAF_unpack(&leaf, address);

                float3 p[3];
                for(int i = 0; i < 3; i++)
                {
                    p[i].x = leaf.QuadVertex[i].X;
                    p[i].y = leaf.QuadVertex[i].Y;
                    p[i].z = leaf.QuadVertex[i].Z;
                }

                assert(leaf.PrimitiveIndex1Delta == 0);

                if (print_tree.is_open())
                {
                    print_tree << "quad node " << (void*)address << " ";
                    print_tree << "primitiveID = " << leaf.PrimitiveIndex0 << "\n";

                    print_tree << "p[0] = (" << p[0].x << ", " << p[0].y << ", " << p[0].z << ") ";
                    print_tree << "p[1] = (" << p[1].x << ", " << p[1].y << ", " << p[1].z << ") ";
                    print_tree << "p[2] = (" << p[2].x << ", " << p[2].y << ", " << p[2].z << ") ";
                    print_tree << "p[3] = (" << p[3].x << ", " << p[3].y << ", " << p[3].z << ")" << std::endl;
                }
            }
            else
            {
                struct GEN_RT_BVH_PROCEDURAL_LEAF leaf;
                GEN_RT_BVH_PROCEDURAL_LEAF_unpack(&leaf, address);

                if (print_tree.is_open())
                {
                    print_tree << "PROCEDURAL node " << (void*)address << " ";
                    print_tree << "NumPrimitives = " << leaf.NumPrimitives << ", LastPrimitive = " << leaf.LastPrimitive << ", PrimitiveIndex[0]" << leaf.PrimitiveIndex[0] << "\n";
                }
            }
        }
    }
}

void VulkanRayTracing::init(uint32_t launch_width, uint32_t launch_height)
{
    if(_init_)
        return;
    _init_ = true;

    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    uint32_t width = (launch_width + 31) / 32;
    uint32_t height = launch_height;

    if(ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->m_rt_intersection_table_type == 0)
        intersectionTableType = IntersectionTableType::Baseline;
    else if(ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->m_rt_intersection_table_type == 1)
        intersectionTableType = IntersectionTableType::Function_Call_Coalescing;
    else
        assert(0);

    if(intersectionTableType == IntersectionTableType::Baseline)
    {
        intersection_table = new Baseline_warp_intersection_table**[width];
        for(int i = 0; i < width; i++)
        {
            intersection_table[i] = new Baseline_warp_intersection_table*[height];
            for(int j = 0; j < height; j++)
                intersection_table[i][j] = new Baseline_warp_intersection_table();
        }
    }
    else
    {
        intersection_table = new Coalescing_warp_intersection_table**[width];
        for(int i = 0; i < width; i++)
        {
            intersection_table[i] = new Coalescing_warp_intersection_table*[height];
            for(int j = 0; j < height; j++)
                intersection_table[i][j] = new Coalescing_warp_intersection_table();
        }

    }
    anyhit_table = new Baseline_warp_intersection_table**[width];
    for(int i = 0; i < width; i++)
    {
        anyhit_table[i] = new Baseline_warp_intersection_table*[height];
        for(int j = 0; j < height; j++)
            anyhit_table[i][j] = new Baseline_warp_intersection_table();
    }
}


bool debugTraversal = false;
bool found_AS = false;
VkAccelerationStructureKHR topLevelAS_first = NULL;

void VulkanRayTracing::traceRay(VkAccelerationStructureKHR _topLevelAS,
				   uint rayFlags,
                   uint cullMask,
                   uint sbtRecordOffset,
                   uint sbtRecordStride,
                   uint missIndex,
                   float3 origin,
                   float Tmin,
                   float3 direction,
                   float Tmax,
                   int payload,
                   const ptx_instruction *pI,
                   ptx_thread_info *thread)
{
    // printf("## calling trceRay function. rayFlags = %d, cullMask = %d, sbtRecordOffset = %d, sbtRecordStride = %d, missIndex = %d, origin = (%f, %f, %f), Tmin = %f, direction = (%f, %f, %f), Tmax = %f, payload = %d\n",
    //         rayFlags, cullMask, sbtRecordOffset, sbtRecordStride, missIndex, origin.x, origin.y, origin.z, Tmin, direction.x, direction.y, direction.z, Tmax, payload);

    if (dump_trace && !dumped) 
    {
        dump_AS(VulkanRayTracing::descriptorSet, _topLevelAS);
        std::cout << "Trace dumped" << std::endl;
        dumped = true;
    }

    // Convert device address back to host address for func sim. This will break if the device address was modified then passed to traceRay. Should be fixable if I also record the size when I malloc then I can check the bounds of the device address.
    uint8_t* deviceAddress = nullptr;
    int64_t device_offset = (uint64_t)tlas_addr - (uint64_t)_topLevelAS;
    if (GPGPU_Context()->func_sim->g_rt_external_launch)
    {
        deviceAddress = (uint8_t*)_topLevelAS;
        bool addressFound = false;
        for (int i = 0; i < MAX_DESCRIPTOR_SETS; i++)
        {
            for (int j = 0; j < MAX_DESCRIPTOR_SET_BINDINGS; j++)
            {
                if (launcher_deviceDescriptorSets[i][j] == (void*)_topLevelAS)
                {
                    _topLevelAS = launcher_descriptorSets[i][j];
                    addressFound = true;
                    break;
                }
            }
            if (addressFound)
                break;
        }
        if (!addressFound)
            abort();
    
        // Calculate offset between host and device for memory transactions
        device_offset = (uint64_t)deviceAddress - (uint64_t)_topLevelAS;
    }

    // if(!found_AS)
    // {
    //     found_AS = true;
    //     topLevelAS_first = _topLevelAS;
    //     print_tree.open("bvh_tree.txt");
    //     traverse_tree((uint8_t*)_topLevelAS);
    //     print_tree.close();
    // }
    // else
    // {
    //     assert(topLevelAS_first != NULL);
    //     assert(topLevelAS_first == _topLevelAS);
    // }

    Traversal_data traversal_data;

    traversal_data.n_all_hits = 0;
    traversal_data.ray_world_direction = direction;
    traversal_data.ray_world_origin = origin;
    traversal_data.sbtRecordOffset = sbtRecordOffset;
    traversal_data.sbtRecordStride = sbtRecordStride;
    traversal_data.missIndex = missIndex;
    traversal_data.Tmin = Tmin;
    traversal_data.Tmax = Tmax;

    bool hit_procedural = false;

    std::ofstream traversalFile;


    bool terminateOnFirstHit = rayFlags & SpvRayFlagsTerminateOnFirstHitKHRMask;
    bool skipClosestHitShader = rayFlags & SpvRayFlagsSkipClosestHitShaderKHRMask;
    bool skipAnyHitShader = rayFlags & SpvRayFlagsOpaqueKHRMask;
    bool rtnn = rayFlags & SpvRayFlagsSkipAABBsKHRMask;
    bool raysphere = rayFlags & SpvRayFlagsSkipTrianglesKHRMask;

    std::vector<unsigned> rtnn_sphere_hit_indices;
    // Currently max at 8 hits per ray
    unsigned rtnn_max_hits = 128; // TODO: make this configurable
    if (rtnn) {
        VSIM_DPRINTF("Using GPRT for rtnn \n");
    }
    else if (raysphere) {
        VSIM_DPRINTF("Using GPRT for ray-sphere. \n");
    }

    if (debugTraversal)
    {
        traversalFile.open("traversal.txt", std::ios_base::app);
        traversalFile << "RAY: ";
        if (terminateOnFirstHit) 
            traversalFile << "(anyhit) ";
        traversalFile << "origin = (" << origin.x << ", " << origin.y << ", " << origin.z << "), ";
        traversalFile << "direction = (" << direction.x << ", " << direction.y << ", " << direction.z << "), ";
        traversalFile << "tmin = " << Tmin << ", tmax = " << Tmax << std::endl << std::endl;
    }

    std::vector<MemoryTransactionRecord> transactions;
    std::vector<MemoryStoreTransactionRecord> store_transactions;

    gpgpu_context *ctx = GPGPU_Context();
    memory_space *mem = thread->get_global_memory();
    int key = ctx->func_sim->g_rt_traversal_key;

    if (terminateOnFirstHit) ctx->func_sim->g_n_anyhit_rays++;
    else ctx->func_sim->g_n_closesthit_rays++;

    unsigned total_nodes_accessed = 0;
    std::map<uint8_t*, unsigned> tree_level_map;
    
	// Create ray
	Ray ray;
	ray.make_ray(origin, direction, Tmin, Tmax);
    thread->add_ray_properties(ray);

	// Set thit to max
    float min_thit = ray.dir_tmax.w;
    struct GEN_RT_BVH_QUAD_LEAF closest_leaf;
    struct GEN_RT_BVH_PROCEDURAL_LEAF closest_proceduralleaf;
    float4 closest_sphere;
    struct GEN_RT_BVH_INSTANCE_LEAF closest_instanceLeaf;    
    float4x4 closest_worldToObject, closest_objectToWorld;
    Ray closest_objectRay;
    float min_thit_object;

	// Get bottom-level AS
    //uint8_t* topLevelASAddr = get_anv_accel_address((VkAccelerationStructureKHR)_topLevelAS);
    GEN_RT_BVH topBVH; //TODO: test hit with world before traversal
    GEN_RT_BVH_unpack(&topBVH, (uint8_t*)_topLevelAS);
    transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)_topLevelAS + device_offset), GEN_RT_BVH_length * 4, TransactionType::BVH_STRUCTURE));
    ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_STRUCTURE)]++;

    uint8_t* topRootAddr = (uint8_t*)_topLevelAS + topBVH.RootNodeOffset;

    // Get min/max
    if (!ctx->func_sim->g_rt_world_set) {
        struct GEN_RT_BVH_INTERNAL_NODE node;
        GEN_RT_BVH_INTERNAL_NODE_unpack(&node, topRootAddr);
        for(int i = 0; i < 6; i++) {
            if (node.ChildSize[i] > 0) {
                float3 idir = calculate_idir(ray.get_direction()); //TODO: this works wierd if one of ray dimensions is 0
                float3 lo, hi;
                set_child_bounds(&node, i, &lo, &hi);
                ctx->func_sim->g_rt_world_min = min(ctx->func_sim->g_rt_world_min, lo);
                ctx->func_sim->g_rt_world_max = min(ctx->func_sim->g_rt_world_max, hi);
            }
        }
        ctx->func_sim->g_rt_world_set = true;
    }

    std::list<StackEntry> stack;
    tree_level_map[topRootAddr] = 1;
    
    {
        float3 lo, hi;
        lo.x = topBVH.BoundsMin.X;
        lo.y = topBVH.BoundsMin.Y;
        lo.z = topBVH.BoundsMin.Z;
        hi.x = topBVH.BoundsMax.X;
        hi.y = topBVH.BoundsMax.Y;
        hi.z = topBVH.BoundsMax.Z;

        float thit;
        if(ray_box_test(lo, hi, calculate_idir(ray.get_direction()), ray.get_origin(), ray.get_tmin(), ray.get_tmax(), thit))
            stack.push_back(StackEntry(topRootAddr, true, false));
    }

    while (!stack.empty())
    {
        uint8_t *node_addr = NULL;
        uint8_t *next_node_addr = NULL;

        // traverse top level internal nodes
        assert(stack.back().topLevel);
        
        if(!stack.back().leaf)
        {
            next_node_addr = stack.back().addr;
            stack.pop_back();
        }

        while (next_node_addr > 0)
        {
            // TLAS offset
            device_offset = (uint64_t)tlas_addr - (uint64_t)_topLevelAS;

            node_addr = next_node_addr;
            next_node_addr = NULL;
            struct GEN_RT_BVH_INTERNAL_NODE node;
            GEN_RT_BVH_INTERNAL_NODE_unpack(&node, node_addr);
            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)node_addr + device_offset), GEN_RT_BVH_INTERNAL_NODE_length * 4, TransactionType::BVH_INTERNAL_NODE));
            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;
            total_nodes_accessed++;

            if (debugTraversal)
            {
                traversalFile << "traversing top level internal node " << (void *)node_addr;
                traversalFile << ", child offset = " << node.ChildOffset << ", node type = " << node.NodeType;
                traversalFile << ", child size = (" << node.ChildSize[0] << ", " << node.ChildSize[1] << ", " << node.ChildSize[2] << ", " << node.ChildSize[3] << ", " << node.ChildSize[4] << ", " << node.ChildSize[5] << ")";
                traversalFile << ", child type = (" << node.ChildType[0] << ", " << node.ChildType[1] << ", " << node.ChildType[2] << ", " << node.ChildType[3] << ", " << node.ChildType[4] << ", " << node.ChildType[5] << ")";
                traversalFile << std::endl;
            }

            uint8_t *child_addr = node_addr + (node.ChildOffset * 64);

            child_node children[6];
            for(int i = 0; i < 6; i++)
            {
                children[i].child_index = i;
                children[i].child_addr = child_addr;

                if (node.ChildSize[i] > 0)
                {
                    float3 idir = calculate_idir(ray.get_direction()); //TODO: this works wierd if one of ray dimensions is 0
                    float3 lo, hi;
                    set_child_bounds(&node, i, &lo, &hi);

                    children[i].hit = ray_box_test(lo, hi, idir, ray.get_origin(), ray.get_tmin(), ray.get_tmax(), children[i].thit);
                    if(children[i].hit && children[i].thit >= min_thit)
                        children[i].hit = false;

                    // Calculate surface area
                    children[i].surface_area = 0.0f;
                    if (children[i].hit) {
                        float x_len = hi.x - lo.x;
                        float y_len = hi.y - lo.y;
                        float z_len = hi.z - lo.z;
                        float SA = (2 * x_len * y_len) + (2 * x_len * z_len) + (2 * y_len * z_len);
                        children[i].surface_area = SA;
                    }

                    if (debugTraversal)
                    {
                        if (children[i].hit) {
                            traversalFile << "hit child number " << i << ", ";
                            traversalFile << "SA = " << children[i].surface_area << ", ";
                            traversalFile << "thit = " << children[i].thit << ", ";
                        }
                        else {
                            traversalFile << "missed child number " << i << ", ";
                        }
                        traversalFile << "lo = (" << lo.x << ", " << lo.y << ", " << lo.z << "), ";
                        traversalFile << "hi = (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;
                    }
                }
                else
                    children[i].hit = false;

                child_addr += node.ChildSize[i] * 64;
            }


            // Reorder child nodes
            std::map<float,child_node> reordered_children; 
            reorder_child_nodes(children, reordered_children, key);

            if (debugTraversal)
                traversalFile << "Reordered children: ";
            for (auto child : reordered_children) {
                int i = child.second.child_index;
                uint8_t *addr_i = child.second.child_addr;
                if (debugTraversal)
                    traversalFile << i << " (" << (void *)addr_i << "), ";

                if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                {
                    assert(node.ChildType[i] == NODE_TYPE_INSTANCE);
                    stack.push_back(StackEntry(addr_i, true, true));
                    assert(tree_level_map.find(node_addr) != tree_level_map.end());
                    tree_level_map[addr_i] = tree_level_map[node_addr] + 1;
                }
                else
                {
                    stack.push_back(StackEntry(addr_i, true, false));
                    assert(tree_level_map.find(node_addr) != tree_level_map.end());
                    tree_level_map[addr_i] = tree_level_map[node_addr] + 1;
                }
            }

            if (reordered_children.size() > 0 && next_node_addr == NULL && !stack.back().leaf) {
                next_node_addr = stack.back().addr;
                stack.pop_back();
            }

            if (debugTraversal)
            {
                traversalFile << std::endl;
            }
        }

        // traverse top level leaf nodes
        while (!stack.empty() && stack.back().leaf)
        {
            // TLAS offset
            device_offset = (uint64_t)tlas_addr - (uint64_t)_topLevelAS;

            assert(stack.back().topLevel);

            uint8_t* leaf_addr = stack.back().addr;
            stack.pop_back();

            GEN_RT_BVH_INSTANCE_LEAF instanceLeaf;
            GEN_RT_BVH_INSTANCE_LEAF_unpack(&instanceLeaf, leaf_addr);
            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_INSTANCE_LEAF_length * 4, TransactionType::BVH_INSTANCE_LEAF));
            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INSTANCE_LEAF)]++;
            total_nodes_accessed++;


            if (debugTraversal)
            {
                traversalFile << "traversing top level leaf node " << (void *)leaf_addr << ", instanceID = " << instanceLeaf.InstanceID << ", BVHAddress = " << instanceLeaf.BVHAddress << ", ShaderIndex = " << instanceLeaf.ShaderIndex << std::endl;
            }


            float4x4 worldToObjectMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.WorldToObjectm00);
            float4x4 objectToWorldMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.ObjectToWorldm00);

            assert(instanceLeaf.BVHAddress != NULL);
            GEN_RT_BVH botLevelASAddr;
            GEN_RT_BVH_unpack(&botLevelASAddr, (uint8_t *)(leaf_addr + instanceLeaf.BVHAddress));

            // BLAS offset
            uint8_t * botLevelRootAddr = (uint8_t *)(leaf_addr + instanceLeaf.BVHAddress);
            RT_DPRINTF("Traversing BLAS %p -> %p\n", (void*)botLevelRootAddr, blas_addr_map[(void*)botLevelRootAddr]);
            assert(blas_addr_map.find((void*)botLevelRootAddr) != blas_addr_map.end());
            device_offset = (uint64_t)blas_addr_map[(void*)botLevelRootAddr] - (uint64_t)botLevelRootAddr;

            transactions.push_back(MemoryTransactionRecord((uint8_t*)(botLevelRootAddr + device_offset), GEN_RT_BVH_length * 4, TransactionType::BVH_STRUCTURE));
            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_STRUCTURE)]++;

            if (debugTraversal)
            {
                traversalFile << "bot level bvh " << (void *)(leaf_addr + instanceLeaf.BVHAddress) << ", RootNodeOffset = (" << botLevelASAddr.RootNodeOffset << std::endl;
            }

            // std::ofstream offsetfile;
            // offsetfile.open("offsets.txt", std::ios::app);
            // offsetfile << (int64_t)instanceLeaf.BVHAddress << std::endl;

            // std::ofstream leaf_addr_file;
            // leaf_addr_file.open("leaf.txt", std::ios::app);
            // leaf_addr_file << (int64_t)((uint64_t)leaf_addr - (uint64_t)_topLevelAS) << std::endl;

            float worldToObject_tMultiplier;
            Ray objectRay = make_transformed_ray(ray, worldToObjectMatrix, &worldToObject_tMultiplier);
            
            botLevelRootAddr = ((uint8_t *)((uint64_t)leaf_addr + instanceLeaf.BVHAddress)) + botLevelASAddr.RootNodeOffset;
            stack.push_back(StackEntry(botLevelRootAddr, false, false));
            assert(tree_level_map.find(leaf_addr) != tree_level_map.end());
            tree_level_map[botLevelRootAddr] = tree_level_map[leaf_addr];

            if (debugTraversal)
            {
                traversalFile << "bot level root address = " << (void*)botLevelRootAddr << std::endl;
                traversalFile << "warped ray to object coordinates, origin = (" << objectRay.get_origin().x << ", " << objectRay.get_origin().y << ", " << objectRay.get_origin().z << "), ";
                traversalFile << "direction = (" << objectRay.get_direction().x << ", " << objectRay.get_direction().y << ", " << objectRay.get_direction().z << "), ";
                traversalFile << "tmin = " << objectRay.get_tmin() << ", tmax = " << objectRay.get_tmax() << std::endl << std::endl;
            }

            // traverse bottom level tree
            while (!stack.empty() && !stack.back().topLevel)
            {
                uint8_t* node_addr = NULL;
                uint8_t* next_node_addr = stack.back().addr;
                stack.pop_back();
                

                // traverse bottom level internal nodes
                while (next_node_addr > 0)
                {
                    node_addr = next_node_addr;
                    next_node_addr = NULL;

                    // if(node_addr == *(++path.rbegin()))
                    //     printf("this is where things go wrong\n");

                    struct GEN_RT_BVH_INTERNAL_NODE node;
                    GEN_RT_BVH_INTERNAL_NODE_unpack(&node, node_addr);
                    transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)node_addr + device_offset), GEN_RT_BVH_INTERNAL_NODE_length * 4, TransactionType::BVH_INTERNAL_NODE));
                    ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;
                    total_nodes_accessed++;

                    if (debugTraversal)
                    {
                        traversalFile << "traversing bot level internal node " << (void *)node_addr;
                        traversalFile << ", child offset = " << node.ChildOffset << ", node type = " << node.NodeType;
                        traversalFile << ", child size = (" << node.ChildSize[0] << ", " << node.ChildSize[1] << ", " << node.ChildSize[2] << ", " << node.ChildSize[3] << ", " << node.ChildSize[4] << ", " << node.ChildSize[5] << ")";
                        traversalFile << ", child type = (" << node.ChildType[0] << ", " << node.ChildType[1] << ", " << node.ChildType[2] << ", " << node.ChildType[3] << ", " << node.ChildType[4] << ", " << node.ChildType[5] << ")";
                        traversalFile << std::endl;
                    }

                    child_node children[6];
                    uint8_t *child_addr = node_addr + (node.ChildOffset * 64);

                    for(int i = 0; i < 6; i++)
                    {
                        children[i].child_index = i;
                        children[i].child_addr = child_addr;

                        if (node.ChildSize[i] > 0)
                        {
                            float3 idir = calculate_idir(objectRay.get_direction()); //TODO: this works wierd if one of ray dimensions is 0
                            float3 lo, hi;
                            set_child_bounds(&node, i, &lo, &hi);

                            children[i].hit = ray_box_test(lo, hi, idir, objectRay.get_origin(), objectRay.get_tmin(), objectRay.get_tmax(), children[i].thit);
                            if(children[i].hit && children[i].thit >= min_thit * worldToObject_tMultiplier)
                                children[i].hit = false;

                            // Calculate surface area
                            children[i].surface_area = 0.0f;
                            if (children[i].hit) {
                                float x_len = hi.x - lo.x;
                                float y_len = hi.y - lo.y;
                                float z_len = hi.z - lo.z;
                                float SA = (2 * x_len * y_len) + (2 * x_len * z_len) + (2 * y_len * z_len);
                                children[i].surface_area = SA;
                            }

                            if (debugTraversal)
                            {
                                if(children[i].hit) {
                                    traversalFile << "hit child number " << i << ", ";
                                    traversalFile << "SA = " << children[i].surface_area << ", ";
                                    traversalFile << "thit = " << children[i].thit << ", ";
                                }
                                else {
                                    traversalFile << "missed child number " << i << ", ";
                                }
                                traversalFile << "lo = (" << lo.x << ", " << lo.y << ", " << lo.z << "), ";
                                traversalFile << "hi = (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;
                            }
                        }
                        else
                            children[i].hit = false;

                        child_addr += node.ChildSize[i] * 64;
                    }

                    // Reorder child nodes
                    std::map<float,child_node> reordered_children;
                    reorder_child_nodes(children, reordered_children, key);

                    if (debugTraversal)
                        traversalFile << "Reordered children: ";
                    for (auto child : reordered_children) {
                        int i = child.second.child_index;
                        uint8_t *addr_i = child.second.child_addr;
                        if (debugTraversal)
                            traversalFile << i << " (" << (void *)addr_i << "), ";
                        
                        if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                        {
                            stack.push_back(StackEntry(addr_i, false, true));
                            assert(tree_level_map.find(node_addr) != tree_level_map.end());
                            tree_level_map[addr_i] = tree_level_map[node_addr] + 1;
                        }
                        else
                        {
                            stack.push_back(StackEntry(addr_i, false, false));
                            assert(tree_level_map.find(node_addr) != tree_level_map.end());
                            tree_level_map[addr_i] = tree_level_map[node_addr] + 1;
                        }
                    }

                    if (reordered_children.size() > 0 && next_node_addr == NULL && !stack.back().leaf) {
                        next_node_addr = stack.back().addr;
                        stack.pop_back();
                    }

                    if (debugTraversal)
                    {
                        traversalFile << std::endl;
                    }
                }

                // traverse bottom level leaf nodes
                while(!stack.empty() && !stack.back().topLevel && stack.back().leaf)
                {
                    uint8_t* leaf_addr = stack.back().addr;
                    stack.pop_back();
                    struct GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR leaf_descriptor;
                    GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_unpack(&leaf_descriptor, leaf_addr);
                    transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_length * 4, TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR));
                    ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR)]++;

                    if (leaf_descriptor.LeafType == TYPE_QUAD)
                    {
                        struct GEN_RT_BVH_QUAD_LEAF leaf;
                        GEN_RT_BVH_QUAD_LEAF_unpack(&leaf, leaf_addr);

                        // if(leaf.PrimitiveIndex0 == 9600)
                        // {
                        //     leaf.QuadVertex[2].Z = -0.001213;
                        // }

                        float3 p[3];
                        for(int i = 0; i < 3; i++)
                        {
                            p[i].x = leaf.QuadVertex[i].X;
                            p[i].y = leaf.QuadVertex[i].Y;
                            p[i].z = leaf.QuadVertex[i].Z;
                        }

                        // Triangle intersection algorithm
                        float thit;
                        bool hit = VulkanRayTracing::mt_ray_triangle_test(p[0], p[1], p[2], objectRay, &thit);

                        assert(leaf.PrimitiveIndex1Delta == 0);

                        if (debugTraversal)
                        {
                            if(hit)
                                traversalFile << "hit quad node " << (void *)leaf_addr << " with thit " << thit << " ";
                            else
                                traversalFile << "miss quad node " << leaf_addr << " ";
                            traversalFile << "primitiveID = " << leaf.PrimitiveIndex0 << ", InstanceID = " << instanceLeaf.InstanceID << "\n";

                            traversalFile << "p[0] = (" << p[0].x << ", " << p[0].y << ", " << p[0].z << ") ";
                            traversalFile << "p[1] = (" << p[1].x << ", " << p[1].y << ", " << p[1].z << ") ";
                            traversalFile << "p[2] = (" << p[2].x << ", " << p[2].y << ", " << p[2].z << ") ";
                            traversalFile << "p[3] = (" << p[3].x << ", " << p[3].y << ", " << p[3].z << ")" << std::endl;
                        }

                        float world_thit = thit / worldToObject_tMultiplier;

                        //TODO: why the Tmin Tmax consition wasn't handled in the object coordinates?
                        if(hit && Tmin <= world_thit && world_thit <= Tmax)
                        {
                            if (debugTraversal)
                            {
                                traversalFile << "quad node " << (void *)leaf_addr << ", primitiveID " << leaf.PrimitiveIndex0 << " is the closest hit. world_thit " << thit / worldToObject_tMultiplier;
                            }

                            if (skipAnyHitShader && world_thit < min_thit) {
                                min_thit = thit / worldToObject_tMultiplier;
                            }
                            min_thit_object = thit;
                            closest_leaf = leaf;
                            closest_instanceLeaf = instanceLeaf;
                            closest_worldToObject = worldToObjectMatrix;
                            closest_objectToWorld = objectToWorldMatrix;
                            closest_objectRay = objectRay;
                            min_thit_object = thit;
                            thread->add_ray_intersect();
                            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_QUAD_LEAF_length * 4, TransactionType::BVH_QUAD_LEAF_HIT));
                            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF_HIT)]++;
                            total_nodes_accessed++;

                            if (!skipAnyHitShader) {
                                VSIM_DPRINTF("gpgpusim: Adding triangle intersection to anyhit shader table\n");
                                warp_intersection_table* table = anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
                                
                                uint32_t hit_group_index = instanceLeaf.InstanceContributionToHitGroupIndex;
                                auto intersectionTransactions = table->add_intersection(hit_group_index, thread->get_tid().x, leaf.PrimitiveIndex0, instanceLeaf.InstanceID, pI, thread); // TODO: switch these to device addresses

                                for(auto & newTransaction : intersectionTransactions.first)
                                {
                                    bool found = false;
                                    for(auto & transaction : transactions)
                                        if(transaction.address == newTransaction.address)
                                        {
                                            found = true;
                                            break;
                                        }
                                    if(!found)
                                        transactions.push_back(newTransaction);

                                }
                                store_transactions.insert(store_transactions.end(), intersectionTransactions.second.begin(), intersectionTransactions.second.end());

                                VSIM_DPRINTF("gpgpusim: Storing triangle intersection HitAttributes for anyhit shader\n");

                                ctx->func_sim->g_rt_num_any_hits++;

                                Hit_data anyhit_hit_attributes;
                                anyhit_hit_attributes.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                                anyhit_hit_attributes.geometry_index = leaf.LeafDescriptor.GeometryIndex;
                                anyhit_hit_attributes.primitive_index = leaf.PrimitiveIndex0;
                                anyhit_hit_attributes.instance_index = instanceLeaf.InstanceID;

                                float anyhit_thit = thit / worldToObject_tMultiplier;
                                float3 intersection_point = ray.get_origin() + make_float3(ray.get_direction().x * anyhit_thit, ray.get_direction().y * anyhit_thit, ray.get_direction().z * anyhit_thit);
                                float3 rayatinter = ray.at(anyhit_thit);

                                anyhit_hit_attributes.intersection_point = intersection_point;
                                anyhit_hit_attributes.worldToObjectMatrix = worldToObjectMatrix;
                                anyhit_hit_attributes.objectToWorldMatrix = objectToWorldMatrix;
                                anyhit_hit_attributes.world_min_thit = anyhit_thit;
                                
                                float3 p[3];
                                for(int i = 0; i < 3; i++)
                                {
                                    p[i].x = leaf.QuadVertex[i].X;
                                    p[i].y = leaf.QuadVertex[i].Y;
                                    p[i].z = leaf.QuadVertex[i].Z;
                                }
                                float3 object_intersection_point = objectRay.get_origin() + make_float3(objectRay.get_direction().x * thit, objectRay.get_direction().y * thit, objectRay.get_direction().z * thit);
                                float3 barycentric = Barycentric(object_intersection_point, p[0], p[1], p[2]);
                                anyhit_hit_attributes.barycentric_coordinates = barycentric;

                                VSIM_DPRINTF("gpgpusim: Ray hit geomID %d primID %d at (%5.3f, %5.3f, %5.3f) with t = %5.3f\n", anyhit_hit_attributes.geometry_index, anyhit_hit_attributes.primitive_index, barycentric.x, barycentric.y, barycentric.z, thit);

                                // Allocate memory to store hit attributes
                                memory_space *mem = thread->get_global_memory();
                                Hit_data* device_hit_attributes = (Hit_data*) VulkanRayTracing::gpgpusim_alloc(sizeof(Hit_data));
                                mem->write(device_hit_attributes, sizeof(Hit_data), &anyhit_hit_attributes, thread, pI);
                                thread->RT_thread_data->all_hit_data.push_back(device_hit_attributes);

                                traversal_data.n_all_hits++;
                            }

                            if(terminateOnFirstHit)
                            {
                                stack.clear();
                            }
                        }
                        else {
                            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_QUAD_LEAF_length * 4, TransactionType::BVH_QUAD_LEAF));
                            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF)]++;
                            total_nodes_accessed++;
                        }
                        if (debugTraversal)
                        {
                            traversalFile << std::endl;
                        }
                    }
                    else if (rtnn) {
                        // Get the sphere array (currently hard-coded to binding 7)
                        void* sphere_array_addr = VulkanRayTracing::getDescriptorAddress(0, 7);
                        uint32_t sphere_index = instanceLeaf.InstanceID;

                        // Read sphere
                        // const vec4 sphere = Spheres[gl_InstanceCustomIndexEXT];
                        // const vec3 center = sphere.xyz;
                        // const float radius = sphere.w;
                        float4 sphere;
                        mem->read(sphere_array_addr + (sphere_index * sizeof(float4)), sizeof(float4), &sphere);
                        transactions.push_back(MemoryTransactionRecord((uint8_t*)(sphere_array_addr + (sphere_index * sizeof(float4))), sizeof(float4), TransactionType::BVH_QUAD_LEAF));

                        VSIM_DPRINTF("[%5d] Sphere %d: (%f, %f, %f), %f ->", thread->get_uid()-1, sphere_index, sphere.x, sphere.y, sphere.z, sphere.w);

                        // Process ray-sphere intersection
                        bool hit = VulkanRayTracing::pointSphereIntersection(sphere, objectRay);
                        VSIM_DPRINTF(" %d\n", hit);

                        if (hit) {
                            ctx->func_sim->g_rt_num_hits++;
                            rtnn_sphere_hit_indices.push_back(sphere_index);
                        }

                        // Early termination when result buffer is full
                        // if (rtnn_sphere_hit_indices.size() >= rtnn_max_hits) {
                        //     VSIM_DPRINTF("[%5d] Result buffer full, terminating\n", thread->get_uid());
                        //     stack.clear();
                        //     ctx->func_sim->g_rtnn_max_hits_reached++;
                        // }
                    }
                    else if (raysphere) {
                        struct GEN_RT_BVH_PROCEDURAL_LEAF leaf;
                        GEN_RT_BVH_PROCEDURAL_LEAF_unpack(&leaf, leaf_addr);
                        ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_PROCEDURAL_LEAF)]++;

                        // Get the sphere array (currently hard-coded to binding 9)
                        void* sphere_array_addr = VulkanRayTracing::getDescriptorAddress(0, 9);
                        uint32_t sphere_index = instanceLeaf.InstanceID;

                        // Read sphere
                        // const vec4 sphere = Spheres[gl_InstanceCustomIndexEXT];
                        // const vec3 center = sphere.xyz;
                        // const float radius = sphere.w;
                        float4 sphere;
                        mem->read(sphere_array_addr + (sphere_index * sizeof(float4)), sizeof(float4), &sphere);

                        VSIM_DPRINTF("[%5d] Sphere %d: (%f, %f, %f), %f ->", thread->get_uid()-1, sphere_index, sphere.x, sphere.y, sphere.z, sphere.w);

                        // Process ray-sphere intersection
                        float thit;
                        bool hit = VulkanRayTracing::raySphereIntersection(sphere, objectRay, &thit);
                        float world_thit = thit / worldToObject_tMultiplier;
                        VSIM_DPRINTF(" %d\n", hit);

                        if(hit && Tmin <= world_thit && world_thit <= Tmax) {
                            transactions.push_back(MemoryTransactionRecord((void*)((uint32_t)sphere_array_addr + (sphere_index * sizeof(float4))), sizeof(float4), TransactionType::BVH_QUAD_LEAF_HIT));
                            // Need geometryType, hitGroupIndex, world_min_thit, primitive_index, instance_index

                            // WKND should always have skipAnyHitShader flag
                            if (skipAnyHitShader && thit < min_thit) {
                                min_thit = thit / worldToObject_tMultiplier;
                            }
                            min_thit_object = thit;
                            closest_proceduralleaf = leaf;
                            closest_instanceLeaf = instanceLeaf;
                            closest_worldToObject = worldToObjectMatrix;
                            closest_objectToWorld = objectToWorldMatrix;
                            closest_objectRay = objectRay;
                            closest_sphere = sphere;

                            thread->add_ray_intersect();
                            if(terminateOnFirstHit)
                            {
                                stack.clear();
                            }
                        }
                        else {
                            transactions.push_back(MemoryTransactionRecord((void*)((uint32_t)sphere_array_addr + (sphere_index * sizeof(float4))), sizeof(float4), TransactionType::BVH_QUAD_LEAF));
                        }
                    }
                    else
                    {
                        hit_procedural = true;
                        struct GEN_RT_BVH_PROCEDURAL_LEAF leaf;
                        GEN_RT_BVH_PROCEDURAL_LEAF_unpack(&leaf, leaf_addr);
                        transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_PROCEDURAL_LEAF_length * 4, TransactionType::BVH_PROCEDURAL_LEAF));
                        ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_PROCEDURAL_LEAF)]++;
                        total_nodes_accessed++;

                        uint32_t hit_group_index = instanceLeaf.InstanceContributionToHitGroupIndex;

                        warp_intersection_table* table = intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
                        auto intersectionTransactions = table->add_intersection(hit_group_index, thread->get_tid().x, leaf.PrimitiveIndex[0], instanceLeaf.InstanceID, pI, thread); // TODO: switch these to device addresses
                        
                        // transactions.insert(transactions.end(), intersectionTransactions.first.begin(), intersectionTransactions.first.end());
                        for(auto & newTransaction : intersectionTransactions.first)
                        {
                            bool found = false;
                            for(auto & transaction : transactions)
                                if(transaction.address == newTransaction.address)
                                {
                                    found = true;
                                    break;
                                }
                            if(!found)
                                transactions.push_back(newTransaction);

                        }
                        store_transactions.insert(store_transactions.end(), intersectionTransactions.second.begin(), intersectionTransactions.second.end());
                    }
                }
            }
        }
    }

    if (min_thit < ray.dir_tmax.w)
    {
        VSIM_DPRINTF("[%5d] Ray hit at t = %f\n", thread->get_uid(), min_thit);
        if (raysphere) {
            ctx->func_sim->g_rt_num_hits++;
            uint32_t hit_group_index = closest_instanceLeaf.InstanceContributionToHitGroupIndex;

            traversal_data.hit_geometry = true;
            traversal_data.closest_hit.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
            traversal_data.closest_hit.hitGroupIndex = hit_group_index;
            traversal_data.closest_hit.world_min_thit = min_thit;
            traversal_data.closest_hit.primitive_index = closest_proceduralleaf.PrimitiveIndex[0];
            traversal_data.closest_hit.instance_index = closest_instanceLeaf.InstanceID;

            thread->RT_thread_data->set_sphereAttribute(closest_sphere, pI, thread);
        }
        else {
            traversal_data.hit_geometry = true;
            ctx->func_sim->g_rt_num_hits++;
            traversal_data.closest_hit.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            traversal_data.closest_hit.geometry_index = closest_leaf.LeafDescriptor.GeometryIndex;
            traversal_data.closest_hit.primitive_index = closest_leaf.PrimitiveIndex0;
            traversal_data.closest_hit.instance_index = closest_instanceLeaf.InstanceID;
            float3 intersection_point = ray.get_origin() + make_float3(ray.get_direction().x * min_thit, ray.get_direction().y * min_thit, ray.get_direction().z * min_thit);
            float3 rayatinter = ray.at(min_thit);
            // assert(intersection_point.x == ray.at(min_thit).x && intersection_point.y == ray.at(min_thit).y && intersection_point.z == ray.at(min_thit).z);
            traversal_data.closest_hit.intersection_point = intersection_point;
            traversal_data.closest_hit.worldToObjectMatrix = closest_worldToObject;
            traversal_data.closest_hit.objectToWorldMatrix = closest_objectToWorld;
            traversal_data.closest_hit.world_min_thit = min_thit;

            VSIM_DPRINTF("gpgpusim: Ray hit geomID %d primID %d\n", traversal_data.closest_hit.geometry_index, traversal_data.closest_hit.primitive_index);
            VSIM_DPRINTF("gpgpusim: Ray [%d] awaiting %d anyhit shader calls\n", thread->get_uid(), traversal_data.n_all_hits);
            assert(thread->RT_thread_data->all_hit_data.size() == traversal_data.n_all_hits);
            float3 p[3];
            for(int i = 0; i < 3; i++)
            {
                p[i].x = closest_leaf.QuadVertex[i].X;
                p[i].y = closest_leaf.QuadVertex[i].Y;
                p[i].z = closest_leaf.QuadVertex[i].Z;
            }
            float3 object_intersection_point = closest_objectRay.get_origin() + make_float3(closest_objectRay.get_direction().x * min_thit_object, closest_objectRay.get_direction().y * min_thit_object, closest_objectRay.get_direction().z * min_thit_object);
            //closest_objectRay.at(min_thit_object);
            float3 barycentric = Barycentric(object_intersection_point, p[0], p[1], p[2]);
            traversal_data.closest_hit.barycentric_coordinates = barycentric;
            thread->RT_thread_data->set_hitAttribute(barycentric, pI, thread);

            // store_transactions.push_back(MemoryStoreTransactionRecord(&traversal_data, sizeof(traversal_data), StoreTransactionType::Traversal_Results));
            if (debugTraversal)
                traversalFile << "HIT" << std::endl;
        }
    }
    else if (hit_procedural)
    {
        VSIM_DPRINTF("gpgpusim: Ray hit procedural geometry; requires intersection shader.\n");
        traversal_data.hit_geometry = false;
    }
    else if (rtnn && rtnn_sphere_hit_indices.size() > 0) {
        VSIM_DPRINTF("[%5d]: writing hit to payload \n", thread->get_uid());

        // Get output image location (hard-coded to binding 1)
        struct DESCRIPTOR_STRUCT* image_desc = (struct DESCRIPTOR_STRUCT*)VulkanRayTracing::getDescriptorAddress(0, 1);
        struct lvp_image *image = (struct lvp_image *)image_desc->info.image_view.image;
        VkFormat vk_format = image->vk.format;

        uint32_t index = thread->get_uid() - 1;

        // Get size of writeback (assuming 4 bytes per sphere index)
        unsigned writeback_size = std::ceil(rtnn_sphere_hit_indices.size() / 8.0);

        transactions.push_back(MemoryTransactionRecord(
            image->pmem_gpgpusim + (rtnn_max_hits * index * sizeof(uint32_t)),
            32,
            TransactionType::WRITE_TRAVERSAL_RESULT)
        );

        if (writeback_size > 1) {
            for (unsigned i = 1; i < writeback_size; i++) {
                transactions.push_back(MemoryTransactionRecord(
                    image->pmem_gpgpusim + (rtnn_max_hits * index * sizeof(uint32_t)) + (i * 32),
                    32,
                    TransactionType::WRITE_TRAVERSAL_RESULT)
                ); 
            }
        }

        GPGPU_Context()->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::WRITE_TRAVERSAL_RESULT)]++;

        uint32_t width = image->vk.extent.width;
        uint32_t height = image->vk.extent.height;

        unsigned counter = 0;
        for (auto it=rtnn_sphere_hit_indices.begin(); it!=rtnn_sphere_hit_indices.end(); it++) {
            uint32_t sphere_index = *it;
            VulkanRayTracing::write_image_file(width, height, (float)sphere_index, 0, 0, index, counter, vk_format);
            counter++;
        }
        
        traversal_data.hit_geometry = true;
        traversal_data.closest_hit.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;

        unsigned node_count = 0;
        unsigned sphere_count = 0;
        for (auto it=transactions.begin(); it!=transactions.end(); it++) {
            if (it->type == TransactionType::BVH_STRUCTURE || it->type == TransactionType::BVH_INTERNAL_NODE || it->type == TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR) {
                node_count++;
            }
            else if (it->type == TransactionType::BVH_QUAD_LEAF) {
                sphere_count++;
            }
        }
        VSIM_DPRINTF("[%5d]: Ray hit %d nodes and %d spheres\n", thread->get_uid(), node_count, sphere_count);
    }
    else
    {
        VSIM_DPRINTF("gpgpusim: Ray [%d] missed.\n", thread->get_uid());
        traversal_data.hit_geometry = false;
        if (debugTraversal)
            traversalFile << "MISS" << std::endl;
    }

    Traversal_data* device_traversal_data = (Traversal_data*) VulkanRayTracing::gpgpusim_alloc(sizeof(Traversal_data));
    mem->write(device_traversal_data, sizeof(Traversal_data), &traversal_data, thread, pI);
    thread->RT_thread_data->traversal_data.push_back(device_traversal_data);
    
    thread->set_rt_transactions(transactions);
    thread->set_rt_store_transactions(store_transactions);

    if (debugTraversal)
    {
        traversalFile << "\n\n";
        traversalFile.close();
    }

    if (total_nodes_accessed > ctx->func_sim->g_max_nodes_per_ray) {
        ctx->func_sim->g_max_nodes_per_ray = total_nodes_accessed;
    }
    ctx->func_sim->g_tot_nodes_per_ray += total_nodes_accessed;

    if (terminateOnFirstHit)
        ctx->func_sim->g_tot_nodes_per_anyhit_ray += total_nodes_accessed;

    unsigned level = 0;
    for (auto it=tree_level_map.begin(); it!=tree_level_map.end(); it++) {
        if (it->second > level) {
            level = it->second;
        }
    }
    if (level > ctx->func_sim->g_max_tree_depth) {
        ctx->func_sim->g_max_tree_depth = level;
    }

    RT_DPRINTF("Traversal: \n");
    for (auto t : transactions) {
        RT_DPRINTF("\ttransaction %d, address %p, size %d\n", t.type, t.address, t.size);
    }

    if (rtnn) {
        set_op_sequence(5, 0, thread);
    }
    else if (raysphere) {
        set_op_sequence(6, 0, thread);
    }
    else {
        set_op_sequence(0, 0, thread);
    }
}

void VulkanRayTracing::endTraceRay(const ptx_instruction *pI, ptx_thread_info *thread)
{
    assert(thread->RT_thread_data->traversal_data.size() > 0);
    thread->RT_thread_data->traversal_data.pop_back();
    thread->RT_thread_data->all_hit_data.clear();
    warp_intersection_table* itable = intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
    itable->clear(pI, thread);
    warp_intersection_table* atable = anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
    atable->clear(pI, thread);
}

bool VulkanRayTracing::mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit)
{
    // Moller Trumbore algorithm (from scratchapixel.com)
    float3 v0v1 = p1 - p0;
    float3 v0v2 = p2 - p0;
    float3 pvec = cross(ray_properties.get_direction(), v0v2);
    float det = dot(v0v1, pvec);

    float idet = 1 / det;

#ifdef INJECT_RCP_ERROR
    float injection = (float)rand() / RAND_MAX * RCP_ERROR * 2 - RCP_ERROR + 1;
    assert(injection >= (1 - RCP_ERROR) && injection <= (1 + RCP_ERROR));
    idet = idet * injection;
#endif

    float3 tvec = ray_properties.get_origin() - p0;
    float u = dot(tvec, pvec) * idet;

    if (u < 0 || u > 1) return false;

    float3 qvec = cross(tvec, v0v1);
    float v = dot(ray_properties.get_direction(), qvec) * idet;

    if (v < 0 || (u + v) > 1) return false;

    *thit = dot(v0v2, qvec) * idet;
    return true;
}

bool VulkanRayTracing::raySphereIntersection(float4 sphere, Ray ray_properties, float* thit) {
    float3 origin = ray_properties.get_origin();
    float3 direction = ray_properties.get_direction();
    float tMin = ray_properties.get_tmin();
    float tMax = ray_properties.get_tmax();
    float3 sphere_origin = make_float3(sphere.x, sphere.y, sphere.z);
    float radius = sphere.w;

	// const vec3 oc = origin - center;
	// const float a = dot(direction, direction);
	// const float b = dot(oc, direction);
	// const float c = dot(oc, oc) - radius * radius;
	// const float discriminant = b * b - a * c;
    // bool hit = (discriminant >= 0);

    float3 oc = origin - sphere_origin;
    float a = dot(direction, direction);
    float b = dot(oc, direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant >= 0) {
        // const float t1 = (-b - sqrt(discriminant)) / a;
		// const float t2 = (-b + sqrt(discriminant)) / a;

        float t1 = (-1 * b - sqrt(discriminant)) / a;
        float t2 = (-1 * b + sqrt(discriminant)) / a;

		// if ((tMin <= t1 && t1 < tMax) || (tMin <= t2 && t2 < tMax))
		// {
		// 	Sphere = sphere;
		// 	reportIntersectionEXT((tMin <= t1 && t1 < tMax) ? t1 : t2, 0);
		// }

        if ((tMin <= t1 && t1 < tMax) || (tMin <= t2 && t2 < tMax)) {
            *thit = (tMin <= t1 && t1 < tMax) ? t1 : t2;
            return true;
        }
    }


    return false;
}

bool VulkanRayTracing::pointSphereIntersection(float4 sphere, Ray ray_properties) {
    float3 origin = ray_properties.get_origin();
    float3 sphere_origin = make_float3(sphere.x, sphere.y, sphere.z);
    float radius = sphere.w;

    // const vec3 oc = origin - center;
    // const float oc2 = dot(oc, oc);
    // const float r2 = radius * radius;
    // bool hit = (oc2 > 0 && oc2 < r2);

    float3 oc = origin - sphere_origin;
    float oc2 = dot(oc, oc);
    float r2 = radius * radius;
    bool hit = (oc2 > 0 && oc2 < r2);
    return hit;
}

bool VulkanRayTracing::rtao_ray_triangle_test(float4 v00, float4 v11, float4 v22, Ray ray_properties, float* thit, float3* bary) {

	float Oz = v00.w - ray_properties.get_origin().x * v00.x - ray_properties.get_origin().y * v00.y - ray_properties.get_origin().z * v00.z;
	float invDz = 1.0f / (ray_properties.get_direction().x*v00.x + ray_properties.get_direction().y*v00.y + ray_properties.get_direction().z*v00.z);
	float t = Oz * invDz;

	if (t > ray_properties.get_tmin() && t < *thit) {
		float Ox = v11.w + ray_properties.get_origin().x * v11.x + ray_properties.get_origin().y * v11.y + ray_properties.get_origin().z * v11.z;
		float Dx = ray_properties.get_direction().x * v11.x + ray_properties.get_direction().y * v11.y + ray_properties.get_direction().z * v11.z;
		float u = Ox + t * Dx;

		if (u >= 0.0f && u <= 1.0f) {
			float Oy = v22.w + ray_properties.get_origin().x * v22.x + ray_properties.get_origin().y * v22.y + ray_properties.get_origin().z * v22.z;
			float Dy = ray_properties.get_direction().x * v22.x + ray_properties.get_direction().y * v22.y + ray_properties.get_direction().z * v22.z;
			float v = Oy + t*Dy;

			if (v >= 0.0f && u + v <= 1.0f) {
				*thit = t;
                *bary = {u, v, 1-u-v};
                return true;
			}
		}
	}

    return false;
}

float3 VulkanRayTracing::Barycentric(float3 p, float3 a, float3 b, float3 c)
{
    //source: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    float3 v0 = b - a;
    float3 v1 = c - a;
    float3 v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return {v, w, u};
}

void VulkanRayTracing::load_descriptor(const ptx_instruction *pI, ptx_thread_info *thread)
{

}


void VulkanRayTracing::setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos)
{
    VulkanRayTracing::pCreateInfos = pCreateInfos;
    VSIM_DPRINTF("gpgpusim: set pipeline info\n");
}


void VulkanRayTracing::setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount)
{
    VulkanRayTracing::pGeometries = pGeometries;
    VulkanRayTracing::geometryCount = geometryCount;
    VSIM_DPRINTF("gpgpusim: set geometries\n");
}

void VulkanRayTracing::setAccelerationStructure(VkAccelerationStructureKHR accelerationStructure)
{
    GEN_RT_BVH topBVH; //TODO: test hit with world before traversal
    GEN_RT_BVH_unpack(&topBVH, (uint8_t *)accelerationStructure);
    VSIM_DPRINTF("gpgpusim: set AS %p\n", accelerationStructure);
    VulkanRayTracing::topLevelAS = accelerationStructure;
}

std::string base_name(std::string & path)
{
  return path.substr(path.find_last_of("/") + 1);
}

void VulkanRayTracing::setDescriptorSet(struct DESCRIPTOR_SET_STRUCT *set)
{
    if (VulkanRayTracing::descriptorSet == NULL) {
        VSIM_DPRINTF("gpgpusim: set descriptor set 0x%x\n", set);
        VulkanRayTracing::descriptorSet = set;
    }
    // TODO: Figure out why it sets the descriptor set twice
    else {
        VSIM_DPRINTF("gpgpusim: descriptor set already set; ignoring update.\n");
    }
}

static bool invoked = false;

void copyHardCodedShaders()
{
    std::ifstream  src;
    std::ofstream  dst;

    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_MISS_2.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_MISS_2.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();
    
    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_CLOSEST_HIT_2.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_CLOSEST_HIT_2.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();

    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_RAYGEN_0.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_RAYGEN_0.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();

    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_INTERSECTION_4.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_INTERSECTION_4.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();

    // {
    //     std::ifstream  src("/home/mrs/emerald-ray-tracing/MESA_SHADER_MISS_0.ptx", std::ios::binary);
    //     std::ofstream  dst("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_MISS_1.ptx",   std::ios::binary);
    //     dst << src.rdbuf();
    //     src.close();
    //     dst.close();
    // }
}

uint32_t VulkanRayTracing::registerShaders(char * shaderPath, gl_shader_stage shaderType)
{
    printf("gpgpusim: register shaders\n");
    copyHardCodedShaders();

    VulkanRayTracing::invoke_gpgpusim();
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    // Register all the ptx files in $MESA_ROOT/gpgpusimShaders by looping through them
    // std::vector <std::string> ptx_list;

    // Add ptx file names in gpgpusimShaders folder to ptx_list
    char *mesa_root = getenv("MESA_ROOT");
    char *gpgpusim_root = getenv("GPGPUSIM_ROOT");
    // char *filePath = "gpgpusimShaders/";
    // char fullPath[200];
    // snprintf(fullPath, sizeof(fullPath), "%s%s", mesa_root, filePath);
    // std::string fullPathString(fullPath);

    // for (auto &p : fs::recursive_directory_iterator(fullPathString))
    // {
    //     if (p.path().extension() == ".ptx")
    //     {
    //         //std::cout << p.path().string() << '\n';
    //         ptx_list.push_back(p.path().string());
    //     }
    // }

    std::string fullpath(shaderPath);
    std::string fullfilename = base_name(fullpath);
    std::string filenameNoExt;
    size_t start = fullfilename.find_first_not_of('.', 0);
    size_t end = fullfilename.find('.', start);
    filenameNoExt = fullfilename.substr(start, end - start);
    std::string idInString = filenameNoExt.substr(filenameNoExt.find_last_of("_") + 1);
    // Register each ptx file in ptx_list
    shader_stage_info shader;
    //shader.ID = VulkanRayTracing::shaders.size();
    shader.ID = std::stoi(idInString);
    shader.type = shaderType;
    shader.function_name = (char*)malloc(200 * sizeof(char));

    std::string deviceFunction;

    switch(shaderType) {
        case MESA_SHADER_RAYGEN:
            // shader.function_name = "raygen_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "raygen_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_RAYGEN";
            break;
        case MESA_SHADER_ANY_HIT:
            // shader.function_name = "anyhit_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "anyhit_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_ANY_HIT";
            break;
        case MESA_SHADER_CLOSEST_HIT:
            // shader.function_name = "closesthit_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "closesthit_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_CLOSEST_HIT";
            break;
        case MESA_SHADER_MISS:
            // shader.function_name = "miss_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "miss_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_MISS";
            break;
        case MESA_SHADER_INTERSECTION:
            // shader.function_name = "intersection_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "intersection_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_INTERSECTION";
            break;
        case MESA_SHADER_CALLABLE:
            // shader.function_name = "callable_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "callable_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "";
            assert(0);
            break;
    }
    deviceFunction += "_func" + std::to_string(shader.ID) + "_main";
    // deviceFunction += "_main";

    symbol_table *symtab;
    unsigned num_ptx_versions = 0;
    unsigned max_capability = 20;
    unsigned selected_capability = 20;
    bool found = false;
    
    unsigned long long fat_cubin_handle = shader.ID;

    // PTX File
    //std::cout << itr << std::endl;
    symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(shaderPath);
    context->add_binary(symtab, fat_cubin_handle);
    // need to add all the magic registers to ptx.l to special_register, reference ayub ptx.l:225

    // PTX info
    // Run the python script and get ptxinfo
    std::cout << "GPGPUSIM: Generating PTXINFO for" << shaderPath << "info" << std::endl;
    char command[400];
    snprintf(command, sizeof(command), "python3 %s/scripts/generate_rt_ptxinfo.py %s", gpgpusim_root, shaderPath);
    int result = system(command);
    if (result != 0) {
        printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
        printf("               Ensure ptxas is in your path.\n");
        exit(1);
    }
    
    char ptxinfo_filename[400];
    snprintf(ptxinfo_filename, sizeof(ptxinfo_filename), "%sinfo", shaderPath);
    ctx->gpgpu_ptx_info_load_from_external_file(ptxinfo_filename); // TODO: make a version where it just loads my ptxinfo instead of generating a new one

    context->register_function(fat_cubin_handle, shader.function_name, deviceFunction.c_str());

    VulkanRayTracing::shaders.push_back(shader);

    return shader.ID;

    // if (itr.find("RAYGEN") != std::string::npos)
    // {
    //     printf("############### registering %s\n", shaderPath);
    //     context->register_function(fat_cubin_handle, "raygen_shader", "MESA_SHADER_RAYGEN_main");
    // }

    // if (itr.find("MISS") != std::string::npos)
    // {
    //     printf("############### registering %s\n", shaderPath);
    //     context->register_function(fat_cubin_handle, "miss_shader", "MESA_SHADER_MISS_main");
    // }

    // if (itr.find("CLOSEST") != std::string::npos)
    // {
    //     printf("############### registering %s\n", shaderPath);
    //     context->register_function(fat_cubin_handle, "closest_hit_shader", "MESA_SHADER_CLOSEST_HIT_main");
    // }
}


void VulkanRayTracing::invoke_gpgpusim()
{
    printf("gpgpusim: invoking gpgpusim\n");
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    if(!invoked)
    {
        //registerShaders();
        invoked = true;
    }
}

// int CmdTraceRaysKHRID = 0;

const bool writeImageBinary = true;

void VulkanRayTracing::vkCmdTraceRaysKHR(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr) {
    printf("gpgpusim: launching cmd trace ray\n");
    srand(0);
    // launch_width = 224;
    // launch_height = 160;
    init(launch_width, launch_height);
    
    // Dump Descriptor Sets
    if (dump_trace) 
    {
        dump_descriptor_sets(VulkanRayTracing::descriptorSet);
        dump_callparams_and_sbt(raygen_sbt, miss_sbt, hit_sbt, callable_sbt, is_indirect, launch_width, launch_height, launch_depth, launch_size_addr);
    }

    // CmdTraceRaysKHRID++;
    // if(CmdTraceRaysKHRID != 1)
    //     return;
    // launch_width = 420;
    // launch_height = 320;

    if(writeImageBinary && !imageFile.is_open())
    {
        char* imageFileName;
        char defaultFileName[40] = "image.binary";
        if(getenv("VULKAN_IMAGE_FILE_NAME"))
            imageFileName = getenv("VULKAN_IMAGE_FILE_NAME");
        else
            imageFileName = defaultFileName;
        imageFile.open(imageFileName, std::ios::out | std::ios::binary);
        
        // imageFile.open("image.txt", std::ios::out);
    }
    // memset(((uint8_t*)descriptors[0][1].address), uint8_t(127), launch_height * launch_width * 4);
    // return;

    // {
    //     std::ifstream infile("debug_printf.log");
    //     std::string line;
    //     while (std::getline(infile, line))
    //     {
    //         if(line == "")
    //             continue;

    //         RayDebugGPUData data;
    //         // sscanf(line.c_str(), "LaunchID:(%d,%d), InstanceCustomIndex = %d, primitiveID = %d, v0 = (%f, %f, %f), v1 = (%f, %f, %f), v2 = (%f, %f, %f), hitAttribute = (%f, %f), normalWorld = (%f, %f, %f), objectIntersection = (%f, %f, %f), worldIntersection = (%f, %f, %f), objectNormal = (%f, %f, %f), worldNormal = (%f, %f, %f), NdotL = %f",
    //         //             &data.launchIDx, &data.launchIDy, &data.instanceCustomIndex, &data.primitiveID, &data.v0pos.x, &data.v0pos.y, &data.v0pos.z, &data.v1pos.x, &data.v1pos.y, &data.v1pos.z, &data.v2pos.x, &data.v2pos.y, &data.v2pos.z, &data.attribs.x, &data.attribs.y, &data.N.x, &data.N.y, &data.N.z, &data.P_object.x, &data.P_object.y, &data.P_object.z, &data.P.x, &data.P.y, &data.P.z, &data.N_object.x, &data.N_object.y, &data.N_object.z, &data.N.x, &data.N.y, &data.N.z, &data.NdotL);
    //         sscanf(line.c_str(), "launchID = (%d, %d), hitValue = (%f, %f, %f)",
    //                     &data.launchIDx, &data.launchIDy, &data.hitValue.x, &data.hitValue.y, &data.hitValue.z);
    //         data.valid = true;
    //         assert(data.launchIDx < 2000 && data.launchIDy < 2000);
    //         // printf("#### (%d, %d)\n", data.launchIDx, data.launchIDy);
    //         // fflush(stdout);
    //         rayDebugGPUData[data.launchIDx][data.launchIDy] = data;

    //     }
    // }

    assert(launch_depth == 1);

#if defined(MESA_USE_INTEL_DRIVER)
    struct DESCRIPTOR_STRUCT desc;
    desc.image_view = NULL;
#endif

    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    unsigned long shaderId = *(uint64_t*)raygen_sbt;
    int index = 0;
    for (int i = 0; i < shaders.size(); i++) {
        if (shaders[i].ID == 0){
            index = i;
            break;
        }
    }
    ctx->func_sim->g_total_shaders = shaders.size();

    shader_stage_info raygen_shader = shaders[index];
    function_info *entry = context->get_kernel(raygen_shader.function_name);
    // printf("################ number of args = %d\n", entry->num_args());

    if (entry->is_pdom_set()) {
        printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
            entry->get_name().c_str());
    } else {
        printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
            entry->get_name().c_str());
        /*
        * Some of the instructions like printf() gives the gpgpusim the wrong
        * impression that it is a function call. As printf() doesnt have a body
        * like functions do, doing pdom analysis for printf() causes a crash.
        */
        if (entry->get_function_size() > 0) entry->do_pdom();
        entry->set_pdom();
    }

    // check that number of args and return match function requirements
    //if (pI->has_return() ^ entry->has_return()) {
    //    printf(
    //        "GPGPU-Sim PTX: Execution error - mismatch in number of return values "
    //        "between\n"
    //        "               call instruction and function declaration\n");
    //    abort();
    //}
    unsigned n_return = entry->has_return();
    unsigned n_args = entry->num_args();
    //unsigned n_operands = pI->get_num_operands();

    // launch_width = 192;
    // launch_height = 32;

    dim3 blockDim = dim3(1, 1, 1);
    dim3 gridDim = dim3(1, launch_height, launch_depth);
    if(launch_width <= 32) {
        blockDim.x = launch_width;
        gridDim.x = 1;
    }
    else {
        blockDim.x = 32;
        gridDim.x = launch_width / 32;
        if(launch_width % 32 != 0)
            gridDim.x++;
    }
    printf("gpgpusim: launch dimensions %d x %d x %d\n", gridDim.x, gridDim.y, gridDim.z);
    printf("gpgpusim: traversing with key %d\n", ctx->func_sim->g_rt_traversal_key);

    gpgpu_ptx_sim_arg_list_t args;
    // kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
    //   raygen_shader.function_name, args, dim3(4, 128, 1), dim3(32, 1, 1), context);
    kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
      raygen_shader.function_name, args, gridDim, blockDim, context);
    grid->vulkan_metadata.raygen_sbt = raygen_sbt;
    grid->vulkan_metadata.miss_sbt = miss_sbt;
    grid->vulkan_metadata.hit_sbt = hit_sbt;
    grid->vulkan_metadata.callable_sbt = callable_sbt;
    grid->vulkan_metadata.launch_width = launch_width;
    grid->vulkan_metadata.launch_height = launch_height;
    grid->vulkan_metadata.launch_depth = launch_depth;
    
    printf("gpgpusim: SBT: raygen %p, miss %p, hit %p, callable %p\n", 
            raygen_sbt, miss_sbt, hit_sbt, callable_sbt);

    VSIM_DPRINTF("gpgpusim: blas address\n");
    for (auto mapping : blas_addr_map) {
        VSIM_DPRINTF("\t[%p] -> %p\n", mapping.first, mapping.second);
    }

    printf("gpgpusim: tlas address %p\n", tlas_addr);
            
    struct CUstream_st *stream = 0;
    stream_operation op(grid, ctx->func_sim->g_ptx_sim_mode, stream);
    ctx->the_gpgpusim->g_stream_manager->push(op);

    //printf("%d\n", descriptors[0][1].address);

    fflush(stdout);

    while(!op.is_done() && !op.get_kernel()->done()) {
        printf("waiting for op to finish\n");
        sleep(1);
        continue;
    }
    // for (unsigned i = 0; i < entry->num_args(); i++) {
    //     std::pair<size_t, unsigned> p = entry->get_param_config(i);
    //     cudaSetupArgumentInternal(args[i], p.first, p.second);
    // }
}

void VulkanRayTracing::callMissShader(const ptx_instruction *pI, ptx_thread_info *thread) {
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

    bool hit_geometry;
    mem->read(&(traversal_data->hit_geometry), sizeof(bool), &hit_geometry);
    assert(!hit_geometry);

    int32_t current_shader_counter = -1;
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &current_shader_counter, thread, pI);

    int32_t current_shader_type = -1;
    mem->write(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &current_shader_type, thread, pI);

    uint32_t missIndex;
    mem->read(&(traversal_data->missIndex), sizeof(traversal_data->missIndex), &missIndex);

    uint32_t shaderID = *((uint32_t *)(thread->get_kernel().vulkan_metadata.miss_sbt) + 8 * missIndex);
    VSIM_DPRINTF("gpgpusim: Calling Miss Shader at ID %d\n", shaderID);

    shader_stage_info miss_shader = shaders[shaderID];

    function_info *entry = context->get_kernel(miss_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callClosestHitShader(const ptx_instruction *pI, ptx_thread_info *thread) {
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

    bool hit_geometry;
    mem->read(&(traversal_data->hit_geometry), sizeof(bool), &hit_geometry);
    assert(hit_geometry);

    int32_t current_shader_counter = -1;
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &current_shader_counter, thread, pI);

    int32_t current_shader_type = -1;
    mem->write(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &current_shader_type, thread, pI);

    VkGeometryTypeKHR geometryType;
    mem->read(&(traversal_data->closest_hit.geometryType), sizeof(traversal_data->closest_hit.geometryType), &geometryType);

    shader_stage_info closesthit_shader;
    if(geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
        uint32_t shaderID = *((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt));
        closesthit_shader = shaders[shaderID];
        VSIM_DPRINTF("gpgpusim: Calling Closest Hit Shader at ID %d\n", shaderID);

    }
    else {
        int32_t hitGroupIndex;
        mem->read(&(traversal_data->closest_hit.hitGroupIndex), sizeof(traversal_data->closest_hit.hitGroupIndex), &hitGroupIndex);
        uint32_t shaderID = *((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex);
        closesthit_shader = shaders[shaderID];
        VSIM_DPRINTF("gpgpusim: Calling Closest Hit Shader at ID %d\n", shaderID);
    }

    function_info *entry = context->get_kernel(closesthit_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callIntersectionShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter) {
    VSIM_DPRINTF("gpgpusim: Calling Intersection Shader\n");
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    
    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &shader_counter, thread, pI);

    int32_t current_shader_type = 1;
    mem->write(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &current_shader_type, thread, pI);

    warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
    uint32_t hitGroupIndex = table->get_hitGroupIndex(shader_counter, thread->get_tid().x, pI, thread);

    shader_stage_info intersection_shader = shaders[*((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex + 1)];
    function_info *entry = context->get_kernel(intersection_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callAnyHitShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter) {
    VSIM_DPRINTF("gpgpusim: Calling Any Hit Shader\n");
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &shader_counter, thread, pI);

    int32_t current_shader_type = 2;
    mem->write(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &current_shader_type, thread, pI);

    warp_intersection_table* table = VulkanRayTracing::anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
    uint32_t hitGroupIndex = table->get_hitGroupIndex(shader_counter, thread->get_tid().x, pI, thread);

    shader_stage_info anyhit_shader = shaders[*((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex + 1)];
    function_info *entry = context->get_kernel(anyhit_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callShader(const ptx_instruction *pI, ptx_thread_info *thread, function_info *target_func) {
    static unsigned call_uid_next = 1;

  if (target_func->is_pdom_set()) {
    // printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
    //        target_func->get_name().c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           target_func->get_name().c_str());
    /*
     * Some of the instructions like printf() gives the gpgpusim the wrong
     * impression that it is a function call. As printf() doesnt have a body
     * like functions do, doing pdom analysis for printf() causes a crash.
     */
    if (target_func->get_function_size() > 0) target_func->do_pdom();
    target_func->set_pdom();
  }

  thread->set_npc(target_func->get_start_PC());

  // check that number of args and return match function requirements
  if (pI->has_return() ^ target_func->has_return()) {
    printf(
        "GPGPU-Sim PTX: Execution error - mismatch in number of return values "
        "between\n"
        "               call instruction and function declaration\n");
    abort();
  }
  unsigned n_return = target_func->has_return();
  unsigned n_args = target_func->num_args();
  unsigned n_operands = pI->get_num_operands();

  // TODO: why this fails?
//   if (n_operands != (n_return + 1 + n_args)) {
//     printf(
//         "GPGPU-Sim PTX: Execution error - mismatch in number of arguements "
//         "between\n"
//         "               call instruction and function declaration\n");
//     abort();
//   }

  // handle intrinsic functions
//   std::string fname = target_func->get_name();
//   if (fname == "vprintf") {
//     gpgpusim_cuda_vprintf(pI, thread, target_func);
//     return;
//   }
// #if (CUDART_VERSION >= 5000)
//   // Jin: handle device runtime apis for CDP
//   else if (fname == "cudaGetParameterBufferV2") {
//     target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_getParameterBufferV2(
//         pI, thread, target_func);
//     return;
//   } else if (fname == "cudaLaunchDeviceV2") {
//     target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_launchDeviceV2(
//         pI, thread, target_func);
//     return;
//   } else if (fname == "cudaStreamCreateWithFlags") {
//     target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_streamCreateWithFlags(
//         pI, thread, target_func);
//     return;
//   }
// #endif

  // read source arguements into register specified in declaration of function
  arg_buffer_list_t arg_values;
  copy_args_into_buffer_list(pI, thread, target_func, arg_values);

  // record local for return value (we only support a single return value)
  const symbol *return_var_src = NULL;
  const symbol *return_var_dst = NULL;
  if (target_func->has_return()) {
    return_var_dst = pI->dst().get_symbol();
    return_var_src = target_func->get_return_var();
  }

  gpgpu_sim *gpu = thread->get_gpu();
  unsigned callee_pc = 0, callee_rpc = 0;
  /*if (gpu->simd_model() == POST_DOMINATOR)*/ { //MRS_TODO: why this fails?
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc);
    assert(callee_pc == thread->get_pc());
  }

  thread->callstack_push(callee_pc + pI->inst_size(), callee_rpc,
                         return_var_src, return_var_dst, call_uid_next++);

  copy_buffer_list_into_frame(thread, arg_values);

  thread->set_npc(target_func);
}

void VulkanRayTracing::setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type)
{
    printf("gpgpusim: set descriptor\n");
    if(descriptors.size() <= setID)
        descriptors.resize(setID + 1);
    if(descriptors[setID].size() <= descID)
        descriptors[setID].resize(descID + 1);
    
    descriptors[setID][descID].setID = setID;
    descriptors[setID][descID].descID = descID;
    descriptors[setID][descID].address = address;
    descriptors[setID][descID].size = size;
    descriptors[setID][descID].type = type;
}


void VulkanRayTracing::setDescriptorSetFromLauncher(void *address, void *deviceAddress, uint32_t setID, uint32_t descID)
{
    launcher_deviceDescriptorSets[setID][descID] = deviceAddress;
    launcher_descriptorSets[setID][descID] = address;
}

void* VulkanRayTracing::getDescriptorAddress(uint32_t setID, uint32_t binding)
{
#if defined(MESA_USE_INTEL_DRIVER)
    if (GPGPU_Context()->func_sim->g_rt_external_launch)
    {
        return launcher_deviceDescriptorSets[setID][binding];
        // return launcher_descriptorSets[setID][binding];
    }
    else 
    {
        // assert(setID < descriptors.size());
        // assert(binding < descriptors[setID].size());

        struct anv_descriptor_set* set = VulkanRayTracing::descriptorSet;

        const struct anv_descriptor_set_binding_layout *bind_layout = &set->layout->binding[binding];
        struct anv_descriptor *desc = &set->descriptors[bind_layout->descriptor_index];
        void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

        assert(desc->type == bind_layout->type);

        switch (desc->type)
        {
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            {
                return (void *)(desc);
            }
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            {
                return desc;
            }

            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            {
                if (desc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                    desc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                {
                    // MRS_TODO: account for desc->offset?
                    return anv_address_map(desc->buffer->address);
                }
                else
                {
                    struct anv_buffer_view *bview = &set->buffer_views[bind_layout->buffer_view_index];
                    return anv_address_map(bview->address);
                }
            }

            case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            {
                struct anv_address_range_descriptor *desc_data = desc_map;
                return (void *)(desc_data->address);
            }

            default:
                assert(0);
                break;
        }

        // return descriptors[setID][binding].address;
    }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    VSIM_DPRINTF("gpgpusim: getDescriptorAddress for binding %d\n", binding);
    struct lvp_descriptor_set* set = VulkanRayTracing::descriptorSet;
    const struct lvp_descriptor_set_binding_layout *bind_layout = &set->layout->binding[binding];
    struct lvp_descriptor *desc = &set->descriptors[bind_layout->descriptor_index];

    // printf("DESCRIPTOR TYPE: %d\n", desc->type);
    switch (desc->type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            VSIM_DPRINTF("gpgpusim: storage image; descriptor address %p\n", desc);
            return (void *) desc;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            VSIM_DPRINTF("gpgpusim: uniform buffer; buffer mem address %p\n", (void *) desc->info.ubo.pmem);
            return (void *) desc->info.ubo.pmem;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            VSIM_DPRINTF("gpgpusim: storage buffer; buffer mem address %p\n", (void *) desc->info.ssbo.pmem);
            return (void *) desc->info.ssbo.pmem;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            VSIM_DPRINTF("gpgpusim: accel struct; root address %p\n", (void *)desc->info.ubo.pmem + desc->info.ubo.buffer_offset);
            return (void *)desc->info.ubo.pmem + desc->info.ubo.buffer_offset;
            break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            VSIM_DPRINTF("gpgpusim: image sampler; descriptor address %p\n", desc);
            return (void *) desc;
            break;
        default:
            VSIM_DPRINTF("gpgpusim: unimplemented descriptor type\n");
            abort();
    }
#endif
}

void VulkanRayTracing::getTexture(struct DESCRIPTOR_STRUCT *desc, 
                                    float x, float y, float lod, 
                                    float &c0, float &c1, float &c2, float &c3, 
                                    std::vector<ImageMemoryTransactionRecord>& transactions,
                                    uint64_t launcher_offset)
{
#if defined(MESA_USE_INTEL_DRIVER)
    Pixel pixel;

    if (GPGPU_Context()->func_sim->g_rt_external_launch)
    {
        pixel = get_interpolated_pixel((anv_image_view*) desc, (anv_sampler*) desc, x, y, transactions, launcher_offset); // cast back to metadata later
    }
    else 
    {
        struct anv_image_view *image_view =  desc->image_view;
        struct anv_sampler *sampler = desc->sampler;

        const struct anv_image *image = image_view->image;
        assert(image->n_planes == 1);
        assert(image->samples == 1);
        assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
        assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
        assert(sampler->conversion == NULL);

        pixel = get_interpolated_pixel(image_view, sampler, x, y, transactions);
    }

    TXL_DPRINTF("Setting transaction type to TEXTURE_LOAD\n");
    for(int i = 0; i < transactions.size(); i++)
        transactions[i].type = ImageTransactionType::TEXTURE_LOAD;
    
    c0 = pixel.c0;
    c1 = pixel.c1;
    c2 = pixel.c2;
    c3 = pixel.c3;


    // uint8_t* address = anv_address_map(image->planes[0].address);

    // for(int x = 0; x < image->extent.width; x++)
    // {
    //     for(int y = 0; y < image->extent.height; y++)
    //     {
    //         int blockX = x / 8;
    //         int blockY = y / 8;

    //         uint32_t offset = (blockX + blockY * (image->extent.width / 8)) * (128 / 8);

    //         uint8_t dst_colors[100];
    //         basisu::astc::decompress(dst_colors, address + offset, true, 8, 8);
    //         uint8_t* pixel_color = &dst_colors[0] + (x % 8 + (y % 8) * 8) * 4;

    //         uint32_t bit_map_offset = x + y * image->extent.width;

    //         float data[4];
    //         data[0] = pixel_color[0] / 255.0;
    //         data[1] = pixel_color[1] / 255.0;
    //         data[2] = pixel_color[2] / 255.0;
    //         data[3] = pixel_color[3] / 255.0;
    //         imageFile.write((char*) data, 3 * sizeof(float));
    //         imageFile.write((char*) (&bit_map_offset), sizeof(uint32_t));
    //         imageFile.flush();
    //     }
    // }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    // printf("gpgpusim: getTexture not implemented for lavapipe.\n");
    //
    // printf("GIVEN DESC: %p\n", desc);

    if (x < 0 || x > 1)
        x -= std::floor(x);
    if (y < 0 || y > 1)
        y -= std::floor(y);

    // printf("X: %f, Y: %f\n", x, y);

    struct lvp_descriptor d = *(struct lvp_descriptor*) desc;
    const struct lvp_image *img = d.info.sampler_view->image;
    uint32_t width = img->vk.extent.width;
    uint32_t height = img->vk.extent.height;
    void *i = img->pmem;

    uint32_t x_int = std::floor(x * width);
    uint32_t y_int = std::floor(y * height);
    if(x_int >= width)
        x_int -= width;
    if(y_int >= height)
        y_int -= height;

    void *c = i + (y_int * height + x_int) * 4;

    ImageMemoryTransactionRecord transaction;
    transaction.type = ImageTransactionType::TEXTURE_LOAD;
    transaction.address = c;
    transaction.size = 4;
    transactions.push_back(transaction);

    uint8_t *colors = (uint8_t*) c;
    c0 = colors[0] / 255.0;
    c1 = colors[1] / 255.0;
    c2 = colors[2] / 255.0;
    c3 = colors[3] / 255.0;

    // abort();
#endif
}

#if defined(MESA_USE_LVPIPE_DRIVER)
FILE *img_bin = nullptr;
#endif

void VulkanRayTracing::image_load(struct DESCRIPTOR_STRUCT *desc, uint32_t x, uint32_t y, float &c0, float &c1, float &c2, float &c3)
{
#if defined(MESA_USE_INTEL_DRIVER)
    ImageMemoryTransactionRecord transaction;

    struct anv_image_view *image_view =  desc->image_view;
    struct anv_sampler *sampler = desc->sampler;

    const struct anv_image *image = image_view->image;
    assert(image->n_planes == 1);
    assert(image->samples == 1);
    assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
    assert(sampler->conversion == NULL);

    Pixel pixel = load_image_pixel(image, x, y, 0, transaction);

    transaction.type = ImageTransactionType::IMAGE_LOAD;
    
    c0 = pixel.c0;
    c1 = pixel.c1;
    c2 = pixel.c2;
    c3 = pixel.c3;

#elif defined(MESA_USE_LVPIPE_DRIVER)
    VSIM_DPRINTF("gpgpusim: image_load not implemented for lavapipe.\n");
    abort();

#endif
}

void VulkanRayTracing::write_image_file(uint32_t width, uint32_t height, float hitValue_X, float hitValue_Y, float hitValue_Z, uint32_t pixelX, uint32_t pixelY, VkFormat img_format) {
    std::string img_name("SCENE");

    if (outputImages.find(img_name) == outputImages.end()) {
        std::time_t raw_time = std::time(0);
        struct tm *time_info;
        char time_buf[30];

        time_info = localtime(&raw_time);

        strftime(time_buf, sizeof(time_buf), "%d-%m-%Y-%H-%M-%S-", time_info);

        std::string time_offset(time_buf);
        std::string new_img_file_name = time_offset + img_name;

        outputImages[img_name] = new_img_file_name + ".ppm";
        printf("gpgpusim: saving image %s to file %s\n", img_name.c_str(), outputImages[img_name].c_str());

        img_bin = fopen(outputImages[img_name].c_str(), "w");
        fprintf(img_bin, "P3\n%d %d\n255\n", width, height);
        for (int i = 0; i < width * height; i++) {
            fprintf(img_bin, "%3d %3d %3d\n", 0, 0, 0);
        }
    }

    uint32_t header_offset = 
        strlen("P3\n \n255\n") + std::to_string(width).length() + std::to_string(height).length();
    uint32_t value_offset = (pixelX + pixelY * width) * (3*3 + 3);
    fseeko(img_bin, header_offset + value_offset, SEEK_SET);

    // Use VK_FORMAT_R8G8B8A8_UINT to trigger RTNN workload special case
    if (img_format == VK_FORMAT_R8G8B8A8_UINT) {
        fprintf(img_bin, "%7d %1d %1d\n", 
                (int)(hitValue_X), (int)(hitValue_Y), (int)(hitValue_Z));
        // printf("gpgpusim: [%d %d] -> [%d %d %d]\n", pixelX, pixelY, (int)(hitValue_X), (int)(hitValue_Y), (int)(hitValue_Z));
    } else {
        fprintf(img_bin, "%3.0f %3.0f %3.0f\n", 
            hitValue_X * 255, hitValue_Y * 255, hitValue_Z * 255);
    }
}

void VulkanRayTracing::image_store(struct DESCRIPTOR_STRUCT* desc, uint32_t gl_LaunchIDEXT_X, uint32_t gl_LaunchIDEXT_Y, uint32_t gl_LaunchIDEXT_Z, uint32_t gl_LaunchIDEXT_W, 
              float hitValue_X, float hitValue_Y, float hitValue_Z, float hitValue_W, const ptx_instruction *pI, ptx_thread_info *thread)
{
#if defined(MESA_USE_INTEL_DRIVER)
    ImageMemoryTransactionRecord transaction;
    Pixel pixel = Pixel(hitValue_X, hitValue_Y, hitValue_Z, hitValue_W);

    VkFormat vk_format;
    if (GPGPU_Context()->func_sim->g_rt_external_launch)
    {
        storage_image_metadata *metadata = (storage_image_metadata*) desc;
        vk_format = metadata->format;
        store_image_pixel((anv_image*) desc, gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, 0, pixel, transaction);
    }
    else
    {
        assert(desc->sampler == NULL);

        struct anv_image_view *image_view = desc->image_view;
        assert(image_view != NULL);
        struct anv_image * image = image_view->image;

        vk_format = image->vk_format;

        store_image_pixel(image, gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, 0, pixel, transaction);
    }

    
    transaction.type = ImageTransactionType::IMAGE_STORE;

    if(writeImageBinary && vk_format != VK_FORMAT_R32G32B32A32_SFLOAT)
    {
        uint32_t image_width = thread->get_kernel().vulkan_metadata.launch_width;
        uint32_t offset = 0;
        offset += gl_LaunchIDEXT_Y * image_width;
        offset += gl_LaunchIDEXT_X;

        float data[4];
        data[0] = hitValue_X;
        data[1] = hitValue_Y;
        data[2] = hitValue_Z;
        data[3] = hitValue_W;
        imageFile.write((char*) data, 3 * sizeof(float));
        imageFile.write((char*) (&offset), sizeof(uint32_t));
        imageFile.flush();

        // imageFile << "(" << gl_LaunchIDEXT_X << ", " << gl_LaunchIDEXT_Y << ") : (";
        // imageFile << hitValue_X << ", " << hitValue_Y << ", " << hitValue_Z << ", " << hitValue_W << ")\n";
    }

    TXL_DPRINTF("Setting transaction for image_store\n");
    thread->set_txl_transactions(transaction);

    // // if(std::abs(hitValue_X - rayDebugGPUData[gl_LaunchIDEXT_X][gl_LaunchIDEXT_Y].hitValue.x) > 0.0001 || 
    // //     std::abs(hitValue_Y - rayDebugGPUData[gl_LaunchIDEXT_X][gl_LaunchIDEXT_Y].hitValue.y) > 0.0001 ||
    // //     std::abs(hitValue_Z - rayDebugGPUData[gl_LaunchIDEXT_X][gl_LaunchIDEXT_Y].hitValue.z) > 0.0001)
    // //     {
    // //         printf("wrong value. (%d, %d): (%f, %f, %f)\n"
    // //                 , gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, hitValue_X, hitValue_Y, hitValue_Z);
    // //     }
    
    // // if (gl_LaunchIDEXT_X == 1070 && gl_LaunchIDEXT_Y == 220)
    // //     printf("this one has wrong value\n");

    // // if(hitValue_X > 1 || hitValue_Y > 1 || hitValue_Z > 1)
    // // {
    // //     printf("this one has wrong value.\n");
    // // }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    assert(desc->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    struct lvp_image *image = (struct lvp_image *)desc->info.image_view.image;
    VkFormat vk_format = image->vk.format;
    assert(image != NULL);
    VSIM_DPRINTF("gpgpusim: image_store to %s at %p, type %d\n", image->vk.base.object_name, image->pmem_gpgpusim, vk_format);

    Pixel pixel = Pixel(hitValue_X, hitValue_Y, hitValue_Z, hitValue_W);

    uint32_t width = image->vk.extent.width;
    uint32_t height = image->vk.extent.height;

    uint32_t pixelX = gl_LaunchIDEXT_X;
    uint32_t pixelY = gl_LaunchIDEXT_Y;

    if (writeImageBinary) {
        // TODO: fix the bottom, is NULL
        // assert(image->vk.base.object_name);
        // std::string img_name(image->vk.base.object_name);
        VulkanRayTracing::write_image_file(width, height, hitValue_X, hitValue_Y, hitValue_Z, pixelX, pixelY, vk_format);
    }

    // Setup transaction record for timing model
    ImageMemoryTransactionRecord transaction;
    transaction.type = ImageTransactionType::IMAGE_STORE;

    VkImageTiling tiling = image->vk.tiling;

    // Size of image_store content depends on data type
    switch (vk_format) {
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            transaction.size = 16;
            break; 

        case VK_FORMAT_B8G8R8A8_UNORM:
            transaction.size = 4;
            break;

        case VK_FORMAT_R8G8B8A8_UINT:
            transaction.size = 4;
            break;

        default:
            printf("gpgpusim: unsupported image format option %d\n", vk_format);
            abort();
    }

    switch (tiling) {
        // Just an arbitrary tiling (TODO: Find a better tiling option)
        case VK_IMAGE_TILING_OPTIMAL:
        {
            uint32_t tileWidth = 16;
            uint32_t tileHeight = 16;

            uint32_t nTileX = ceil(width / tileWidth);
            uint32_t tileX = floor(pixelX / tileWidth);
            uint32_t tileY = floor(pixelY / tileHeight);

            uint32_t tileOffset = tileWidth * tileHeight * (tileY * nTileX + tileX);
            uint32_t pixelOffset = (pixelY % tileHeight) * tileWidth + (pixelX % tileWidth);

            transaction.address = image->pmem_gpgpusim + ((tileOffset + pixelOffset) * transaction.size);
            break;
        }
        // Linear
        case VK_IMAGE_TILING_LINEAR:
        {
            uint32_t offset = pixelY * width + pixelX;
            transaction.address = image->pmem_gpgpusim + offset * transaction.size;
            break;
        }
        default:
        {
            printf("gpgpusim: unsupported image tiling option %d\n", tiling);
            abort();
        }
    }

    TXL_DPRINTF("Setting transaction for image_store\n");
    thread->set_txl_transactions(transaction);

    // store_image_pixel(image, gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, 0, pixel, transaction);
#endif
}

// variable_decleration_entry* VulkanRayTracing::get_variable_decleration_entry(std::string name, ptx_thread_info *thread)
// {
//     std::vector<variable_decleration_entry>& table = thread->RT_thread_data->variable_decleration_table;
//     for (int i = 0; i < table.size(); i++) {
//         if (table[i].name == name) {
//             assert (table[i].address != NULL);
//             return &(table[i]);
//         }
//     }
//     return NULL;
// }

// void VulkanRayTracing::add_variable_decleration_entry(uint64_t type, std::string name, uint64_t address, uint32_t size, ptx_thread_info *thread)
// {
//     variable_decleration_entry entry;

//     entry.type = type;
//     entry.name = name;
//     entry.address = address;
//     entry.size = size;
//     thread->RT_thread_data->variable_decleration_table.push_back(entry);
// }


void VulkanRayTracing::dumpTextures(struct DESCRIPTOR_STRUCT *desc, uint32_t setID, uint32_t binding, VkDescriptorType type)
{
#if defined(MESA_USE_INTEL_DRIVER)
    DESCRIPTOR_STRUCT *desc_offset = ((DESCRIPTOR_STRUCT*)((void*)desc)); // offset for raytracing_extended
    struct anv_image_view *image_view =  desc_offset->image_view;
    struct anv_sampler *sampler = desc_offset->sampler;

    const struct anv_image *image = image_view->image;
    assert(image->n_planes == 1);
    assert(image->samples == 1);
    assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
    assert(sampler->conversion == NULL);

    uint8_t* address = anv_address_map(image->planes[0].address);
    uint32_t image_extent_width = image->extent.width;
    uint32_t image_extent_height = image->extent.height;
    VkFormat format = image->vk_format;
    uint64_t size = image->size;

    VkFilter filter;
    if(sampler->conversion == NULL)
        filter = VK_FILTER_NEAREST;

    // Data to dump
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsettexturedata";

    int VkDescriptorTypeNum;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            VkDescriptorTypeNum = 0;
            break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            VkDescriptorTypeNum = 1;
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            VkDescriptorTypeNum = 2;
            break;
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            VkDescriptorTypeNum = 10;
            break;
        default:
            abort(); // should not be here!
    }

    // Texture data
    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.vktexturedata", mesa_root, filePath, setID, binding);
    // File name format: setID_descID.vktexturedata

    fp = fopen(fullPath, "wb+");
    fwrite(address, 1, size, fp);
    fclose(fp);

    // Texture metadata
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.vktexturemetadata", mesa_root, filePath, setID, binding);
    fp = fopen(fullPath, "w+");
    // File name format: setID_descID.vktexturemetadata

    fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", size, 
                                                 image_extent_width, 
                                                 image_extent_height, 
                                                 format, 
                                                 VkDescriptorTypeNum, 
                                                 image->n_planes, 
                                                 image->samples, 
                                                 image->tiling, 
                                                 image->planes[0].surface.isl.tiling,
                                                 image->planes[0].surface.isl.row_pitch_B,
                                                 filter);
    fclose(fp);
#elif defined(MESA_USE_LVPIPE_DRIVER)
    printf("gpgpusim: dumpTextures not implemented for lavapipe.\n");
    abort();

#endif

}


void VulkanRayTracing::dumpStorageImage(struct DESCRIPTOR_STRUCT *desc, uint32_t setID, uint32_t binding, VkDescriptorType type)
{
#if defined(MESA_USE_INTEL_DRIVER)
    assert(type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    assert(desc->sampler == NULL);

    struct anv_image_view *image_view = desc->image_view;
    assert(image_view != NULL);
    struct anv_image * image = image_view->image;
    assert(image->n_planes == 1);
    assert(image->samples == 1);

    void* mem_address = anv_address_map(image->planes[0].address);

    VkFormat format = image->vk_format;
    VkImageTiling tiling = image->tiling;
    isl_tiling isl_tiling_mode = image->planes[0].surface.isl.tiling;
    uint32_t row_pitch_B  = image->planes[0].surface.isl.row_pitch_B;

    uint32_t width = image->extent.width;
    uint32_t height = image->extent.height;

    // Dump storage image metadata
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsetdata";

    int VkDescriptorTypeNum = 3;

    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.vkstorageimagemetadata", mesa_root, filePath, setID, binding);
    fp = fopen(fullPath, "w+");
    // File name format: setID_descID.vktexturemetadata

    fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d",   width, 
                                                height, 
                                                format, 
                                                VkDescriptorTypeNum, 
                                                image->n_planes, 
                                                image->samples, 
                                                tiling, 
                                                isl_tiling_mode,
                                                row_pitch_B);
    fclose(fp);
#elif defined(MESA_USE_LVPIPE_DRIVER)
    printf("gpgpusim: dumpStorageImage not implemented for lavapipe.\n");
    abort();

#endif
}


void VulkanRayTracing::dump_descriptor_set_for_AS(uint32_t setID, uint32_t descID, void *address, uint32_t desc_size, VkDescriptorType type, uint32_t backwards_range, uint32_t forward_range, bool split_files, VkAccelerationStructureKHR _topLevelAS)
{
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsetdata";

    int VkDescriptorTypeNum;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            VkDescriptorTypeNum = 1000150000;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            VkDescriptorTypeNum = 1000165000;
            break;
        default:
            abort(); // should not be here!
    }

    char fullPath[200];
    int result;

    int64_t max_backwards; // negative number
    int64_t min_backwards; // negative number
    int64_t min_forwards;
    int64_t max_forwards;
    int64_t back_buffer_amount = 0; //20kB buffer just in case
    int64_t front_buffer_amount = 1024*20; //20kB buffer just in case
    findOffsetBounds(max_backwards, min_backwards, min_forwards, max_forwards, _topLevelAS);

    bool haveBackwards = (max_backwards != 0) && (min_backwards != 0);
    bool haveForwards = (min_forwards != 0) && (max_forwards != 0);
    
    if (split_files) // Used when the AS is too far apart between top tree and BVHAddress and cant just dump the whole thing
    {
        // Main Top Level
        snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asmain", mesa_root, filePath, setID, descID);
        fp = fopen(fullPath, "wb+");
        result = fwrite(address, 1, desc_size, fp);
        assert(result == desc_size);
        fclose(fp);

        // Bot level whose address is smaller than top level
        if (haveBackwards)
        {
            snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asback", mesa_root, filePath, setID, descID);
            fp = fopen(fullPath, "wb+");
            result = fwrite(address + max_backwards, 1, min_backwards - max_backwards + back_buffer_amount, fp);
            assert(result == min_backwards - max_backwards + back_buffer_amount);
            fclose(fp);
        }

        // Bot level whose address is larger than top level
        if (haveForwards)
        {
            snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asfront", mesa_root, filePath, setID, descID);
            fp = fopen(fullPath, "wb+");
            result = fwrite(address + min_forwards, 1, max_forwards - min_forwards + front_buffer_amount, fp);
            assert(result == max_forwards - min_forwards + front_buffer_amount);
            fclose(fp);
        }

        // AS metadata
        snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asmetadata", mesa_root, filePath, setID, descID);
        fp = fopen(fullPath, "w+");
        fprintf(fp, "%d,%d,%ld,%ld,%ld,%ld,%ld,%ld,%d,%d", desc_size,
                                                            VkDescriptorTypeNum,
                                                            max_backwards,
                                                            min_backwards,
                                                            min_forwards,
                                                            max_forwards,
                                                            back_buffer_amount,
                                                            front_buffer_amount,
                                                            haveBackwards,
                                                            haveForwards);
        fclose(fp);

        
        // uint64_t total_size = (desc_size + backwards_range + forward_range);
        // uint64_t chunk_size = 1024*1024*20; // 20MB chunks
        // int totalFiles =  (total_size + chunk_size) / chunk_size; // rounds up

        // for (int i = 0; i < totalFiles; i++)
        // {
        //     // if split_files is 1, then look at the next number to see what the file part number is
        //     snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d_%d_%d_%d_%d_%d_%d%s", mesa_root, filePath, setID, descID, desc_size, VkDescriptorTypeNum, backwards_range, forward_range, split_files, i, extension);
        //     fp = fopen(fullPath, "wb+");
        //     int result = fwrite(address-(uint64_t)backwards_range + chunk_size * i, 1, chunk_size, fp);
        //     printf("File part %d, %d bytes written, starting address 0x%.12" PRIXPTR "\n", i, result, (uintptr_t)(address-(uint64_t)backwards_range + chunk_size * i));
        //     fclose(fp);
        // }
    }
    else 
    {
        snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d_%d_%d_%d_%d%s", mesa_root, filePath, setID, descID, desc_size, VkDescriptorTypeNum, backwards_range, forward_range, extension);
        // File name format: setID_descID_SizeInBytes_VkDescriptorType_desired_range.vkdescrptorsetdata

        fp = fopen(fullPath, "wb+");
        int result = fwrite(address-(uint64_t)backwards_range, 1, desc_size + backwards_range + forward_range, fp);
        fclose(fp);
    }
}


void VulkanRayTracing::dump_descriptor_set(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type)
{
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsetdata";

    int VkDescriptorTypeNum;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            VkDescriptorTypeNum = 0;
            break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            VkDescriptorTypeNum = 1;
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            VkDescriptorTypeNum = 2;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            VkDescriptorTypeNum = 3;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            VkDescriptorTypeNum = 4;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            VkDescriptorTypeNum = 5;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            VkDescriptorTypeNum = 6;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            VkDescriptorTypeNum = 7;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            VkDescriptorTypeNum = 8;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            VkDescriptorTypeNum = 9;
            break;
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            VkDescriptorTypeNum = 10;
            break;
        case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
            VkDescriptorTypeNum = 1000138000;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            VkDescriptorTypeNum = 1000150000;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            VkDescriptorTypeNum = 1000165000;
            break;
        case VK_DESCRIPTOR_TYPE_MUTABLE_VALVE:
            VkDescriptorTypeNum = 1000351000;
            break;
        case VK_DESCRIPTOR_TYPE_MAX_ENUM:
            VkDescriptorTypeNum = 0x7FFFFFF;
            break;
        default:
            abort(); // should not be here!
    }

    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d_%d_%d%s", mesa_root, filePath, setID, descID, size, VkDescriptorTypeNum, extension);
    // File name format: setID_descID_SizeInBytes_VkDescriptorType.vkdescrptorsetdata

    fp = fopen(fullPath, "wb+");
    fwrite(address, 1, size, fp);
    fclose(fp);
}


void VulkanRayTracing::dump_descriptor_sets(struct DESCRIPTOR_SET_STRUCT *set)
{
#if defined(MESA_USE_INTEL_DRIVER)
   for(int i = 0; i < set->descriptor_count; i++)
   {
       if(i == 3 || i > 9)
       {
            // for some reason raytracing_extended skipped binding = 3
            // and somehow they have 34 descriptor sets but only 10 are used
            // so we just skip those
            continue;
       }

        struct DESCRIPTOR_SET_STRUCT* set = VulkanRayTracing::descriptorSet;

        const struct DESCRIPTOR_LAYOUT_STRUCT *bind_layout = &set->layout->binding[i];
        struct DESCRIPTOR_STRUCT *desc = &set->descriptors[bind_layout->descriptor_index];
        void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

        assert(desc->type == bind_layout->type);

        switch (desc->type)
        {
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            {
                //return (void *)(desc);
                dumpStorageImage(desc, 0, i, desc->type);
                break;
            }
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            {
                //return desc;
                dumpTextures(desc, 0, i, desc->type);
                break;
            }

            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            {
                if (desc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                    desc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                {
                    // MRS_TODO: account for desc->offset?
                    //return anv_address_map(desc->buffer->address);
                    dump_descriptor_set(0, i, anv_address_map(desc->buffer->address), set->descriptors[i].buffer->size, set->descriptors[i].type);
                    break;
                }
                else
                {
                    struct anv_buffer_view *bview = &set->buffer_views[bind_layout->buffer_view_index];
                    //return anv_address_map(bview->address);
                    dump_descriptor_set(0, i, anv_address_map(bview->address), bview->range, set->descriptors[i].type);
                    break;
                }
            }

            case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            {
                struct anv_address_range_descriptor *desc_data = desc_map;
                //return (void *)(desc_data->address);
                //dump_descriptor_set_for_AS(0, i, (void *)(desc_data->address), desc_data->range, set->descriptors[i].type, 1024*1024*10, 1024*1024*10, true);
                break;
            }

            default:
                assert(0);
                break;
        }
   }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    printf("gpgpusim: dump_descriptor_sets not implemented for lavapipe.\n");
    abort();

#endif
}

void VulkanRayTracing::dump_AS(struct DESCRIPTOR_SET_STRUCT *set, VkAccelerationStructureKHR _topLevelAS)
{
#if defined(MESA_USE_INTEL_DRIVER)
   for(int i = 0; i < set->descriptor_count; i++)
   {
       if(i == 3 || i > 9)
       {
            // for some reason raytracing_extended skipped binding = 3
            // and somehow they have 34 descriptor sets but only 10 are used
            // so we just skip those
            continue;
       }

        struct DESCRIPTOR_SET_STRUCT* set = VulkanRayTracing::descriptorSet;

        const struct DESCRIPTOR_LAYOUT_STRUCT *bind_layout = &set->layout->binding[i];
        struct DESCRIPTOR_STRUCT *desc = &set->descriptors[bind_layout->descriptor_index];
        void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

        assert(desc->type == bind_layout->type);

        switch (desc->type)
        {
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            {
                struct anv_address_range_descriptor *desc_data = desc_map;
                //return (void *)(desc_data->address);
                dump_descriptor_set_for_AS(0, i, (void *)(desc_data->address), desc_data->range, set->descriptors[i].type, 1024*1024*10, 1024*1024*10, true, _topLevelAS);
                break;
            }

            default:
                break;
        }
    }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    printf("gpgpusim: dump_AS not implemented for lavapipe.\n");
    abort();

#endif
}

void VulkanRayTracing::dump_callparams_and_sbt(void *raygen_sbt, void *miss_sbt, void *hit_sbt, void *callable_sbt, bool is_indirect, uint32_t launch_width, uint32_t launch_height, uint32_t launch_depth, uint32_t launch_size_addr)
{
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";

    char call_params_filename [200];
    int trace_rays_call_count = 0; // just a placeholder for now
    snprintf(call_params_filename, sizeof(call_params_filename), "%s%s%d.callparams", mesa_root, filePath, trace_rays_call_count);
    fp = fopen(call_params_filename, "w+");
    fprintf(fp, "%d,%d,%d,%d,%lu", is_indirect, launch_width, launch_height, launch_depth, launch_size_addr);
    fclose(fp);

    // TODO: Is the size always 32?
    int sbt_size = 64 *sizeof(uint64_t);
    if (raygen_sbt) {
        char raygen_sbt_filename [200];
        snprintf(raygen_sbt_filename, sizeof(raygen_sbt_filename), "%s%s%d.raygensbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(raygen_sbt_filename, "wb+");
        fwrite(raygen_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }

    if (miss_sbt) {
        char miss_sbt_filename [200];
        snprintf(miss_sbt_filename, sizeof(miss_sbt_filename), "%s%s%d.misssbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(miss_sbt_filename, "wb+");
        fwrite(miss_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }

    if (hit_sbt) {
        char hit_sbt_filename [200];
        snprintf(hit_sbt_filename, sizeof(hit_sbt_filename), "%s%s%d.hitsbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(hit_sbt_filename, "wb+");
        fwrite(hit_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }

    if (callable_sbt) {
        char callable_sbt_filename [200];
        snprintf(callable_sbt_filename, sizeof(callable_sbt_filename), "%s%s%d.callablesbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(callable_sbt_filename, "wb+");
        fwrite(callable_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }
}

void VulkanRayTracing::setStorageImageFromLauncher(void *address, 
                                                void *deviceAddress, 
                                                uint32_t setID, 
                                                uint32_t descID, 
                                                uint32_t width,
                                                uint32_t height,
                                                VkFormat format,
                                                uint32_t VkDescriptorTypeNum,
                                                uint32_t n_planes,
                                                uint32_t n_samples,
                                                VkImageTiling tiling,
                                                uint32_t isl_tiling_mode, 
                                                uint32_t row_pitch_B)
{
    storage_image_metadata *storage_image = new storage_image_metadata;
    storage_image->address = address;
    storage_image->setID = setID;
    storage_image->descID = descID;
    storage_image->width = width;
    storage_image->height = height;
    storage_image->format = format;
    storage_image->VkDescriptorTypeNum = VkDescriptorTypeNum;
    storage_image->n_planes = n_planes;
    storage_image->n_samples = n_samples;
    storage_image->tiling = tiling;
    storage_image->isl_tiling_mode = isl_tiling_mode; 
    storage_image->row_pitch_B = row_pitch_B;
    storage_image->deviceAddress = deviceAddress;

    launcher_descriptorSets[setID][descID] = (void*) storage_image;
    launcher_deviceDescriptorSets[setID][descID] = (void*) storage_image;
}

void VulkanRayTracing::setTextureFromLauncher(void *address, 
                                            void *deviceAddress, 
                                            uint32_t setID, 
                                            uint32_t descID, 
                                            uint64_t size,
                                            uint32_t width,
                                            uint32_t height,
                                            VkFormat format,
                                            uint32_t VkDescriptorTypeNum,
                                            uint32_t n_planes,
                                            uint32_t n_samples,
                                            VkImageTiling tiling,
                                            uint32_t isl_tiling_mode,
                                            uint32_t row_pitch_B,
                                            uint32_t filter)
{
    texture_metadata *texture = new texture_metadata;
    texture->address = address;
    texture->setID = setID;
    texture->descID = descID;
    texture->size = size;
    texture->width = width;
    texture->height = height;
    texture->format = format;
    texture->VkDescriptorTypeNum = VkDescriptorTypeNum;
    texture->n_planes = n_planes;
    texture->n_samples = n_samples;
    texture->tiling = tiling;
    texture->isl_tiling_mode = isl_tiling_mode;
    texture->row_pitch_B = row_pitch_B;
    texture->filter = filter;
    texture->deviceAddress = deviceAddress;

    launcher_descriptorSets[setID][descID] = (void*) texture;
    launcher_deviceDescriptorSets[setID][descID] = (void*) texture;
}

void VulkanRayTracing::pass_child_addr(void *address)
{
    child_addrs_from_driver.push_back(address);
}

void VulkanRayTracing::allocBLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr) {
    VSIM_DPRINTF("gpgpusim: set BLAS address for 0x%lx at %p to %p\n", bufferSize, rootAddr, gpgpusimAddr);
    blas_addr_map[rootAddr] = gpgpusimAddr;
}

void VulkanRayTracing::allocTLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr) {
    printf("gpgpusim: set TLAS address %p to %p\n", rootAddr, gpgpusimAddr);
    tlas_addr = gpgpusimAddr;
}

void VulkanRayTracing::findOffsetBounds(int64_t &max_backwards, int64_t &min_backwards, int64_t &min_forwards, int64_t &max_forwards, VkAccelerationStructureKHR _topLevelAS)
{
    // uint64_t current_min_backwards = 0;
    // uint64_t current_max_backwards = 0;
    // uint64_t current_min_forwards = 0;
    // uint64_t current_max_forwards = 0;
    int64_t offset;

    std::vector<int64_t> positive_offsets;
    std::vector<int64_t> negative_offsets;

    for (auto addr : child_addrs_from_driver)
    {
        offset = (uint64_t)addr - (uint64_t)_topLevelAS;
        if (offset >= 0)
            positive_offsets.push_back(offset);
        else
            negative_offsets.push_back(offset);
    }

    sort(positive_offsets.begin(), positive_offsets.end());
    sort(negative_offsets.begin(), negative_offsets.end());

    if (negative_offsets.size() > 0)
    {
        max_backwards = negative_offsets.front();
        min_backwards = negative_offsets.back();
    }
    else
    {
        max_backwards = 0;
        min_backwards = 0;
    }

    if (positive_offsets.size() > 0)
    {
        min_forwards = positive_offsets.front();
        max_forwards = positive_offsets.back();
    }
    else
    {
        min_forwards = 0;
        max_forwards = 0;
    }
}


void* VulkanRayTracing::gpgpusim_alloc(uint32_t size)
{
    gpgpu_context *ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    void* devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
    if (g_debug_execution >= 3) {
        printf("GPGPU-Sim PTX: gpgpusim_allocing %zu bytes starting at 0x%llx..\n",
            size, (unsigned long long)devPtr);
        ctx->api->g_mallocPtr_Size[(unsigned long long)devPtr] = size;
    }
    assert(devPtr);

    if(!GPGPU_Context()->func_sim->g_rt_external_launch) {
        void* bufferAddr = malloc(size);
        memory_space *mem = context->get_device()->get_gpgpu()->get_global_memory();
        mem->bind_vulkan_buffer(bufferAddr, size, devPtr);
    }

    return devPtr;
}

void* VulkanRayTracing::allocBuffer(void* bufferAddr, uint64_t bufferSize)
{
    gpgpu_context *ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    void* devPtr = context->get_device()->get_gpgpu()->gpu_malloc(bufferSize);
    assert(devPtr);

    memory_space *mem = context->get_device()->get_gpgpu()->get_global_memory();
    
    printf("gpgpusim: binding gpgpusim buffer %p (size %d) to vulkan buffer %p\n", devPtr, bufferSize, bufferAddr);
    mem->bind_vulkan_buffer(bufferAddr, bufferSize, devPtr);
    return devPtr;
}

