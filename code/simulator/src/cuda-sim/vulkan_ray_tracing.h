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

#ifndef VULKAN_RAY_TRACING_H
#define VULKAN_RAY_TRACING_H

#include "vulkan/vulkan.h"
#if defined(MESA_USE_INTEL_DRIVER)
#include "vulkan/vulkan_intel.h"
#include "vulkan/anv_acceleration_structure.h"

#elif defined(MESA_USE_LVPIPE_DRIVER)
#include "lvp_acceleration_structure.h"
#endif

#include "intersection_table.h"
#include "compiler/spirv/spirv.h"

// #include "ptx_ir.h"
#include "ptx_ir.h"
#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "compiler/shader_enums.h"
#include <fstream>
#include <cmath>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN_MAX(a,b,c) MAX(MIN((a), (b)), (c))
#define MAX_MIN(a,b,c) MIN(MAX((a), (b)), (c))

// For Collision Detection
#define DOT(T, A1, A2, B1, B2)                         \
    (T(A1[2] A2) * T(B1[2] B2) + T(A1[1] A2) * T(B1[1] B2) + \
     T(A1[0] A2) * T(B1[0] B2))
#define SUB(T, A, B) {T(A[0]) - T(B[0]), T(A[1]) - T(B[1]), T(A[2]) - T(B[2])}
#define ADD(T, A, B) {T(A[0]) + T(B[0]), T(A[1]) + T(B[1]), T(A[2]) + T(B[2])}
#define COPY(T, X) {T(X[0]), T(X[1]), T(X[2])}

#define MAX_DESCRIPTOR_SETS 1
#define MAX_DESCRIPTOR_SET_BINDINGS 32

// enum class TransactionType {
//     BVH_STRUCTURE,
//     BVH_INTERNAL_NODE,
//     BVH_INSTANCE_LEAF,
//     BVH_PRIMITIVE_LEAF_DESCRIPTOR,
//     BVH_QUAD_LEAF,
//     BVH_PROCEDURAL_LEAF,
//     Intersection_Table_Load,
// };

// typedef struct MemoryTransactionRecord {
//     MemoryTransactionRecord(void* address, uint32_t size, TransactionType type)
//     : address(address), size(size), type(type) {}
//     void* address;
//     uint32_t size;
//     TransactionType type;
// } MemoryTransactionRecord;
// typedef struct float4 {
//     float x, y, z, w;
// } float4;

// enum class StoreTransactionType {
//     Intersection_Table_Store,
//     Traversal_Results,
// };

// typedef struct MemoryStoreTransactionRecord {
//     MemoryStoreTransactionRecord(void* address, uint32_t size, StoreTransactionType type)
//     : address(address), size(size), type(type) {}
//     void* address;
//     uint32_t size;
//     StoreTransactionType type;
// } MemoryStoreTransactionRecord;

enum class CollisionResult {
    HIT = 0,
    AXIS1_CLEAR,
    AXIS2_CLEAR,
    AXIS3_CLEAR,
    AXIS4_CLEAR,
    AXIS5_CLEAR,
    AXIS6_CLEAR,
    AXIS7_CLEAR,
    AXIS8_CLEAR,
    AXIS9_CLEAR,
    AXIS10_CLEAR,
    AXIS11_CLEAR,
    AXIS12_CLEAR,
    AXIS13_CLEAR,
    AXIS14_CLEAR,
    AXIS15_CLEAR,
    OUTER_SPHERE_MISS,
    INNER_SPHERE_HIT,
};

typedef struct float4x4 {
  float m[4][4];

  float4 operator*(const float4& _vec) const
  {
    float vec[] = {_vec.x, _vec.y, _vec.z, _vec.w};
    float res[] = {0, 0, 0, 0};
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            res[i] += this->m[j][i] * vec[j];
    return {res[0], res[1], res[2], res[3]};
  }
} float4x4;

typedef struct RayDebugGPUData
{
    bool valid;
    int launchIDx;
    int launchIDy;
    int instanceCustomIndex;
    int primitiveID;
    float3 v0pos;
    float3 v1pos;
    float3 v2pos;
    float3 attribs;
    float3 P_object;
    float3 P; //world intersection point
    float3 N_object;
    float3 N;
    float NdotL;
    float3 hitValue;
} RayDebugGPUData;

// float4 operator*(const float4& _vec, const float4x4& matrix)
// {
//     float vec[] = {_vec.x, _vec.y, _vec.z, _vec.w};
//     float res[] = {0, 0, 0, 0};
//     for(int i = 0; i < 4; i++)
//         for(int j = 0; j < 4; j++)
//             res[i] += matrix.m[j][i] * vec[j];
//     return {res[0], res[1], res[2], res[3]};
// }


typedef struct Descriptor
{
    uint32_t setID;
    uint32_t descID;
    void *address;
    uint32_t size;
    VkDescriptorType type;
} Descriptor;

typedef struct shader_stage_info {
    uint32_t ID;
    gl_shader_stage type;
    char* function_name;
} shader_stage_info;


typedef struct Pixel{
    Pixel(float c0, float c1, float c2, float c3)
    : c0(c0), c1(c1), c2(c2), c3(c3) {}
    Pixel() {}

    union
    {
        float r;
        float c0;
    };
    union
    {
        float g;
        float c1;
    };
    union
    {
        float b;
        float c2;
    };
    union
    {
        float a;
        float c3;
    };
} Pixel;

// For launcher
typedef struct storage_image_metadata
{
    void *address;
    void *deviceAddress;
    uint32_t setID;
    uint32_t descID;
    uint32_t width;
    uint32_t height;
    VkFormat format;
    uint32_t VkDescriptorTypeNum;
    uint32_t n_planes;
    uint32_t n_samples;
    VkImageTiling tiling;
    uint32_t isl_tiling_mode; 
    uint32_t row_pitch_B;
} storage_image_metadata;

typedef struct texture_metadata
{
    void *address;
    void *deviceAddress;
    uint32_t setID;
    uint32_t descID;
    uint64_t size;
    uint32_t width;
    uint32_t height;
    VkFormat format;
    uint32_t VkDescriptorTypeNum;
    uint32_t n_planes;
    uint32_t n_samples;
    VkImageTiling tiling;
    uint32_t isl_tiling_mode;
    uint32_t row_pitch_B;
    VkFilter filter;
} texture_metadata;


#if defined(MESA_USE_INTEL_DRIVER)
#define DESCRIPTOR_SET_STRUCT anv_descriptor_set
#define DESCRIPTOR_STRUCT anv_descriptor
#define DESCRIPTOR_LAYOUT_STRUCT anv_descriptor_set_binding_layout

#define VSIM_DEBUG_PRINT 0

struct anv_descriptor_set;
struct anv_descriptor;

#elif defined(MESA_USE_LVPIPE_DRIVER)
#define DESCRIPTOR_SET_STRUCT lvp_descriptor_set
#define DESCRIPTOR_STRUCT lvp_descriptor
#define DESCRIPTOR_LAYOUT_STRUCT lvp_descriptor_set_binding_layout

struct lvp_descriptor_set;
struct lvp_descriptor;

#endif

typedef struct child_node
{
    int child_index;
    bool hit;
    float surface_area;
    float thit; 
    uint8_t *child_addr;
} child_node;

#define VSIM_DPRINTF(...) \
   if(VSIM_DEBUG_PRINT) { \
      printf(__VA_ARGS__); \
      fflush(stdout); \
   }

void rt_traverse_tree(const ptx_instruction *pI, ptx_thread_info *thread);
void rt_ray_box_intersect(const ptx_instruction *pI, ptx_thread_info *thread);
void rt_ray_triangle_intersect(const ptx_instruction *pI, ptx_thread_info *thread);
void rt_tree_unit_magic_func(const ptx_instruction *pI, ptx_thread_info *thread);
class VulkanRayTracing
{
private:
    static VkRayTracingPipelineCreateInfoKHR* pCreateInfos;
    static VkAccelerationStructureGeometryKHR* pGeometries;
    static uint32_t geometryCount;
    static VkAccelerationStructureKHR topLevelAS;
    static std::vector<std::vector<Descriptor> > descriptors;
    static std::ofstream imageFile;
    static bool firstTime;
    static struct DESCRIPTOR_SET_STRUCT *descriptorSet;

    // For Launcher
    static void* launcher_descriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS];
    static void* launcher_deviceDescriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS];
    static std::vector<void*> child_addrs_from_driver;
    static std::map<void*, void*> VulkanRayTracing::blas_addr_map;
    static void* tlas_addr;
    static bool dumped;
    static bool _init_;
public:
    // static RayDebugGPUData rayDebugGPUData[2000][2000];
    static warp_intersection_table*** intersection_table;
    static warp_intersection_table*** anyhit_table;
    static IntersectionTableType intersectionTableType;

private:
    static std::vector<shader_stage_info> shaders;

    static void init(uint32_t launch_width, uint32_t launch_height);


public:
    static void traceRay( // called by raygen shader
                       VkAccelerationStructureKHR _topLevelAS,
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
                       ptx_thread_info *thread);
    static void endTraceRay(const ptx_instruction *pI, ptx_thread_info *thread);
    
    static void load_descriptor(const ptx_instruction *pI, ptx_thread_info *thread);

    static bool rtao_ray_triangle_test(float4 v00, float4 v11, float4 v22, Ray ray_properties, float* thit, float3* bary);
    static bool raySphereIntersection(float4 sphere, Ray ray_properties, float* thit);
    static bool pointSphereIntersection(float4 sphere, Ray ray_properties);
    static bool mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit);
    static float3 Barycentric(float3 p, float3 a, float3 b, float3 c);
    
    static void setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos);
    static void setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount);
    static void setAccelerationStructure(VkAccelerationStructureKHR accelerationStructure);
    static void setDescriptorSet(struct DESCRIPTOR_SET_STRUCT *set);
    static void invoke_gpgpusim();
    static uint32_t registerShaders(char * shaderPath, gl_shader_stage shaderType);
    static void vkCmdTraceRaysKHR( // called by vulkan application
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);
    static void callShader(const ptx_instruction *pI, ptx_thread_info *thread, function_info *target_func);
    static void callMissShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void callClosestHitShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void callIntersectionShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter);
    static void callAnyHitShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter);
    static void setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);
    static void* getDescriptorAddress(uint32_t setID, uint32_t binding);

    static void write_image_file(uint32_t width, uint32_t height, float hitValue_X, float hitValue_Y, float hitValue_Z, uint32_t pixelX, uint32_t pixelY, VkFormat img_format);
    static void image_store(struct DESCRIPTOR_STRUCT* desc, uint32_t gl_LaunchIDEXT_X, uint32_t gl_LaunchIDEXT_Y, uint32_t gl_LaunchIDEXT_Z, uint32_t gl_LaunchsIDEXT_W, 
              float hitValue_X, float hitValue_Y, float hitValue_Z, float hitValue_W, const ptx_instruction *pI, ptx_thread_info *thread);
    static void getTexture(struct DESCRIPTOR_STRUCT *desc, float x, float y, float lod, float &c0, float &c1, float &c2, float &c3, std::vector<ImageMemoryTransactionRecord>& transactions, uint64_t launcher_offset = 0);
    static void image_load(struct DESCRIPTOR_STRUCT *desc, uint32_t x, uint32_t y, float &c0, float &c1, float &c2, float &c3);

    static void dump_descriptor_set(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);
    static void dump_descriptor_set_for_AS(uint32_t setID, uint32_t descID, void *address, uint32_t desc_size, VkDescriptorType type, uint32_t backwards_range, uint32_t forward_range, bool split_files, VkAccelerationStructureKHR _topLevelAS);
    static void dump_descriptor_sets(struct DESCRIPTOR_SET_STRUCT *set);
    static void dump_AS(struct DESCRIPTOR_SET_STRUCT *set, VkAccelerationStructureKHR _topLevelAS);
    static void dump_callparams_and_sbt(void *raygen_sbt, void *miss_sbt, void *hit_sbt, void *callable_sbt, bool is_indirect, uint32_t launch_width, uint32_t launch_height, uint32_t launch_depth, uint32_t launch_size_addr);
    static void dumpTextures(struct DESCRIPTOR_STRUCT *desc, uint32_t setID, uint32_t binding, VkDescriptorType type);
    static void dumpStorageImage(struct DESCRIPTOR_STRUCT *desc, uint32_t setID, uint32_t binding, VkDescriptorType type);
    static void setDescriptorSetFromLauncher(void *address, void *deviceAddress, uint32_t setID, uint32_t descID);
    static void setStorageImageFromLauncher(void *address, 
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
                                            uint32_t row_pitch_B);
    static void setTextureFromLauncher(void *address,
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
                                       uint32_t filter);
    static void pass_child_addr(void *address);
    static void allocBLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr);
    static void allocTLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr);
    static void* allocBuffer(void* bufferAddr, uint64_t bufferSize);
    static void findOffsetBounds(int64_t &max_backwards, int64_t &min_backwards, int64_t &min_forwards, int64_t &max_forwards, VkAccelerationStructureKHR _topLevelAS);
    static void* gpgpusim_alloc(uint32_t size);
};


/* Collision Detection (Ningfeng's code) */
namespace geometry {

template <typename S>
inline int obbDisjoint(const S B[3][3], const S T[3], const S a[3],
                        const S b[3]);

template <typename S>
struct AABB {
    S c[3];  // AABB center point
    S e[3];  // Positive halfwidth extents of AABB along each axis
};

typedef struct octree_node {
    int child_address;
    int child_status[8];
    struct AABB<double> children_box[8];
    int level;
} octree_node;

template <typename S>
struct OBB {
    S u[3][3];  // Local x-, y-, and z-axes
    S c[3];     // OBB center point
    S e[3];     // Positive halfwidth extents of OBB along each axis
    static S reps;

    /**
     * Test if this OBB overlaps with the other G (geometry)
     * G must be one of OBB<S> or AABB<S>.
     * If G is AABB<S>, then R is u transposed.
     * If G is OBB<S>, then R is u transposed times other.u.
     **/
    template <typename G>
    inline int disjoint(const G &other) {
        S t[3] = SUB(, other.c, c);
        S T[3] = {DOT(, u, [0], t, ), DOT(, u, [1], t, ), DOT(, u, [2], t, )};
        S R[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // if constexpr (std::is_same<G, AABB<S>>::value) { // for CPU version
                if (std::is_same<G, AABB<S>>::value) { // for GPU verison
                    R[i][j] = u[j][i];
                } else {
                    R[i][j] = DOT(, u, [i], other.u, [j]);
                }
            }
        }
        return geometry::obbDisjoint(R, T, e, other.e);
    }

    /**
     * Same as the function above, except that
     * intermediate results are stored in a larger type L
     **/
    template <typename L, typename G>
    inline int disjoint(const G &other) {
        L t[3] = SUB(L, other.c, c);
        L T[3] = {DOT(L, u, [0], t, ), DOT(L, u, [1], t, ),
                  DOT(L, u, [2], t, )};
        L R[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // if constexpr (std::is_same<G, AABB<S>>::value) { // for CPU version
                if (std::is_same<G, AABB<S>>::value) { // for GPU verison
                    R[i][j] = L(u[j][i]);
                } else {
                    R[i][j] = DOT(L, u, [i], other.u, [j]);
                }
            }
        }
        L my_e[3] = COPY(L, e);
        L other_e[3] = COPY(L, other.e);
        return geometry::obbDisjoint(R, T, my_e, other_e);
    }

    /**
     * Tests if two OBBs overlap
     */
    bool overlap(const OBB<S> &other) { return !disjoint(other); }

    /**
     * Tests if this OBBs overlaps with the other AABB
     */
    bool overlap(const AABB<S> &other) { return !disjoint(other); }

    /**
     * This is basically the function above with the intermediate
     * results stored in a larger type L to prevent overflow
     **/
    template <typename L>
    bool overlap(const AABB<S> &other) {
        return !disjoint<L, AABB<S>>(other);
    }

    void computeVertices(S vertices[8][3]) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 3; j++) {
                vertices[i][j] = c[j];
                for (int k = 0; k < 3; k++) {
                    S extAxis = u[j][k] * e[k];
                    vertices[i][j] += ((i >> k) & 1) ? extAxis : -extAxis;
                }
            }
        }
        // align with FCL's preference
        for(int i = 0; i < 3; i++){
            std::swap(vertices[2][i], vertices[3][i]);
            std::swap(vertices[6][i], vertices[7][i]);
        }
    }

    void computeEdges(S edges[12][2][3]){
        const int n[12][2] = {
            {0, 1}, {0, 3}, {0, 4}, {1, 2}, {1, 5}, {2, 3},
            {2, 6}, {3, 7}, {4, 5}, {4, 7}, {5, 6}, {6, 7},
        };
        S vertices[8][3];
        computeVertices(vertices);
        for(int i = 0; i < 12; i++){
            for(int j = 0; j < 3; j++){
                edges[i][0][j] = vertices[n[i][0]][j];
                edges[i][1][j] = vertices[n[i][1]][j];
            }
        }
    }
};

template <typename S>
struct Sphere {
    S c[3];
    S radius_squared;
    static struct Sphere<S> create_min(const OBB<S> obb) {
        struct Sphere<S> res;
        for (int i = 0; i < 3; i++) {
            res.c[i] = obb.c[i];
        }
        S radius = std::min(std::min(obb.e[0], obb.e[1]), obb.e[2]);
        res.radius_squared = radius * radius;
        return res;
    }

    static struct Sphere<S>
    create_max(const OBB<S> obb) {
        struct Sphere<S> res;
        for (int i = 0; i < 3; i++) {
            res.c[i] = obb.c[i];
        }
        res.radius_squared = obb.e[0] * obb.e[0] + obb.e[1] * obb.e[1] + obb.e[2] * obb.e[2];
        return res;
    }

    int
    overlap(const AABB<S> aabb) {
        S min[3] = SUB(, aabb.c, aabb.e);
        S max[3] = ADD(, aabb.c, aabb.e);
        S dist = S(0);
        for (int i = 0; i < 3; i++) {
            S v = c[i];
            if (v < min[i]) {
                dist += (min[i] - v) * (min[i] - v);
            } else if (v > max[i]) {
                dist += (max[i] - v) * (max[i] - v);
            }
        }
        return (dist <= radius_squared);
    }
};

/**
 * Common logic between OBB-OBB tests and OBB-AABB tests
 * With all matrix arithmetic simplified to scalar arithmetic
 */
template <typename S>
inline int obbDisjoint(const S B[3][3], const S T[3], const S a[3],
                        const S b[3]) {
    // FIXPOINT: bits(B) = 2
    // FIXPOINT: bits(T) = bits(c) + 5
    // FIXPOINT: bits(a) = bits(e)
    // FIXPOINT: bits(b) = bits(e)
    S t, s;

    // FIXPOINT: bits(Bf) = max(bits(B), bits(reps)) + 1 = 3
    S Bf[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Bf[i][j] = abs(B[i][j]) + OBB<S>::reps;
        }
    }

    // if any of these tests are one-sided, then the polyhedra are disjoint

    // A1 x A2 = A0
    // FIXPOINT: bits(t) = bits(T) = bits(c) + 5
    t = abs(T[0]);
    // FIXPOINT: bits(RHS) 
    // = max(bits(a), bits(Bf) + bits(b) + 2) + 1 
    // = max(bits(e), 3 + bits(e) + 2) + 1 
    // = bits(e) + 6
    if (t > (a[0] + DOT(, Bf[0], , b, ))) return 1;

    // B1 x B2 = B0
    // FIXPOINT: bits(s) 
    // = bits(B) + bits(T) + 2 
    // = 2 + (bits(c) + 5) + 2 
    // = bits(c) + 9
    s = DOT(, B, [0], T, );
    // FIXPOINT: bits(t) = bits(s) = bits(c) + 9
    t = abs(s);

    // FIXPOINT: bits(RHS) 
    // = max(bits(b), bits(Bf) + bits(a) + 2) + 1
    // = max(bits(e), 3 + bits(e) + 2) + 1
    // = bits(e) + 6
    if (t > (b[0] + DOT(, Bf, [0], a, ))) return 2;

    // A2 x A0 = A1
    t = abs(T[1]);

    if (t > (a[1] + DOT(, Bf[1], , b, ))) return 3;

    // A0 x A1 = A2
    t = abs(T[2]);

    if (t > (a[2] + DOT(, Bf[2], , b, ))) return 4;

    // B2 x B0 = B1
    s = DOT(, B, [1], T, );
    t = abs(s);

    if (t > (b[1] + DOT(, Bf, [1], a, ))) return 5;

    // B0 x B1 = B2
    s = DOT(, B, [2], T, );
    t = abs(s);

    if (t > (b[2] + DOT(, Bf, [2], a, ))) return 6;

    // A0 x B0
    // FIXPOINT: bits(s) 
    // = bits(B) + bits(T) + 1 
    // = 2 + (bits(c) + 5) + 1
    // = bits(c) + 8
    s = T[2] * B[1][0] - T[1] * B[2][0];
    // FIXPOINT: bits(t) = bits(s) = bits(c) + 8
    t = abs(s);

    // FIXPOINT: bits(RHS) 
    // = max(bits(Bf) + bits(a) + 1, bits(Bf) + bits(b) + 1) + 1
    // = max(3 + bits(e) + 1, 3 + bits(e) + 1) + 1
    // = bits(e) + 5
    if (t >
        (a[1] * Bf[2][0] + a[2] * Bf[1][0] + b[1] * Bf[0][2] + b[2] * Bf[0][1]))
        return 7;

    // A0 x B1
    s = T[2] * B[1][1] - T[1] * B[2][1];
    t = abs(s);

    if (t >
        (a[1] * Bf[2][1] + a[2] * Bf[1][1] + b[0] * Bf[0][2] + b[2] * Bf[0][0]))
        return 8;

    // A0 x B2
    s = T[2] * B[1][2] - T[1] * B[2][2];
    t = abs(s);

    if (t >
        (a[1] * Bf[2][2] + a[2] * Bf[1][2] + b[0] * Bf[0][1] + b[1] * Bf[0][0]))
        return 9;

    // A1 x B0
    s = T[0] * B[2][0] - T[2] * B[0][0];
    t = abs(s);

    if (t >
        (a[0] * Bf[2][0] + a[2] * Bf[0][0] + b[1] * Bf[1][2] + b[2] * Bf[1][1]))
        return 10;

    // A1 x B1
    s = T[0] * B[2][1] - T[2] * B[0][1];
    t = abs(s);

    if (t >
        (a[0] * Bf[2][1] + a[2] * Bf[0][1] + b[0] * Bf[1][2] + b[2] * Bf[1][0]))
        return 11;

    // A1 x B2
    s = T[0] * B[2][2] - T[2] * B[0][2];
    t = abs(s);

    if (t >
        (a[0] * Bf[2][2] + a[2] * Bf[0][2] + b[0] * Bf[1][1] + b[1] * Bf[1][0]))
        return 12;

    // A2 x B0
    s = T[1] * B[0][0] - T[0] * B[1][0];
    t = abs(s);

    if (t >
        (a[0] * Bf[1][0] + a[1] * Bf[0][0] + b[1] * Bf[2][2] + b[2] * Bf[2][1]))
        return 13;

    // A2 x B1
    s = T[1] * B[0][1] - T[0] * B[1][1];
    t = abs(s);

    if (t >
        (a[0] * Bf[1][1] + a[1] * Bf[0][1] + b[0] * Bf[2][2] + b[2] * Bf[2][0]))
        return 14;

    // A2 x B2
    s = T[1] * B[0][2] - T[0] * B[1][2];
    t = abs(s);

    if (t >
        (a[0] * Bf[1][2] + a[1] * Bf[0][2] + b[0] * Bf[2][1] + b[1] * Bf[2][0]))
        return 15;

    return 0;
}

#define C_STYLE
#define BLAH(FLOAT) FLOAT

/**
 * Print an OBB to an ostream using C style or 
 * csv-cell style
 */
template <typename S>
std::ostream &operator<<(std::ostream &os, const AABB<S> &x) {
#ifdef C_STYLE
    return (os << "{"
               << "{" << BLAH(x.c[0]) << "," << BLAH(x.c[1]) << ","
               << BLAH(x.c[2]) << "},"
               << "{" << BLAH(x.e[0]) << "," << BLAH(x.e[1]) << ","
               << BLAH(x.e[2]) << "},"
               << "}");
#else
    return (os << x.c[0] << " " << x.c[1] << " " << x.c[2] << " " << x.e[0]
               << " " << x.e[1] << " " << x.e[2]);
#endif
}

/**
 * Print an OBB to an ostream using C style or 
 * csv-cell style
 */
template <typename S>
std::ostream &operator<<(std::ostream &os, const OBB<S> &x) {
#ifdef C_STYLE
    return (os << "{"
               << "{"
               << "{" << BLAH(x.u[0][0]) << "," << BLAH(x.u[0][1]) << ","
               << BLAH(x.u[0][2]) << "},"
               << "{" << BLAH(x.u[1][0]) << "," << BLAH(x.u[1][1]) << ","
               << BLAH(x.u[1][2]) << "},"
               << "{" << BLAH(x.u[2][0]) << "," << BLAH(x.u[2][1]) << ","
               << BLAH(x.u[2][2]) << "},"
               << "},"
               << "{" << BLAH(x.c[0]) << "," << BLAH(x.c[1]) << ","
               << BLAH(x.c[2]) << "},"
               << "{" << BLAH(x.e[0]) << "," << BLAH(x.e[1]) << ","
               << BLAH(x.e[2]) << "},"
               << "}");
#else
    return (os << x.c[0] << " " << x.c[1] << " " << x.c[2] << " " << x.e[0]
               << " " << x.e[1] << " " << x.e[2] << " " << x.u[0][0] << " "
               << x.u[0][1] << " " << x.u[0][2] << " " << x.u[1][0] << " "
               << x.u[1][1] << " " << x.u[1][2] << " " << x.u[2][0] << " "
               << x.u[2][1] << " " << x.u[2][2]);
#endif
}

template <typename S>
CollisionResult obbDisjoint(const S B[3][3], const S T[3], const S a[3], const S b[3], S reps) {
  // FIXPOINT: bits(B) = 2
  // FIXPOINT: bits(T) = bits(c) + 5
  // FIXPOINT: bits(a) = bits(e)
  // FIXPOINT: bits(b) = bits(e)
  S t, s;

  // FIXPOINT: bits(Bf) = max(bits(B), bits(reps)) + 1 = 3
  S Bf[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Bf[i][j] = abs(B[i][j]) + reps;
    }
  }

  // if any of these tests are one-sided, then the polyhedra are disjoint

  // A1 x A2 = A0
  // FIXPOINT: bits(t) = bits(T) = bits(c) + 5
  t = abs(T[0]);
  // FIXPOINT: bits(RHS)
  // = max(bits(a), bits(Bf) + bits(b) + 2) + 1
  // = max(bits(e), 3 + bits(e) + 2) + 1
  // = bits(e) + 6
  if (t > (a[0] + DOT(, Bf[0], , b, ))) return CollisionResult::AXIS1_CLEAR;

  // B1 x B2 = B0
  // FIXPOINT: bits(s)
  // = bits(B) + bits(T) + 2
  // = 2 + (bits(c) + 5) + 2
  // = bits(c) + 9
  s = DOT(, B, [0], T, );
  // FIXPOINT: bits(t) = bits(s) = bits(c) + 9
  t = abs(s);

  // FIXPOINT: bits(RHS)
  // = max(bits(b), bits(Bf) + bits(a) + 2) + 1
  // = max(bits(e), 3 + bits(e) + 2) + 1
  // = bits(e) + 6
  if (t > (b[0] + DOT(, Bf, [0], a, ))) return CollisionResult::AXIS2_CLEAR;

  // A2 x A0 = A1
  t = abs(T[1]);

  if (t > (a[1] + DOT(, Bf[1], , b, ))) return CollisionResult::AXIS3_CLEAR;

  // A0 x A1 = A2
  t = abs(T[2]);

  if (t > (a[2] + DOT(, Bf[2], , b, ))) return CollisionResult::AXIS4_CLEAR;

  // B2 x B0 = B1
  s = DOT(, B, [1], T, );
  t = abs(s);

  if (t > (b[1] + DOT(, Bf, [1], a, ))) return CollisionResult::AXIS5_CLEAR;

  // B0 x B1 = B2
  s = DOT(, B, [2], T, );
  t = abs(s);

  if (t > (b[2] + DOT(, Bf, [2], a, ))) return CollisionResult::AXIS6_CLEAR;

  // A0 x B0
  // FIXPOINT: bits(s)
  // = bits(B) + bits(T) + 1
  // = 2 + (bits(c) + 5) + 1
  // = bits(c) + 8
  s = T[2] * B[1][0] - T[1] * B[2][0];
  // FIXPOINT: bits(t) = bits(s) = bits(c) + 8
  t = abs(s);

  // FIXPOINT: bits(RHS)
  // = max(bits(Bf) + bits(a) + 1, bits(Bf) + bits(b) + 1) + 1
  // = max(3 + bits(e) + 1, 3 + bits(e) + 1) + 1
  // = bits(e) + 5
  if (t >
      (a[1] * Bf[2][0] + a[2] * Bf[1][0] + b[1] * Bf[0][2] + b[2] * Bf[0][1]))
    return CollisionResult::AXIS7_CLEAR;

  // A0 x B1
  s = T[2] * B[1][1] - T[1] * B[2][1];
  t = abs(s);

  if (t >
      (a[1] * Bf[2][1] + a[2] * Bf[1][1] + b[0] * Bf[0][2] + b[2] * Bf[0][0]))
    return CollisionResult::AXIS8_CLEAR;

  // A0 x B2
  s = T[2] * B[1][2] - T[1] * B[2][2];
  t = abs(s);

  if (t >
      (a[1] * Bf[2][2] + a[2] * Bf[1][2] + b[0] * Bf[0][1] + b[1] * Bf[0][0]))
    return CollisionResult::AXIS9_CLEAR;

  // A1 x B0
  s = T[0] * B[2][0] - T[2] * B[0][0];
  t = abs(s);

  if (t >
      (a[0] * Bf[2][0] + a[2] * Bf[0][0] + b[1] * Bf[1][2] + b[2] * Bf[1][1]))
    return CollisionResult::AXIS10_CLEAR;

  // A1 x B1
  s = T[0] * B[2][1] - T[2] * B[0][1];
  t = abs(s);

  if (t >
      (a[0] * Bf[2][1] + a[2] * Bf[0][1] + b[0] * Bf[1][2] + b[2] * Bf[1][0]))
    return CollisionResult::AXIS11_CLEAR;

  // A1 x B2
  s = T[0] * B[2][2] - T[2] * B[0][2];
  t = abs(s);

  if (t >
      (a[0] * Bf[2][2] + a[2] * Bf[0][2] + b[0] * Bf[1][1] + b[1] * Bf[1][0]))
    return CollisionResult::AXIS12_CLEAR;

  // A2 x B0
  s = T[1] * B[0][0] - T[0] * B[1][0];
  t = abs(s);

  if (t >
      (a[0] * Bf[1][0] + a[1] * Bf[0][0] + b[1] * Bf[2][2] + b[2] * Bf[2][1]))
    return CollisionResult::AXIS13_CLEAR;

  // A2 x B1
  s = T[1] * B[0][1] - T[0] * B[1][1];
  t = abs(s);

  if (t >
      (a[0] * Bf[1][1] + a[1] * Bf[0][1] + b[0] * Bf[2][2] + b[2] * Bf[2][0]))
    return CollisionResult::AXIS14_CLEAR;

  // A2 x B2
  s = T[1] * B[0][2] - T[0] * B[1][2];
  t = abs(s);

  if (t >
      (a[0] * Bf[1][2] + a[1] * Bf[0][2] + b[0] * Bf[2][1] + b[1] * Bf[2][0]))
    return CollisionResult::AXIS15_CLEAR;

  return CollisionResult::HIT; // hit
}

template <typename S>
CollisionResult obbOverlap(const geometry::OBB<S> &self, const geometry::AABB<S> &other) {
  S t[3] = SUB(, other.c, self.c);
  S T[3] = {DOT(, self.u, [0], t, ), DOT(, self.u, [1], t, ),
            DOT(, self.u, [2], t, )};
  S R[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R[i][j] = self.u[j][i];
    }
  }
  return obbDisjoint(R, T, self.e, other.e, 1e-6);
}

template <typename S>
struct geometry::Sphere<S> createMaxSphere(const geometry::OBB<S> &obb) {
  struct geometry::Sphere<S> res;
  for (int i = 0; i < 3; i++) {
    res.c[i] = obb.c[i];
  }
  res.radius_squared =
      obb.e[0] * obb.e[0] + obb.e[1] * obb.e[1] + obb.e[2] * obb.e[2];
  return res;
}

template <typename S>
struct geometry::Sphere<S> createMinSphere(const geometry::OBB<S> &obb) {
  struct geometry::Sphere<S> res;
  for (int i = 0; i < 3; i++) {
    res.c[i] = obb.c[i];
  }
  S radius = std::min(std::min(obb.e[0], obb.e[1]), obb.e[2]);
  res.radius_squared = radius * radius;
  return res;
}

template <typename S>
int sphereOverlap(const geometry::Sphere<S> &sphere, const geometry::AABB<S> &aabb) {
  S min[3] = SUB(, aabb.c, aabb.e);
  S max[3] = ADD(, aabb.c, aabb.e);
  S dist = 0;
  for (int i = 0; i < 3; i++) {
    S v = sphere.c[i];
    if (v < min[i]) {
      dist += (min[i] - v) * (min[i] - v);
    } else if (v > max[i]) {
      dist += (max[i] - v) * (max[i] - v);
    }
  }
  return (dist <= sphere.radius_squared);
}

template <typename S>
CollisionResult collisionOverlap(const geometry::OBB<S> &obb, const geometry::AABB<S> &aabb) {
  int optimize = GPGPU_Context()->func_sim->g_rt_optimize;

  if (optimize == 1) {
    auto max = createMaxSphere(obb);
    if (!sphereOverlap(max, aabb)) {
        return CollisionResult::OUTER_SPHERE_MISS; // If outer sphere does not overlap, there is no collision
    }
    auto min = createMinSphere(obb);
    if (sphereOverlap(min, aabb)) {
        return CollisionResult::INNER_SPHERE_HIT; // If inner sphere overlaps, there is a collision
    }
  }
  // Only run bounding sphere test   
  else if (optimize == 2) {
    auto max = createMaxSphere(obb);
    if (!sphereOverlap(max, aabb)) {
        return CollisionResult::OUTER_SPHERE_MISS; // If outer sphere does not overlap, there is no collision
    }
  }

  return obbOverlap(obb, aabb); // Returns 0 for collision, 1-15 for no collision exits
}

}  // namespace geometry

#endif /* VULKAN_RAY_TRACING_H */
