#ifndef CUDA_QUEUE_H
#define CUDA_QUEUE_H

#include <cuda_runtime.h>
#include <stdlib.h>

#include "geometry.h"

#ifdef RT_UNIT_ACCEL
// These structures just need to match in size to the ray tracing version

// Search Conditions (Ray - 48 bytes)
typedef struct TreeSearchConditions {
    uint32_t values[15]; // was 12
} TreeSearchConditions;

// Search results (HitPayload - 32 bytes)
typedef struct TreeSearchResults {
    uint32_t values[8];
} TreeSearchResults;

#endif

typedef struct queue {
  int len;
  int head;
  int tail;
  int *data;
} queue_t;

typedef struct group_queue {
  int group_size;
  int len;
  int *head;
  int *tail;
  int *data;
} group_queue_t;

void init(queue_t *q, int len, bool gpu = false) {
  assert(len >= 0);
  len += 1;
  q->len = len;
  q->head = 0;
  q->tail = 0;
  if (gpu) {
    cudaMalloc(&q->data, sizeof(int) * len);
    cudaMemset(q->data, 0, sizeof(int) * len);
  } else {
    q->data = (int *)calloc(len, sizeof(int));
  }
}

void group_init(group_queue_t *q, int group_size, int len, bool gpu = false) {
  assert(len > 0);
  assert(len >= 0);
  len += 1;
  q->group_size = group_size;
  q->len = len;
  if (gpu) {
    cudaMalloc(&q->head, sizeof(int) * group_size);
    cudaMemset(q->head, 0, sizeof(int) * group_size);

    cudaMalloc(&q->tail, sizeof(int) * group_size);
    cudaMemset(q->tail, 0, sizeof(int) * group_size);

    cudaMalloc(&q->data, sizeof(int) * len * group_size);
    cudaMemset(q->data, 0, sizeof(int) * len * group_size);
  } else {
    q->head = (int *)calloc(group_size, sizeof(int));
    q->tail = (int *)calloc(group_size, sizeof(int));
    q->data = (int *)calloc(len * group_size, sizeof(int));
  }
}

void destroy(queue_t *q, bool gpu = false) {
  if (gpu) {
    cudaFree(q->data);
  } else {
    free(q->data);
  }
  q->data = NULL;
}

void group_destroy(group_queue_t *q, bool gpu = false) {
  if (gpu) {
    cudaFree(q->head);
    cudaFree(q->tail);
    cudaFree(q->data);
  } else {
    free(q->head);
    free(q->tail);
    free(q->data);
  }
  q->head = NULL;
  q->tail = NULL;
  q->data = NULL;
  q->group_size = 0;
  q->len = 0;
}

__host__ __device__ bool full(queue_t *q) {
  return ((q->tail + 1) % q->len) == q->head;
}

__host__ __device__ bool group_full(group_queue *q, int g) {
  assert(g < q->group_size);
  return ((q->tail[g] + 1) % q->len) == q->head[g];
}

__host__ __device__ bool empty(queue_t *q) { return (q->head == q->tail); }

__host__ __device__ bool group_empty(group_queue *q, int g) {
  assert(g < q->group_size);
  return (q->head[g] == q->tail[g]);
}

__host__ __device__ void push(queue_t *q, int val) {
  assert(!full(q));
  q->data[q->tail] = val;
  q->tail = (q->tail + 1) % q->len;
}

__host__ __device__ void group_push(group_queue *q, int g, int val) {
  assert(!group_full(q, g));
  q->data[q->tail[g] * q->group_size + g] = val;
  q->tail[g] = (q->tail[g] + 1) % q->len;
}

__host__ __device__ int pop(queue_t *q) {
  assert(!empty(q));
  int result = q->data[q->head];
  q->head = (q->head + 1) % q->len;
  return result;
}

__host__ __device__ int group_pop(group_queue *q, int g) {
  assert(!group_empty(q, g));
  int result = q->data[q->head[g] * q->group_size + g];
  q->head[g] = (q->head[g] + 1) % q->len;
  return result;
}

template <typename S>
__host__ __device__ int cudaObbDisjoint(const S B[3][3], const S T[3],
                                        const S a[3], const S b[3], S reps) {
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

template <typename S>
__host__ __device__ int cudaObbOverlap(const geometry::OBB<S> &self,
                                       const geometry::AABB<S> &other) {
  S t[3] = SUB(, other.c, self.c);
  S T[3] = {DOT(, self.u, [0], t, ), DOT(, self.u, [1], t, ),
            DOT(, self.u, [2], t, )};
  S R[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R[i][j] = self.u[j][i];
    }
  }
  return !cudaObbDisjoint(R, T, self.e, other.e, 1e-6);
}

template <typename S>
__host__ __device__ struct geometry::Sphere<S> cudaCreateMinSphere(
    const geometry::OBB<S> &obb) {
  struct geometry::Sphere<S> res;
  for (int i = 0; i < 3; i++) {
    res.c[i] = obb.c[i];
  }
  S radius = min(min(obb.e[0], obb.e[1]), obb.e[2]);
  res.radius_squared = radius * radius;
  return res;
}

template <typename S>
__host__ __device__ struct geometry::Sphere<S> cudaCreateMaxSphere(
    const geometry::OBB<S> &obb) {
  struct geometry::Sphere<S> res;
  for (int i = 0; i < 3; i++) {
    res.c[i] = obb.c[i];
  }
  res.radius_squared =
      obb.e[0] * obb.e[0] + obb.e[1] * obb.e[1] + obb.e[2] * obb.e[2];
  return res;
}

template <typename S>
__host__ __device__ int cudaSphereOverlap(const geometry::Sphere<S> &sphere,
                                          const geometry::AABB<S> &aabb) {
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
__host__ __device__ int cudaOverlapMod(const geometry::OBB<S> &obb,
                                       const geometry::AABB<S> &aabb) {
  auto max = cudaCreateMaxSphere(obb);
  if (!cudaSphereOverlap(max, aabb)) {
    return 0;
  }
  auto min = cudaCreateMinSphere(obb);
  if (cudaSphereOverlap(min, aabb)) {
    return 1;
  }
  return cudaObbOverlap(obb, aabb);
}

#endif