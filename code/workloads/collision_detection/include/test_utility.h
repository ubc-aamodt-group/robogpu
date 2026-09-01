#ifndef TEST_UTILITY_H
#define TEST_UTILITY_H
#include <fcl/common/types.h>
#include <fcl/math/bv/AABB-inl.h>
#include <fcl/math/bv/OBB-inl.h>

#include <cstring>
#include <random>

#include "fpm/fixed.hpp"
#include "fpm/ios.hpp"
#include "fpm/math.hpp"

#define TWO_PI (2 * 3.1415926)
#define VOXEL_UNIT_LENGTH 1.0
using fixed_8_8 = fpm::fixed<std::int16_t, std::int32_t, 8>;
using fixed_24_8 = fpm::fixed<std::int32_t, std::int64_t, 8>;
enum FCL_TYPE { OBB, AABB, Voxel };

template <>
float geometry::OBB<float>::reps = 1e-6f;
template <>
double geometry::OBB<double>::reps = 1e-6f;
template <>
Eigen::half geometry::OBB<Eigen::half>::reps = Eigen::half(1e-6f);
template <>
fixed_8_8 geometry::OBB<fixed_8_8>::reps = fixed_8_8(4e-3f);
template <>
fixed_24_8 geometry::OBB<fixed_24_8>::reps = fixed_24_8(4e-3f);

/**
 * Given three angles between 0 and 2 pi,
 * Generate the corresponding 3x3 matrix,
 * and store it in the FCL Matrix3 object
 **/
template <typename S>
void eulerToMatrix(S a, S b, S c, fcl::Matrix3<S> &R) {
    auto c1 = cos(a);
    auto c2 = cos(b);
    auto c3 = cos(c);
    auto s1 = sin(a);
    auto s2 = sin(b);
    auto s3 = sin(c);
    R << c1 * c2, -c2 * s1, s2, c3 * s1 + c1 * s2 * s3, c1 * c3 - s1 * s2 * s3,
        -c2 * s3, s1 * s3 - c1 * c3 * s2, c3 * s1 * s2 + c1 * s3, c2 * c3;
}

/**
 * Given three angles between 0 and 2 pi,
 * Generate the corresponding 3x3 matrix,
 * and store it in the given 3x3 2D array
 **/
template <typename S>
void eulerToMatrix(S a, S b, S c, S R[3][3]) {
    auto c1 = cos(a);
    auto c2 = cos(b);
    auto c3 = cos(c);
    auto s1 = sin(a);
    auto s2 = sin(b);
    auto s3 = sin(c);
    R[0][0] = S(c1 * c2);
    R[0][1] = S(-c2 * s1);
    R[0][2] = S(s2);
    R[1][0] = S(c3 * s1 + c1 * s2 * s3);
    R[1][1] = S(c1 * c3 - s1 * s2 * s3);
    R[1][2] = S(-c2 * s3);
    R[2][0] = S(s1 * s3 - c1 * c3 * s2);
    R[2][1] = S(c3 * s1 * s2 + c1 * s3);
    R[2][2] = S(c2 * c3);
}

static std::random_device rd;
static std::mt19937 gen(rd());

/**
 * Returns a random number between min and max
 **/
static inline double r(double min, double max) {
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

/**
 * Generate a random FCL OBB object to represent
 * an OBB, AABB, or Voxel
 **/
template <typename S>
void random_fcl(fcl::OBB<S> &res, FCL_TYPE type = FCL_TYPE::OBB,
                double min = -100000.0, double max = 100000.0) {
    fcl::Matrix3<S> axis;
    fcl::Vector3<S> center;
    fcl::Vector3<S> extent;

    // the center can be anything
    for (int i = 0; i < 3; i++) {
        center[i] = S(r(min, max));
    }

    switch (type) {
        case FCL_TYPE::OBB: {
            S a = S(r(0.0, TWO_PI));
            S b = S(r(0.0, TWO_PI));
            S c = S(r(0.0, TWO_PI));
            eulerToMatrix<S>(a, b, c, axis);
            for (int i = 0; i < 3; i++) {
                extent[i] = S(r(0.0, max));
            }
            break;
        }
        case FCL_TYPE::AABB: {
            axis = fcl::Matrix3<S>::Identity();
            for (int i = 0; i < 3; i++) {
                extent[i] = S(r(0.0, max));
            }
            break;
        }

        default: {
            axis = fcl::Matrix3<S>::Identity();
            for (int i = 0; i < 3; i++) {
                extent[i] = S(VOXEL_UNIT_LENGTH);
            }
        }
    }
    res = fcl::OBB<S>(axis, center, extent);
}

/**
 * Given an FCL OBB object, convert it
 * to a geometry (namespace) OBB object
 **/
template <typename S>
void convert_obb(const fcl::OBB<S> &in, geometry::OBB<S> &out) {
    for (int i = 0; i < 3; i++) {
        out.c[i] = in.To[i];
        out.e[i] = in.extent[i];
        for (int j = 0; j < 3; j++) {
            out.u[i][j] = in.axis(i, j);
        }
    }
}

/**
 * Given an FCL OBB object that represents an AABB, convert it
 * to a geometry (namespace) AABB object
 **/
template <typename S>
void convert_aabb(const fcl::OBB<S> &in, geometry::AABB<S> &out) {
    assert(in.axis == fcl::Matrix3<S>::Identity());
    for (int i = 0; i < 3; i++) {
        out.c[i] = in.To[i];
        out.e[i] = in.extent[i];
    }
}

/**
 * Given an FCL OBB object that represents a voxel, convert it
 * to a geometry (namespace) AABB object
 **/
template <typename S>
void convert_voxel(const fcl::OBB<S> &in, geometry::AABB<S> &out) {
    assert(in.axis == fcl::Matrix3<S>::Identity());
    for (int i = 0; i < 3; i++) {
        assert(in.extent[i] == S(VOXEL_UNIT_LENGTH));
        out.c[i] = in.To[i];
        out.e[i] = in.extent[i];
    }
}

/**
 * Generate a random geometry (namespace) OBB object
 * given bounds
 **/
template <typename S>
void random_obb(geometry::OBB<S> &obb, double min = -100000.0,
                double max = 100000.0) {
    for (int i = 0; i < 3; i++) {
        obb.c[i] = S(r(min, max));
        obb.e[i] = S(r(0.0, max));
    }
    S a = S(r(0.0, TWO_PI));
    S b = S(r(0.0, TWO_PI));
    S c = S(r(0.0, TWO_PI));
    eulerToMatrix(a, b, c, obb.u);
}

/**
 * Generate a random geometry (namespace) AABB object
 * given bounds
 **/
template <typename S>
void random_aabb(geometry::AABB<S> &aabb, double min = -100000.0,
                 double max = 100000.0) {
    for (int i = 0; i < 3; i++) {
        aabb.c[i] = S(r(min, max));
        aabb.e[i] = S(r(0.0, max));
    }
}

/**
 * Generate a random geometry (namespace) AABB object
 * that represents a voxel, given bounds
 **/
template <typename S>
void random_voxel(geometry::AABB<S> &voxel, double min = -100000.0,
                  double max = 100000.0) {
    for (int i = 0; i < 3; i++) {
        voxel.c[i] = S(r(min, max));
        voxel.e[i] = S(VOXEL_UNIT_LENGTH);
    }
}

/**
 * Given a geometry (namespace) OBB object of type T1 (say float),
 * convert it to a geometry (namespace) OBB object of type T2 (say
 * fixed-point-16)
 **/
template <typename T1, typename T2>
void cast_type(const geometry::OBB<T1> &in, geometry::OBB<T2> &out,
               T1 scale_extent = T1(1.0f), bool log = false) {
    for (int i = 0; i < 3; i++) {
        out.c[i] = T2(in.c[i]);
        out.e[i] = T2(in.e[i] * scale_extent);
        if (log) {
            std::cout << in.c[i] << " " << out.c[i] << std::endl;
            std::cout << in.e[i] << " " << out.e[i] << std::endl;
        }
        for (int j = 0; j < 3; j++) {
            out.u[i][j] = T2(in.u[i][j]);
            if (log) {
                std::cout << in.u[i][j] << " " << out.u[i][j] << std::endl;
            }
        }
    }
}

/**
 * Given a geometry (namespace) AABB object of type T1 (say float),
 * convert it to a geometry (namespace) AABB object of type T2 (say
 *fixed-point-16)
 **/
template <typename T1, typename T2>
void cast_type(const geometry::AABB<T1> &in, geometry::AABB<T2> &out,
               bool log = false) {
    for (int i = 0; i < 3; i++) {
        out.c[i] = T2(in.c[i]);
        out.e[i] = T2(in.e[i]);
        if (log) {
            std::cout << in.c[i] << " " << out.c[i] << std::endl;
            std::cout << in.e[i] << " " << out.e[i] << std::endl;
        }
    }
}

template <typename S>
inline bool operator==(const geometry::AABB<S> &lhs,
                       const geometry::AABB<S> &rhs) {
    return !std::memcmp((void *)&lhs, (void *)&rhs, sizeof(geometry::AABB<S>));
}

#endif