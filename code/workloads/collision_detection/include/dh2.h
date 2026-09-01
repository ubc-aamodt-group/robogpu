#ifndef DH2_H
#define DH2_H

#include <fcl/common/types.h>

#include <iomanip>
#include <iostream>

#include "fpm/fixed.hpp"
#include "fpm/ios.hpp"
#include "fpm/math.hpp"
#include "geometry.h"

template <typename S>
class DH2 {
   public:
    S matrix[3][4];

    DH2() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                matrix[i][j] = S(i == j);
            }
        }
    }

    DH2(S q, S a, S r, S d) {
        // row 0
        matrix[0][0] = cos(q);
        matrix[0][1] = -cos(a) * sin(q);
        matrix[0][2] = sin(a) * sin(q);
        // row 1
        matrix[1][0] = sin(q);
        matrix[1][1] = cos(a) * cos(q);
        matrix[1][2] = -sin(a) * cos(q);
        // row 2
        matrix[2][0] = 0;
        matrix[2][1] = sin(a);
        matrix[2][2] = cos(a);
        // translation
        matrix[0][3] = r * cos(q);
        matrix[1][3] = r * sin(q);
        matrix[2][3] = d;
    }

    DH2<S> mult(DH2<S> other, const S extent[3], geometry::OBB<S> &obb) {
        DH2<S> res;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                res.matrix[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    res.matrix[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
            res.matrix[i][3] += matrix[i][3];
        }

        other.matrix[0][3] /= S(2.0);
        other.matrix[1][3] /= S(2.0);

        for (int i = 0; i < 3; i++) {
            obb.c[i] = matrix[i][3];
            for (int k = 0; k < 3; k++) {
                obb.c[i] += matrix[i][k] * other.matrix[k][3];
            }
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                obb.u[i][j] = res.matrix[i][j];
            }
            obb.e[i] = extent[i];
        }
        return res;
    }

    void print() {
        std::cout << std::setprecision(2) << std::fixed;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

#endif
