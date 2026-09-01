#ifndef DH_H
#define DH_H

#include <fcl/common/types.h>

#include <iomanip>
#include <iostream>

#include "fpm/fixed.hpp"
#include "fpm/ios.hpp"
#include "fpm/math.hpp"
#include "geometry.h"

template <typename S>
class DH {
   public:
    S matrix[4][4];

    DH() {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                matrix[i][j] = S(i == j);
            }
        }
    }

    DH(S q, S a, S r, S d) {
        // row 0
        matrix[0][0] = cos(q);
        matrix[0][1] = -cos(a) * sin(q);
        matrix[0][2] = sin(a) * sin(q);
        matrix[0][3] = r * cos(q);
        // row 1
        matrix[1][0] = sin(q);
        matrix[1][1] = cos(a) * cos(q);
        matrix[1][2] = -sin(a) * cos(q);
        matrix[1][3] = r * sin(q);
        // row 2
        matrix[2][0] = 0;
        matrix[2][1] = sin(a);
        matrix[2][2] = cos(a);
        matrix[2][3] = d;
        // row 3
        matrix[3][0] = S(0);
        matrix[3][1] = S(0);
        matrix[3][2] = S(0);
        matrix[3][3] = S(1);
    }

    DH<S> mult(DH<S> other) {
        DH<S> res;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                res.matrix[i][j] = 0;
                for (int k = 0; k < 4; k++) {
                    res.matrix[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        return res;
    }

    geometry::OBB<S> transformHalfOf(DH<S> other, const S extent[3]) {
        // temp = this * half(other)
        DH<S> temp;
        for (int i = 0; i < 2; i++) {
            other.matrix[i][3] /= S(2.0);
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                temp.matrix[i][j] = 0;
                for (int k = 0; k < 4; k++) {
                    temp.matrix[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        // temp.print();

        geometry::OBB<S> res;
        for (int i = 0; i < 3; i++) {
            res.c[i] = temp.matrix[i][3];
            res.e[i] = extent[i];
            for (int j = 0; j < 3; j++) {
                res.u[i][j] = temp.matrix[i][j];
            }
        }

        return res;
    }

    void print() {
        std::cout << std::setprecision(2) << std::fixed;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

#endif
