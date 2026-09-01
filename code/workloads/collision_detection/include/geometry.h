#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <iostream>

#define DOT(T, A1, A2, B1, B2)                         \
    (T(A1[2] A2) * T(B1[2] B2) + T(A1[1] A2) * T(B1[1] B2) + \
     T(A1[0] A2) * T(B1[0] B2))
#define SUB(T, A, B) {T(A[0]) - T(B[0]), T(A[1]) - T(B[1]), T(A[2]) - T(B[2])}
#define ADD(T, A, B) {T(A[0]) + T(B[0]), T(A[1]) + T(B[1]), T(A[2]) + T(B[2])}
#define COPY(T, X) {T(X[0]), T(X[1]), T(X[2])}

namespace geometry {

template <typename S>
inline int obbDisjoint(const S B[3][3], const S T[3], const S a[3],
                        const S b[3]);

template <typename S>
struct AABB {
    S c[3];  // AABB center point
    S e[3];  // Positive halfwidth extents of AABB along each axis
};

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
}  // namespace geometry
#endif