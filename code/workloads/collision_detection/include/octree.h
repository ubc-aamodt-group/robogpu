#ifndef OCTREE_H
#define OCTREE_H

#include <stdint.h>

#include "geometry.h"

#define MAX_LEVEL 5
#define BIT(X, N) (((X) >> (N)) & 1)
#define POWER_OF_8(EXP) (1 << (3 * (EXP)))
#define CHILD(NODE, C) (((NODE) << 3) + (C) + 1)

namespace geometry {

/**
 * An Octree represents a 3D space divided into
 * layers of AABBs. The last level AABBs are voxels.
 **/
template <typename S>
class Octree {
   private:
    S origin[3] = {S(0.0), S(0.0),
                   S(0.0)};  // default center point for our 3D space

   public:
    uint32_t size;     // our 3D space is size x size x size units
    size_t num_nodes;  // number of nodes in tree
    AABB<S> *nodes;    // array storage for tree nodes

    /**
     * Given log_2(size), create a tree representing
     * a size x size x size 3D space
     */
    Octree(uint32_t log_2_size) : Octree(origin, log_2_size) {}

    /**
     * Given a center and log_2(size),
     * create a tree representing
     * a size x size x size 3D space
     */
    Octree(const S center[3], int log_2_size) {
        assert(log_2_size > 0);
        assert(log_2_size <= MAX_LEVEL);
        size = 1 << log_2_size;
        num_nodes = get_num_nodes(log_2_size);
        nodes = new AABB<S>[num_nodes];
        create(0, center, S(size >> 1));
    }

    ~Octree() { delete[] nodes; }

    /**
     * Using geometric series sum:
     * (8^(h+1) - 1) / (8 - 1)
     * to find out number of nodes 
     * in an octree with height h
     */
    size_t get_num_nodes(int height) {
        return (POWER_OF_8(height + 1) - 1) / 7;
    }

   private:
    /**
     * Recursively create the tree given
     * the current node (which is a cube)
     * the center of the current node
     * the side length of the current node
     */
    void create(uint32_t node, const S c[3], S e) {
        assert(node < num_nodes);
        AABB<S> &aabb = nodes[node];
        for (int i = 0; i < 3; i++) {
            aabb.c[i] = c[i];
            aabb.e[i] = e;
        }
        if (e == S(0.5)) {  // is this safe?
            return;
        }
        S new_c[3];
        S new_e = e / S(2.0);
        for (uint32_t i = 0; i < 8; i++) {
            for (int j = 0; j < 3; j++) {
                new_c[j] = c[j] + (BIT(i, j) ? -new_e : new_e);
            }
            create(CHILD(node, i), new_c, new_e);
        }
    }
};

}  // namespace geometry

#endif