#ifndef ENTRY_H
#define ENTRY_H

#include "geometry.h"

template <typename S>
struct entry {
    int child_address;
    int child_status[8];
    struct geometry::AABB<S> children_box[8];
    int level;
};

#endif