
#include <chrono>
#include <iostream>
#include <string>

#include "fpm/fixed.hpp"
#include "fpm/ios.hpp"
#include "geometry.h"
#include "octree.h"

template <>
float geometry::OBB<float>::reps = 1e-6f;
template <>
double geometry::OBB<double>::reps = 1e-6f;

// void run_cpu(int option, unsigned int num_obbs);
// void run_cpu2(unsigned int num_obbs, int sram_idx);
void run_gpu(unsigned int num_obbs, int sram_idx, bool opt = true);
void run_gpu2(unsigned int num_obbs, int sram_idx);
using namespace std::chrono;

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: \"%s <gpu alg number (1)> <number: AABB intersection count> <octree selection>\"\n", argv[0]);
        return 1;
    }

    int sram_idx = std::stoi(argv[3]);
    printf("octree index: %d\n", sram_idx);

    int aabb_count = std::stoi(argv[2]);
    // printf("aabb intersection count: %d\n", aabb_count);

    switch (argv[1][0]) {
        // case '5': {
        //     for(unsigned int i = 0; i < 20; i++){
        //         run_cpu(1, 4 << i);
        //     }
        //     break;
        // }

        // case '6': {
        //     for(unsigned int i = 0; i < 20; i++){
        //         run_cpu2(4 << i, sram_idx);
        //     }
        //     break;
        // }
        case '1': {
            run_gpu(aabb_count, sram_idx, 0); // scales logrithmically with number of AABBs
            // for(unsigned int i = 0; i < 20; i++){
            //     run_gpu(4 << i);
            // }
            break;
        }
        case '2': {
            run_gpu(aabb_count, sram_idx, 1);
            // for(unsigned int i = 0; i < 20; i++){
            //     run_gpu(4 << i);
            // }
            break;
        }
        case '3': {
            run_gpu2(aabb_count, sram_idx); // scales linearly with number of AABBs and OBBs
            // for(unsigned int i = 0; i < 20; i++){
            //     run_gpu2((4 << i));
            // }
            break;
        }
        case 'b': {
            for(unsigned int i = 0; i < 20; i++){
                run_gpu(4 << i, false);
            }
            break;
        }

        default:
            fprintf(stderr, "Usage: \"%s <number>\"\n", argv[0]);
            return 1;
    }
    return 0;
}
