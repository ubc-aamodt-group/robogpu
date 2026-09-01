# RoboGPU Collision Detection

*This code is modified from "Energy-Efficient Realtime Motion Planning" (ISCA 2023) - Deval Shah, Ningfeng Yang, and Tor M. Aamodt. Please refer to their original README for more information.*

### Compile
Compile with included Makefile. Use `make` for the baseline binary and `make rt` for the RoboCore binary.

### Execute
The baseline is compiled to the `main` binary with regular GPU CUDA execution.
The other `rtmain` binary offloads collision detection to RoboCores modelled in the simulator. 

Use `./<binary> 1 <max-AABB-intersection-count> <octree-selection>` to run (ex. `./main 1 1024 1` to run 1024 tests using octree #1 - Cubby environment).

Available environments:
- `1`-`4` M$\pi$Net environments (Cubby, Dresser, Merged Cubby, Tabletop)
- `5`-`205` MPNet environments
- `600`-`613` Alpamayo Physical AI AV environments