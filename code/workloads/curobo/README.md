# CuRobo

Our CuRobo experiments utilize the CuRobo collision detection APIs. Since CuRobo does not include OBB-AABB collisions, we treat OBBs and AABBs as cuboids in CuRobo and collisions are performed as cuboid-sphere tests with robot OBBs as spheres. 

CSV files of OBB/AABB pairs for each environment is included.

Profile using `python obb_aabb_cuboid.py`. 