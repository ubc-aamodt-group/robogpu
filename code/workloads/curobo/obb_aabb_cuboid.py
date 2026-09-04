import csv
import time
from collections import defaultdict
 
import numpy as np
import torch
from scipy.spatial.transform import Rotation
 
from curobo.geom.types import WorldConfig, Cuboid
from curobo.geom.sdf.world import WorldCollisionConfig, WorldPrimitiveCollision, CollisionQueryBuffer
from curobo.types.base import TensorDeviceType
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import matplotlib.pyplot as plt
# set numpy random seed
np.random.seed(0)

# ─── AABB CSV -> CuRobo Cuboids 
# Each row is an axis-aligned box.
# Cuboid pose = [cx, cy, cz, qw, qx, qy, qz] 

def load_aabb_csv(csv_path):
    cuboids = []
    with open(csv_path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            cx = float(row['center_x'])
            cy = float(row['center_y'])
            cz = float(row['center_z'])
            hx = float(row['half_length_x'])
            hy = float(row['half_length_y'])
            hz = float(row['half_length_z'])
            cuboids.append(Cuboid(
                name=f"aabb_{i}",
                pose=[cx, cy, cz, 1.0, 0.0, 0.0, 0.0], 
                dims=[2*hx, 2*hy, 2*hz], 
            ))
    return WorldConfig(cuboid=cuboids)


# ─── OBB CSV -> Bounding Spheres ──────────────────────────────────────────────
# Each OBB becomes a sphere

def obb_row_to_cuboid(row, idx):
    cx, cy, cz = float(row['obb_center_x']), float(row['obb_center_y']), float(row['obb_center_z'])
    hx, hy, hz = float(row['obb_half_length_x']), float(row['obb_half_length_y']), float(row['obb_half_length_z'])

    # build rotation matrix from the 3 axis columns
    R = np.array([
        [float(row['obb_axis_x1']), float(row['obb_axis_x2']), float(row['obb_axis_x3'])],
        [float(row['obb_axis_y1']), float(row['obb_axis_y2']), float(row['obb_axis_y3'])],
        [float(row['obb_axis_z1']), float(row['obb_axis_z2']), float(row['obb_axis_z3'])],
    ])
     # Force right-handed: flip the third axis column if the matrix is mirrored.
    # Same fix used in compute_obb() for the PCA-based OBB.
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
 
    qw, qx, qy, qz = Rotation.from_matrix(R).as_quat(scalar_first=True)      # scipy gives [qx,qy,qz,qw]

    return Cuboid(
        name=f"obb_{idx}",
        pose=[cx, cy, cz, qw, qx, qy, qz],
        dims=[2*hx, 2*hy, 2*hz],
    )

def load_obb_csv_as_sphere_tensor(csv_path, tensor_args, n_spheres=1):
    """
    For each OBB row: build a Cuboid, fit n_spheres bounding spheres to it,
    then collect everything into a [n_configs, 1, n_joints*n_spheres, 4] tensor.
    """
    configs = defaultdict(list)  # pos -> list of [x, y, z, radius]
 
    with open(csv_path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            pos = int(row['pos'])
            cuboid = obb_row_to_cuboid(row, i)
            spheres = cuboid.get_bounding_spheres(n_spheres=n_spheres)
 
            # Sphere has .pose = [x, y, z, qw, qx, qy, qz] and .radius
            for sph in spheres:
                x, y, z = sph.pose[0], sph.pose[1], sph.pose[2]
                configs[pos].append([x, y, z, sph.radius])
 
    n_configs = max(configs.keys()) + 1
    n_spheres_total = len(configs[0])
    data = np.zeros((n_configs, 1, n_spheres_total, 4), dtype=np.float32)
    for pos, spheres in configs.items():
        data[pos, 0] = spheres
 
    return torch.tensor(data, device=tensor_args.device, dtype=tensor_args.dtype)
 
 

# ─── Set up the collision checker ────────────────────────────────────────────
 
def build_world_checker(aabb_csv, tensor_args):
    world_config = load_aabb_csv(aabb_csv)
    col_config = WorldCollisionConfig(tensor_args, world_model=world_config)
    return WorldPrimitiveCollision(col_config)
 
# ─── Benchmark: sweep batch sizes with CUDA graph replay ─────────────────────
 
def bench_collision_obb(world_ccheck, all_spheres, tensor_args, b_size, n, use_cuda_graph=True):
    """
    all_spheres: [n_configs, 1, n_joints, 4] tensor of all available configs
    b_size:      how many configs to check per batch
    n:           number of timed iterations
    """
    n_configs = all_spheres.shape[0]
    if b_size > n_configs:
        return 0
 
    weight = tensor_args.to_device([1.0])
    act_distance = tensor_args.to_device([0.0])
 
    if not use_cuda_graph:
        idx = np.random.choice(n_configs, b_size, replace=False)
        x_sph = all_spheres[idx].clone()
        query_buffer = CollisionQueryBuffer.initialize_from_shape(
            x_sph.shape, tensor_args, world_ccheck.collision_types
        )
 
        for _ in range(10):
            out = world_ccheck.get_sphere_distance(x_sph, query_buffer, weight, act_distance)
            torch.cuda.synchronize()
 
        dt = []
        for _ in range(n):
            idx = np.random.choice(n_configs, b_size, replace=False)
            x_sph.copy_(all_spheres[idx])
            torch.cuda.synchronize()
            st = time.time()
            out = world_ccheck.get_sphere_distance(x_sph, query_buffer, weight, act_distance)
            torch.cuda.synchronize()
            dt.append(time.time() - st)
    else:
        idx = np.random.choice(n_configs, b_size, replace=False)
        x_sph = all_spheres[idx].clone()
        query_buffer = CollisionQueryBuffer.initialize_from_shape(
            x_sph.shape, tensor_args, world_ccheck.collision_types
        )
 
        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = world_ccheck.get_sphere_distance(x_sph, query_buffer, weight, act_distance)
        torch.cuda.current_stream().wait_stream(s)
 
        with torch.cuda.graph(g):
            out = world_ccheck.get_sphere_distance(x_sph, query_buffer, weight, act_distance)
 
        dt = []
        for _ in range(n):
            idx = np.random.choice(n_configs, b_size, replace=False)
            x_sph.copy_(all_spheres[idx])
 
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
 
            t0 = time.time()
            g.replay()
            torch.cuda.synchronize()
            t1 = time.time()
            dt.append(t1 - t0)
 
        torch.cuda.synchronize()
 
    mean_us = np.mean(dt) * 1e6
    std_us  = np.std(dt) * 1e6
    print(f"  b_size={b_size:5d}  mean={mean_us:9.2f} us  std={std_us:8.2f} us")
    return np.mean(dt)
 
 
# ─── Save results to CSV ─────────────────────────────────────────────────────
 
def save_results_csv(env_name, results, out_dir="."):
    path = f"{out_dir}/curobo_{env_name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Batch Size", "Collision Checking"])
        for b, t in zip(results["Batch Size"], results["Collision Checking"]):
            writer.writerow([b, t])
    print(f"  wrote {path}")
    return path
 
 # ─── Plot: time per trajectory vs batch size, one line per environment ──────
 
def plot_results(all_results, out_path="curobo_latency.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
 
    for env_name, results in all_results.items():
        b = np.array(results["Batch Size"])
        t_us = np.array(results["Collision Checking"]) * 1e6  # seconds -> microseconds
        t_per_traj = t_us / b  # match Julia plot's "Time [us / trajectory]"
 
        order = np.argsort(b)
        ax.plot(b[order], t_per_traj[order], marker="o", markersize=4, label=env_name)
 
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Trajectories per Batch")
    ax.set_ylabel("Time [µs / trajectory]")
    ax.set_title("CuRobo Collision Checking Latency")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Environment")
 
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    return fig
 


if __name__ == "__main__":
    environments = {
        "Cubby":        ("1cubby_point_cloud_aabb.csv", "1cubby_obbs.csv"),
        "Dresser":      ("2dresser_point_cloud_aabb.csv", "2dresser_obbs.csv"),
        "MergedCubby":  ("3merged_cubby_point_cloud_aabb.csv", "3merged_cubby_obbs.csv"),
        "Tabletop":     ("4tabletop_point_cloud_aabb.csv", "4tabletop_obbs.csv"),
    }
    
    tensor_args = TensorDeviceType()
    
    all_results = {}

    for env_name, (aabb_csv, obb_csv) in environments.items():
        print("Loading environment AABBs...")
        world_ccheck = build_world_checker(aabb_csv, tensor_args)
    
        print("Loading robot OBB spheres...")
        all_spheres = load_obb_csv_as_sphere_tensor(obb_csv, tensor_args)
        print(f"  spheres tensor shape: {all_spheres.shape}  (n_configs, horizon, n_joints, 4)")
    
        b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        n_list = [4000, 4000, 1500, 500, 500, 250, 250, 125, 125]
        b_list = [b for b in b_list if b <= all_spheres.shape[0]]
        n_list = n_list[:len(b_list)]
        b_list = b_list[::-1]
        n_list = n_list[::-1]
    
        results = {"Batch Size": [], "Collision Checking": []}
        print("\nRunning sweep (CUDA graph on)...")
        for n, b_size in zip(n_list, b_list):
            dt = bench_collision_obb(world_ccheck, all_spheres, tensor_args, b_size, n, use_cuda_graph=True)
            results["Batch Size"].append(b_size)
            results["Collision Checking"].append(float(dt))
        
        all_results[env_name] = results
        save_results_csv(env_name, results)

    plot_results(all_results)
    print("\nDone.")