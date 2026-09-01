## RoboGPU: Accelerating GPU Collision Detection for Robotics

***To appear in Int'l Symposium on Microarchitecture 2026***

### Authors: 
Lufei Liu, Liwei Xue, Yuan Hsi Chou, Jocelyn Zhao, Lara Kawasme, Youssef Mohammed, and Tor M. Aamodt

---

### Abstract:
Autonomous robots are anticipated to be deployed soon in domains ranging from transportation to healthcare and home assistance. Enabling autonomous robotics requires a computation platform flexible enough to execute a diverse and evolving workload while meeting real-time requirements. We believe a GPU-like architecture will be a key component of such platforms. Recent GPUs combine a flexible parallel processing fabric augmented with efficient support for important application domains via embedded accelerators (e.g., Tensor Cores), and a GPU-like architecture has reportedly been adopted for Tesla's upcoming AI5 accelerator. While current GPUs are effective at supporting emerging neural motion planners, we find that collision detection is crucial for evaluating their proposed trajectories and that this step appears to require dedicated acceleration to operate in real-time.
In this work, we propose RoboCore, an accelerator block embedded within a robotics-focused GPU (RoboGPU) architecture. We explore and compare architectural modifications to address the gaps of existing ray tracing accelerators (RTAs) for robotics and find that RoboCore computes collision queries 2.8x faster than RTA implementations using 48% less energy with 2% more area than RTAs. RoboCore is 13.3x faster than a CUDA baseline, and achieves 3.4x end-to-end speedup on a neural motion planner and 1.1x speedup on Monte Carlo Localization compared to a baseline GPU. This demonstrates that a hybrid approach of embedded specialization within a flexible general-purpose GPU architecture is suitable for supporting advancements in robotics.



### In this repository:

- `./docs`
    - Website contents
- `./presentation` 
    - *[Coming Soon]*
- `./code`
    - *[Coming Soon]*