# 并行编程导论于 CUDA 入门

> 由 PDF 转为 Markdown 的清洗版。已尽量保留原始结构、代码块与示意图。

## 1. 并行 VS 并发

- **串行（serial）**：一件事做完再做下一件事
- **并发（concurrent）**：多件事“交错推进”，看起来同时（常见于 CPU 多任务 / IO）
- **并行（parallel）**：多件事“同一时刻同时做”（典型：多核 CPU、GPU 大规模线程）

## 2. CPU vs GPU：为什么深度学习 / 仿真会偏爱 GPU

- **CPU**：少量复杂核心、控制流强、低延迟，像“全能战士”
- **GPU**：大量相对简单核心、吞吐优先、适合数据并行，像“流水线工人”

这解释了：为什么矩阵乘、卷积、向量加法这种“对很多元素做同样操作”的任务更适合 GPU。

## 3. CUDA 编程模型

CUDA 的核心是：

- **Host（CPU）** 负责组织与调度
- **Device（GPU）** 负责并行计算

典型运行流程：

1. CPU 准备数据
2. Host -> Device 拷贝（HtoD）
3. GPU 从全局内存读数据 -> 计算 -> 写回全局内存
4. Device -> Host 拷贝（DtoH）
5. CPU 验证 / 收尾

性能评估通常要把工时间拆成：

```text
T_total = T_H2D + T_kernel + T_D2H
```

很多时候拷贝才是大头。

## 4. 线程组织：Grid / Block / Thread

线程层级结构：

```text
Grid -> Block -> Thread
```

- 一次 kernel launch = 一个 Grid
- Grid 里面有很多个 Block
- Block 里面有很多个 Thread

## 5. 一维索引：idx 计算公式

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

CUDA 内置变量（1D）：

- `blockIdx.x`：当前 block 编号
- `blockDim.x`：每个 block 的线程数
- `threadIdx.x`：线程在 block 内的编号

## vector add

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cassert>
using namespace std;

#define CUDA_CHECK(call) do {                                  \
  cudaError_t err = (call);                                    \
  if (err != cudaSuccess) {                                    \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(err));      \
    std::exit(1);                                              \
  }                                                            \
} while (0)

template<class T>
__global__ void add(const T* a, const T* b, T* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char ** argv) {
    int size = 1<<20;
    int bytes = sizeof(float) * size;
    vector<float> a, b, c(size);
    for (int i = 0; i < size; i++) {
        a.push_back(1.0f);
        b.push_back(2.0f);
    }

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));

    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(dc, c, bytes, cudaMemcpyHostToDevice));

    int thread_num = 1024;
    int block_num = (size + thread_num - 1) / thread_num;

    add<float><<<block_num, thread_num>>>(da, db, dc, size);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    return 0;
}
```

编译运行：

```bash
nvcc add.cu -O2 -o add
./add
```

## 6. kernel launch 开销：为啥 `<<<1, 1>>>` 可能更慢？

- kernel 启动本身有固定开销
- 线程太少会导致“并行收益 < launch 成本”
- 外部循环反复 launch kernel，开销巨大

不要这么写：

```cpp
for (int iter = 0; iter < 100000; ++iter) {
    vec_add<<<blocks, threads>>>(...);
}
```

## 7. 标准解法：grid-stride loop（把循环放进 kernel）

如果开启的总线程小于数据，那么这种做法也能处理所有数据。

```cpp
__global__ void vec_add_stride(const float* a, const float* b, float* c, int n) {
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}
```

解释：

```text
gridDim.x * blockDim.x = “这次 launch 的总线程数”
i += step              = “线程按固定步长分片处理更多元素”
```

好处：

- 减少 launch 次数
- 提高吞吐

## 8. 设备上限：block / thread 不是想设多少就设多少

`block size` / `grid size` 受硬件限制，需要看 `cudaDeviceProp`。

```cpp
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU: %s\n", prop.name);
    printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}
```

## 9. 调试：`cuda-gdb` + 编译参数 `-g -G`

用 CUDA-GDB 调试，并强调编译要加 `-g -G`：

- `-g`：Host 端调试信息
- `-G`：Device 端调试信息

```bash
nvcc add.cu -g -G -o add_dbg
cuda-gdb ./add_dbg
```

## 10. 性能观测：`nvidia-smi` / `nsys`

- `nvidia-smi`：看 GPU utilization，确认你不是“只在 GPU 跑”
- `nsys`：能把 HtoD、kernel、DtoH 分段看清楚

示例命令：

```bash
root@autodl-container-k5xt4fu27v-e7b3f9b5:~/cuda/infini/01# nsys stats add_cuda.nsys-rep
Generating SQLite file add_cuda.sqlite from add_cuda.nsys-rep
```

### `nsys stats` 输出示例

```text
Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/nvtx_sum.py]...
SKIPPED: add_cuda.sqlite does not contain NV Tools Extension (NVTX) data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/osrt_sum.py]...

** OS Runtime Summary (osrt_sum):

Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)  Name
--------  ---------------  ---------  ----------  ---------  --------  ---------   -----------  ----------------------
73.4      453336103        12         37778008.6  7852142.5  1524      336278292   95608923.7   poll
23.2      143038901        539        265378.3    20129.0    1177      32126055    1493735.1    ioctl
2.9       17670489         74         238790.4    3280.0     1916      17316193    2012405.8    fopen
0.3       2115042          25         84601.7     7187.0     3663      1552698     307639.3     mmap64
0.1       638482           9          70942.4     52081.0    36069     199634      54409.6      sem_timedwait
0.0       285566           43         6641.1      6055.0     2983      12727       2318.8       open64
0.0       143020           3          47673.3     49121.0    36867     57032       10160.1      pthread_create
0.0       139078           13         10698.3     3645.0     1676      78446       20657.0      mmap
0.0       57475            30         1915.8      1600.5     1064      4209        865.3        fclose
0.0       52809            1          52809.0     52809.0    52809     52809       0.0          fgets
0.0       36207            13         2785.2      1969.0     1157      5599        1559.4       read
0.0       35228            6          5871.3      6298.0     2698      9223        2371.0       open
0.0       21375            2          10687.5     10687.5    7543      13832       4447.0       socket
0.0       20704            10         2070.4      2138.0     1220      3080        688.6        write
0.0       18868            3          6289.3      3660.0     2877      12331       5246.9       munmap
0.0       17369            3          5789.7      6820.0     2990      7559        2452.6       pipe2
0.0       11253            1          11253.0     11253.0    11253     11253       0.0          connect
0.0       11091            3          3697.0      4435.0     1075      5581        2341.9       fread
0.0       11078            3          3692.7      4354.0     2129      4595        1359.5       pthread_cond_broadcast
0.0       7285             2          3642.5      3642.5     3233      4052        579.1        fwrite
0.0       4543             2          2271.5      2271.5     1049      3494        1728.9       fcntl
0.0       3301             1          3301.0      3301.0     3301      3301        0.0          bind
0.0       2066             1          2066.0      2066.0     2066      2066        0.0          listen

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_api_sum.py]...

** CUDA API Summary (cuda_api_sum):

Time (%)  Total Time (ns)  Num Calls  Avg (ns)    Med (ns)  Min (ns)  Max (ns)    StdDev (ns)   Name
--------  ---------------  ---------  ----------  --------  --------  ----------  ------------   ---------------------
99.9      136776274        3          45592091.3  3472.0    2280      136770522   78962837.2    cudaMalloc
0.1       101039           3          33679.7     6147.0    2763      92129       50646.9       cudaFree
0.0       44824            3          14941.3     15317.0   5092      24415       9667.0        cudaMemcpy
0.0       43672            1          43672.0     43672.0   43672     43672       0.0           cudaLaunchKernel
0.0       11763            1          11763.0     11763.0   11763     11763       0.0           cudaDeviceSynchronize

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_gpu_kern_sum.py]...

** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name
--------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------------------------------------------
100.0     1600             1          1600.0    1600.0    1600      1600      0.0          add_func(float *, float *, float *, unsigned long)

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_gpu_mem_time_sum.py]...

** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Operation
--------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
52.3      1792             1      1792.0    1792.0    1792      1792      0.0          [CUDA memcpy Device-to-Host]
47.7      1632             2      816.0     816.0     672       960       203.6        [CUDA memcpy Host-to-Device]

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_gpu_mem_size_sum.py]...

** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)  Operation
----------  -----  --------  --------  --------  --------  -----------  ----------------------------
0.008       2      0.004     0.004     0.004     0.004     0.000        [CUDA memcpy Host-to-Device]
0.004       1      0.004     0.004     0.004     0.004     0.000        [CUDA memcpy Device-to-Host]

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/openmp_sum.py]...
SKIPPED: add_cuda.sqlite does not contain OpenMP event data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/opengl_khr_range_sum.py]...
SKIPPED: add_cuda.sqlite does not contain KHR Extension (KHR_DEBUG) data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/opengl_khr_gpu_range_sum.py]...
SKIPPED: add_cuda.sqlite does not contain GPU KHR Extension (KHR_DEBUG) data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/vulkan_marker_sum.py]...
SKIPPED: add_cuda.sqlite does not contain Vulkan Debug Extension (Vulkan Debug Util) data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/vulkan_gpu_marker_sum.py]...
SKIPPED: add_cuda.sqlite does not contain GPU Vulkan Debug Extension (GPU Vulkan Debug markers) data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/dx11_pix_sum.py]...
SKIPPED: add_cuda.sqlite does not contain DX11 CPU debug markers.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/dx12_gpu_marker_sum.py]...
SKIPPED: add_cuda.sqlite does not contain DX12 GPU debug markers.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/dx12_pix_sum.py]...
SKIPPED: add_cuda.sqlite does not contain DX12 CPU debug markers.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/wddm_queue_sum.py]...
SKIPPED: add_cuda.sqlite does not contain WDDM context data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/um_sum.py]...
SKIPPED: add_cuda.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/um_total_sum.py]...
SKIPPED: add_cuda.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/um_cpu_page_faults_sum.py]...
SKIPPED: add_cuda.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/openacc_sum.py]...
SKIPPED: add_cuda.sqlite does not contain OpenACC event data.

Processing [add_cuda.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/syscall_sum.py]...
SKIPPED: add_cuda.sqlite does not contain syscall data.

root@autodl-container-k5xt4fu27v-e7b3f9b5:~/cuda/infini/01# ls
add  add.cu  add_cuda.nsys-rep  add_cuda.sqlite
```

## 11. 图

### CUDA 执行流程

```text
CPU
│
│ HtoD
▼
GPU global memory
│
▼
CUDA Kernel
│
▼
GPU memory
│
│ DtoH
▼
CPU
```

### CUDA 线程模型

```text
Grid
├── Block0
│   ├── Thread0
│   ├── Thread1
│   └── Thread2
│
├── Block1
│   ├── Thread0
│   ├── Thread1
│   └── Thread2
```

### grid-stride loop

```text
Thread0 -> 0, 512, 1024 ...
Thread1 -> 1, 513, 1025 ...
Thread2 -> 2, 514, 1026 ...
```
