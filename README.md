# AI Infra Learning

深入学习 AI 基础设施核心技术的笔记与分析，涵盖 **CUDA 并行编程** 与 **vLLM 推理引擎**源码解析。

---

## 目录结构

```
ai-infra-learning/
├── cuda/
│   └── initTensor/          # CUDA 并行编程基础
│       ├── 01_并行编程导论于CUDA_入门.md
│       ├── 02_性能模型与逐元素优化.md
│       ├── 03_内存模型与规约优化.md
│       └── 04_分块和不规则访存.md
└── vllm/
    └── model_executor/      # vLLM v1 GPU 模型执行流程分析
        ├── 00 - execute_model 流程分析.md
        ├── 01 - update_states.md
        ├── 02 - _prepare_inputs 函数详解.md
        ├── 03 - _determine_batch_execution_and_padding.md
        ├── 04 - maybe_create_ubatch_slices.md
        ├── 05 - _build_attention_metadata.md
        ├── 06 - _preprocess.md
        ├── 07 - model_forward.md
        └── 08 - postprocess.md
```

---

## CUDA 并行编程

从零开始系统学习 CUDA GPU 编程，涵盖性能建模与关键优化技术。

| 文档 | 内容 |
|------|------|
| [01 - CUDA 入门](cuda/initTensor/01_并行编程导论于CUDA_入门.md) | CPU vs GPU 架构、Host-Device 模型、Grid/Block/Thread 层级、向量加法示例、grid-stride loop |
| [02 - 性能模型与逐元素优化](cuda/initTensor/02_性能模型与逐元素优化.md) | Roofline 模型、计算强度（AI = FLOPs/Bytes）、内存合并、向量化（float4 / half2） |
| [03 - 内存模型与规约优化](cuda/initTensor/03_内存模型与规约优化.md) | atomicAdd、Warp 级规约、共享内存树形规约、寄存器→共享内存→L1/L2 层级 |
| [04 - 分块和不规则访存](cuda/initTensor/04_分块和不规则访存.md) | GEMM 优化、访存模式分析、Warp 内存合并、stride 模式对带宽的影响 |

**核心知识点：**
- CUDA 内核（`__global__`）启动与线程索引计算
- 性能瓶颈定位：访存密集型 vs 计算密集型
- 使用 `nvidia-smi`、`nsys`、`ncu` 进行性能分析
- 半精度（FP16）与向量化操作提升内存带宽利用率

---

## vLLM Model Executor 源码分析

对 **vLLM v1** `GPUModelRunner` 的完整 GPU 执行流水线进行逐函数深度解析。

### 执行流水线总览

```
execute_model()
    │
    ├── 1. update_states()              # 同步调度输出到 CPU 侧请求缓存
    ├── 2. _prepare_inputs()            # 构建 GPU 张量（token ids、positions、slot mapping）
    ├── 3. _determine_batch_execution_and_padding()  # 决定 CUDA Graph 执行模式
    ├── 4. maybe_create_ubatch_slices() # 微批（ubatch）切片（动态批处理优化）
    ├── 5. _build_attention_metadata()  # 构建 KV Cache 注意力元数据
    ├── 6. _preprocess()                # VLM 视觉编码 / prompt embeds / 纯文本三路处理
    ├── 7. model_forward()              # Transformer 前向计算（支持 CUDA Graph）
    └── 8. postprocess()                # Logits 提取 → 采样 → 输出构造
```

| 文档 | 内容 |
|------|------|
| [00 - 流程总览](vllm/model_executor/00%20-%20execute_model%20流程分析.md) | 11 阶段完整流水线、异步调度模式、ExecuteModelState 数据流 |
| [01 - update_states](vllm/model_executor/01%20-%20update_states.md) | 双层状态管理（`self.requests` dict + `InputBatch` 持久化缓冲区） |
| [02 - _prepare_inputs](vllm/model_executor/02%20-_prepare_inputs%20函数详解.md) | 7 步张量构建、异步块表提交、多维 RoPE（Qwen2-VL）、投机解码元数据 |
| [03 - batch 执行模式判断](vllm/model_executor/03%20-%20_determine_batch_execution_and_padding.md) | CUDA Graph FULL/PARTIAL/NONE 模式、DP 多实例协调 |
| [04 - ubatch 切片](vllm/model_executor/04%20-%20maybe_create_ubatch_slices.md) | 动态批处理微批优化（DBO） |
| [05 - attention metadata](vllm/model_executor/05%20-%20_build_attention_metadata.md) | query_start_loc、seq_lens、block_table、Cascade Attention 前缀共享 |
| [06 - _preprocess](vllm/model_executor/06%20-%20_preprocess.md) | VLM 视觉编码、Pipeline Parallel 分级处理 |
| [07 - model_forward](vllm/model_executor/07%20-%20model_forward.md) | `set_forward_context` 元数据注入、KV Connector、CUDA Graph 执行 |
| [08 - postprocess](vllm/model_executor/08%20-%20postprocess.md) | Logits 提取、Pipeline Parallel broadcast、Pooling 模型处理 |

**核心工程技术：**
- **持久化批处理（Persistent Batch）**：跨步复用 CPU 缓冲区，消除分配开销
- **CPU-GPU 重叠**：块表异步提交与 CPU 计算并行
- **CUDA Graph**：预捕获执行图，消除 decode 阶段的 kernel 启动开销
- **投机解码（Speculative Decoding）**：草稿 token 生成 + 拒绝采样验证
- **多模态支持（VLM）**：视觉编码器集成与多维位置编码
- **流水线并行（PP）**：跨设备张量广播与分级处理

---

## 技术栈

- **CUDA C++** — 设备内核、共享内存、CUDA Runtime API
- **PyTorch** — Tensor 操作、`index_select`、半精度计算
- **vLLM v1** — GPUModelRunner、Scheduler、KV Cache 管理
- **性能工具** — `nvidia-smi`、`nsys`（Nsight Systems）、`ncu`（Nsight Compute）

---

## 学习路径

```
CUDA 基础
  └── 并行编程入门 → 性能建模 → 规约优化 → GEMM 访存优化
        │
        ▼
vLLM 推理引擎
  └── 整体流程 → 状态管理 → 张量准备 → 注意力元数据 → 前向计算 → 后处理
```

适合有 Python/PyTorch 基础、希望深入理解 **GPU 推理系统底层实现**的工程师和研究者。
