# `_prepare_inputs` 函数详解

> 源文件：`vllm/v1/worker/gpu_model_runner.py` L1294–L1515

***

## 函数签名与返回值

```python
def _prepare_inputs(
    self,
    scheduler_output: SchedulerOutput,
    num_scheduled_tokens: np.ndarray,   # 每个请求本次调度的 token 数
) -> tuple[
    torch.Tensor,            # logits_indices：采样时从哪些位置取 logit
    SpecDecodeMetadata | None,
]:
```

**核心目标：** 将调度器输出转换为模型推理所需的所有 GPU 张量，并返回采样索引。

***

## "本次调度 token 数"含义

`num_scheduled_tokens[req_idx]` 表示**本次前向计算中该请求送入模型的 token 个数**：

| 请求阶段                         | 调度 token 数 | 具体内容                          |
| ---------------------------- | ---------- | ----------------------------- |
| **Chunked Prefill**          | N（分片大小）    | prompt 中本次要处理的那一片 token       |
| **Decode（无 Spec）**           | 1          | 上一步采样得到的新 token               |
| **Decode（有 Spec，K 个 draft）** | 1 + K      | 1 个真实 token + K 个 draft token |

***

## 示例场景

以下所有 Step 均基于同一个场景推导，`max_model_len = 2048`：

| req_idx | req_id | 已计算 token 数 | 本次调度 token 数 | 阶段说明 |
| --- | --- | --- | --- | --- |
| 0 | req-A | 0 | 4 | chunked prefill，prompt 共 10 个 token，本次处理前 4 个 |
| 1 | req-B | 100 | 1 | decode（无 spec），送入上一步采样的 1 个 token |
| 2 | req-C | 50 | 3 | decode（有 spec），1 个真实 token + 2 个 draft token |

    num_scheduled_tokens          = [4, 1, 3]
    total_num_scheduled_tokens    = 8
    num_reqs                      = 3

***

## 各步骤详解

### Step 1 — 异步提交 Block Table

**目的：** 尽早触发 block table 的 CPU→GPU 拷贝，让 GPU 拷贝与后续 CPU 计算并行执行（overlap），是整个函数最重要的性能优化。

**代码（L1314）：**

```python
self.input_batch.block_table.commit_block_table(num_reqs)
```

**示例：** 直接启动 3 个请求的 block table 异步拷贝，不等待结果，继续执行 CPU 侧计算。

***

### Step 2 — 构造批量索引

**目的：** 将"每个请求调度了多少 token"展开成两个扁平数组：`req_indices`（每个 token 对应哪个请求）和 `arange`（每个 token 在该请求内部的偏移），供后续向量化计算使用，无需 Python 循环。

**代码（L1318–1322）：**

```python
# 每个 token 对应的请求下标
req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

# 累积 token 数 和 每请求内部偏移
cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
```

`_get_cumsum_and_arange` 内部实现（L1128–1134）：

```python
cu_num_tokens   = np.cumsum(num_tokens)                              # Step1: 累积和
cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)  # Step2: 每个 token 所属请求的起始下标
arange          = self.arange_np[:total_num_tokens] - cumsums_offsets # Step3: 全局下标 - 起始下标 = 局部偏移
```

**示例数据变化：**

    num_scheduled_tokens = [4, 1, 3]

    req_indices = np.repeat([0,1,2], [4,1,3])
                = [0, 0, 0, 0,  1,  2, 2, 2]

    cu_num_tokens = np.cumsum([4, 1, 3])
                  = [4, 5, 8]

    # _get_cumsum_and_arange 内部：
    cu_num_tokens - num_tokens = [4-4, 5-1, 8-3] = [0, 4, 5]  # 每个请求在展平数组中的起始位置
    cumsums_offsets = np.repeat([0, 4, 5], [4, 1, 3])
                    = [0, 0, 0, 0,  4,  5, 5, 5]
    arange_np[:8]   = [0, 1, 2, 3,  4,  5, 6, 7]
    arange          = [0, 1, 2, 3,  4,  5, 6, 7]
                    - [0, 0, 0, 0,  4,  5, 5, 5]
                    = [0, 1, 2, 3,  0,  0, 1, 2]
    #                  ← req-A → req-B  ← req-C →

***

### Step 3 — 计算绝对位置

**目的：** 计算每个 token 在序列中的绝对位置（即 KV cache 中的偏移），作为 RoPE 位置编码的输入。

**代码（L1325–1330）：**

```python
positions_np = self.positions.np[:total_num_scheduled_tokens]
np.add(
    self.input_batch.num_computed_tokens_cpu[req_indices],  # 每个 token 所属请求已计算的 token 数
    arange,                                                  # 在本次调度中的内部偏移
    out=positions_np,
)
```

公式：`position = num_computed_tokens[req] + arange`

**示例数据变化：**

    num_computed_tokens_cpu = [0, 100, 50]

    num_computed_tokens_cpu[req_indices]
      = num_computed_tokens_cpu[[0,0,0,0, 1, 2,2,2]]
      = [0, 0, 0, 0,  100,  50, 50, 50]

    positions_np = [0, 0, 0, 0,  100,  50, 50, 50]
                 + [0, 1, 2, 3,    0,   0,  1,  2]
                 = [0, 1, 2, 3,  100,  50, 51, 52]
    #               ←  req-A  →  req-B  ← req-C →

含义：

*   **req-A**：处理 prompt 位置 0、1、2、3
*   **req-B**：处理位置 100（上一步采样的新 token）
*   **req-C**：处理位置 50（真实 token）、51、52（2 个 draft token）

***

### Step 4 — 特殊 RoPE 位置计算（按需）

**目的：** 多模态模型需要在多个维度上计算位置编码，普通文本模型跳过此步。

**代码（L1334–1340）：**

```python
if self.uses_mrope:
    self._calc_mrope_positions(scheduler_output)   # Qwen2-VL：时间/高/宽三维位置
elif self.uses_xdrope_dim > 0:
    self._calc_xdrope_positions(scheduler_output)  # HunYuan-VL：多维位置
# 普通文本模型：跳过，直接使用 Step 3 的 positions_np
```

**示例：** 本例为纯文本模型，跳过此步，`positions_np = [0,1,2,3, 100, 50,51,52]` 不变。

***

### Step 5 — 提取 Input Token IDs

**目的：** 从持久化的 `token_ids_cpu`（存放所有请求全量 token 序列的 2D 表）中，按 `positions_np` 指定的位置取出本次调度的 token ID，写入模型输入 buffer。

**代码（L1346–1359）：**

```python
# token_ids_cpu: shape [num_reqs, max_model_len]
# 将行列二维坐标转换为一维展平索引
token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
token_indices_tensor = torch.from_numpy(token_indices)

# 从展平的 token 表中批量取出，直接写入预分配的 input_ids buffer
torch.index_select(
    self.input_batch.token_ids_cpu_tensor.flatten(),      # 展平为 [num_reqs * max_model_len]
    0,
    token_indices_tensor,
    out=self.input_ids.cpu[:total_num_scheduled_tokens],  # 写入预分配 buffer，避免内存分配
)
```

> 用 `torch.index_select` 而非 `np.take`，是因为在大张量上前者利用了 PyTorch 的优化内存访问，速度显著更快。

**示例数据变化：**

`token_ids_cpu` 展平后，行 0 占索引 `[0, 2047]`，行 1 占 `[2048, 4095]`，行 2 占 `[4096, 6143]`：

    token_indices = positions_np + req_indices * 2048
                  = [0,1,2,3,        100,       50,  51,  52]
                  + [0,0,0,0,    1*2048,   2*2048,2*2048,2*2048]
                  = [0, 1, 2, 3,   2148,   4146, 4147, 4148]
    #               ← req-A →     req-B    ←    req-C    →

`torch.index_select` 用这 8 个索引取值，写入 `input_ids.cpu`：

    input_ids.cpu[:8] = [
      token_ids_cpu[0][0],   # req-A prompt 位置0
      token_ids_cpu[0][1],   # req-A prompt 位置1
      token_ids_cpu[0][2],   # req-A prompt 位置2
      token_ids_cpu[0][3],   # req-A prompt 位置3
      token_ids_cpu[1][100], # req-B decode token
      token_ids_cpu[2][50],  # req-C 真实 token
      token_ids_cpu[2][51],  # req-C draft token 1
      token_ids_cpu[2][52],  # req-C draft token 2
    ]

***

### Step 6 — 填充 Prompt Embeddings（可选）

**目的：** 若请求携带预计算的 embedding（如图像特征），将其拷贝到 `inputs_embeds` buffer 对应位置。与 token id 路径不同，embeds 没有预分配大型 CPU 张量，需要逐请求手动填入。

**代码（L1372–1405）：**

```python
for req_idx in range(num_reqs):
    if req_idx not in self.input_batch.req_prompt_embeds:
        continue
    req_embeds = self.input_batch.req_prompt_embeds[req_idx]
    start_pos  = self.input_batch.num_computed_tokens_cpu[req_idx]
    end_pos    = start_pos + num_scheduled_tokens[req_idx]
    self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
        req_embeds[start_pos:end_pos]
    )
```

**示例：** 本例 3 个请求均无 prompt embeddings，跳过。

***

### Step 7 — 计算 Slot Mapping

**目的：** 根据每个 token 的绝对位置和 block table，计算它在 KV cache 物理存储中的 slot 编号。Attention kernel 写入新 K/V 时依赖此映射。

**代码（L1407–1408）：**

```python
self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
```

**示例：** 假设 block size = 16，req-B 的 position 100 对应 block `100 // 16 = 6`，block 内偏移 `100 % 16 = 4`，最终 slot = `block_table[1][6] * 16 + 4`。

***

### Step 8 — 构建 Attention 辅助张量

**目的：** 为 FlashAttention 等 kernel 准备三个关键辅助张量，描述 batch 中每个请求的 query 范围、序列长度，以及哪些请求的采样结果需要丢弃。

**`query_start_loc`（L1411–1416）：**

```python
self.query_start_loc.np[0] = 0
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])  # 尾部非递减填充
self.query_start_loc.copy_to_gpu()
```

**`seq_lens`（L1419–1424）：**

```python
self.seq_lens.np[:num_reqs] = (
    self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
)
self.seq_lens.np[num_reqs:].fill(0)  # full cuda graph 模式下未用槽位填 0
self.seq_lens.copy_to_gpu()
```

**`discard_request_mask`（L1431–1434）：**

```python
num_tokens_np = np.array([self.requests[r].num_tokens for r in req_ids])
self.discard_request_mask.np[:num_reqs] = (self.seq_lens.np[:num_reqs] < num_tokens_np)
self.discard_request_mask.copy_to_gpu(num_reqs)
```

**示例数据变化：**

    cu_num_tokens = [4, 5, 8]

    query_start_loc.np = [0,  4,  5,  8,  8, ...]
                          ^   ^   ^   ^
                          |   |   |   req-C 结束
                          |   |   req-B 结束
                          |   req-A 结束
                          起始
    # 尾部填充 8（最大值），保证数组单调不减，FlashAttention kernel 安全访问

    seq_lens.np[:3] = [0+4,  100+1,  50+3]
                    = [4,    101,    53]

    # req-A: seq_lens=4 < prompt总长10 → 本轮采样结果需丢弃
    # req-B: seq_lens=101，decode完成 → 保留
    # req-C: seq_lens=53，decode完成 → 保留
    discard_request_mask[:3] = [True, False, False]

***

### Step 9 — 异步拷贝数据到 GPU

**目的：** 将 CPU 上准备好的 `input_ids` 和 `positions` 异步拷贝到 GPU，供模型前向使用。

**代码（L1437–1457）：**

```python
# input_ids：普通调度直接拷贝；异步调度时从 GPU 上的 prev_sampled_token_ids 直接 scatter
self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens, cu_num_tokens)

if self.uses_mrope:
    self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
        self.mrope_positions.cpu[:, :total_num_scheduled_tokens], non_blocking=True
    )
elif self.uses_xdrope_dim > 0:
    self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
        self.xdrope_positions.cpu[:, :total_num_scheduled_tokens], non_blocking=True
    )
else:
    self.positions.copy_to_gpu(total_num_scheduled_tokens)  # 标准 1D 位置
```

`_prepare_input_ids` 的两条路径：

| 场景                                | 处理方式                                                       |
| --------------------------------- | ---------------------------------------------------------- |
| 普通调度                              | `input_ids.copy_to_gpu()`，CPU→GPU 直接拷贝                     |
| 异步调度（`prev_sampled_token_ids` 非空） | 上一轮采样结果仍在 GPU，直接在 GPU 内 scatter，避免 GPU→CPU→GPU 的 roundtrip |

**示例数据变化：**

    # 普通调度路径：
    input_ids.gpu[:8]  ← input_ids.cpu[:8]  (异步拷贝)
    positions.gpu[:8]  ← [0,1,2,3, 100, 50,51,52]  (异步拷贝)

***

### Step 10 — 计算 `logits_indices`

**目的：** 确定从模型输出的 `hidden_states`（形状 `[total_tokens, hidden_dim]`）中取哪些行来计算 logits 并采样。无 spec decode 时每个请求只取最后一个 token；有 spec decode 时还需取所有 draft token 位置。

#### 无投机解码（L1466–1469）

**代码：**

```python
logits_indices = query_start_loc[1:] - 1
```

**示例数据变化：**

    query_start_loc[1:] = [4, 5, 8]
    logits_indices      = [4-1, 5-1, 8-1]
                        = [3, 4, 7]

    hidden_states: shape [8, hidden_dim]
    sample_hidden_states = hidden_states[[3, 4, 7]]  # shape [3, hidden_dim]
    logits = lm_head(sample_hidden_states)            # shape [3, vocab_size]

    # req-A → logits[0]：被 discard_mask 丢弃，不产生输出
    # req-B → logits[1]：直接采样下一个 token
    # req-C → logits[2]：... 但 req-C 有 spec decode，见下

#### 有投机解码（L1474–1500）

**代码：**

```python
num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
    req_idx = self.input_batch.req_id_to_index[req_id]
    num_draft_tokens[req_idx] = len(draft_token_ids)

spec_decode_metadata = self._calc_spec_decode_metadata(num_draft_tokens, cu_num_tokens)
logits_indices = spec_decode_metadata.logits_indices
```

`_calc_spec_decode_metadata` 内部（L1992–2069）：

```python
num_sampled_tokens = num_draft_tokens + 1
cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(num_sampled_tokens)
logits_indices = np.repeat(
    cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
) + arange
bonus_logits_indices  = cu_num_sampled_tokens - 1
target_logits_indices = ...  # draft token 对应的位置
```

**示例数据变化（req-C 有 2 个 draft token）：**

    num_draft_tokens   = [0,  0,  2]
    num_sampled_tokens = [1,  1,  3]  (= draft + 1 bonus)

    cu_num_scheduled_tokens = [4, 5, 8]
    cu_num_sampled_tokens   = cumsum([1,1,3]) = [1, 2, 5]

    # logits_indices 推导（从每个请求末尾取 num_sampled_tokens 个位置）：
    np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
      = np.repeat([4-1, 5-1, 8-3], [1,1,3])
      = np.repeat([3,   4,   5  ], [1,1,3])
      = [3,  4,  5, 5, 5]
    + arange of [1,1,3]
      = [0,  0,  0, 1, 2]
      -----------------
    logits_indices = [3,  4,  5, 6, 7]
    #                req-A req-B ← req-C →

    # bonus_logits_indices（每个请求最后一个位置，产生下一个真实 token）：
    bonus_logits_indices = cu_num_sampled_tokens - 1 = [0, 1, 4]
    #                       req-A  req-B  req-C

    # target_logits_indices（req-C 的 2 个 draft token 位置）：
    target_logits_indices = [2, 3]
    #                        draft1 draft2（在 logits 输出中的第 2、3 行）

最终采样时：

    hidden_states: shape [8, hidden_dim]
    sample_hidden_states = hidden_states[[3, 4, 5, 6, 7]]  # shape [5, hidden_dim]
    logits = lm_head(sample_hidden_states)                   # shape [5, vocab_size]

    logits[0] → req-A bonus：被 discard_mask 丢弃
    logits[1] → req-B bonus：直接采样下一个 token
    logits[2] → req-C draft token[51] 的 logit：rejection sampler 验证
    logits[3] → req-C draft token[52] 的 logit：rejection sampler 验证
    logits[4] → req-C bonus token：draft 全被拒绝时的回退

***

### Step 11 — Hot-Swap LoRA（可选）

**目的：** 若启用 LoRA，根据当前 batch 中各请求使用的适配器，动态切换模型中的 LoRA 权重层。

**代码（L1503–1510）：**

```python
if self.lora_config:
    self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
```

**示例：** 本例未启用 LoRA，跳过。

***

## 关键设计要点

| 设计点                      | 具体做法                                        | 收益                                      |
| ------------------------ | ------------------------------------------- | --------------------------------------- |
| **CPU-GPU overlap**      | Step 1 最先触发 block table 拷贝                  | GPU 拷贝与后续 CPU 计算并行                      |
| **向量化索引**                | `np.repeat + arange` 代替 Python 循环           | 消除逐请求循环开销                               |
| **`torch.index_select`** | 优于 `np.take` 提取 token ids                   | 大张量上速度显著更快                              |
| **异步 input\_ids**        | prev\_sampled\_token\_ids 在 GPU 上直接 scatter | 避免 decode token 的 GPU→CPU→GPU roundtrip |
| **非递减 padding**          | `query_start_loc` 尾部填充最大值                   | FlashAttention kernel 安全性               |
| **投机解码统一索引**             | `_calc_spec_decode_metadata` 向量化计算          | draft + bonus token 位置一次性确定             |

***

## 返回值说明

```python
return (
    logits_indices,       # torch.Tensor，指定从模型输出中取哪些行来做采样
    spec_decode_metadata, # SpecDecodeMetadata | None，包含投机解码验证所需的全部元数据
)
```

这两个值被传递给后续的 `_sample()` 和 `rejection_sampler()`，完成最终的 token 采样。
