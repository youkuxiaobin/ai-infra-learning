# &#x20;`_model_forward` 之后到函数结束

> 源文件：`vllm/v1/worker/gpu_model_runner.py`，L3229–L3299
>
> 这段代码在 `_model_forward` 返回后执行：将 model\_output 拆包、按 PP 拓扑路由隐状态、计算 logits，最后将所有中间结果打包存入 `execute_model_state`，然后返回 `None`（让调度器有机会做 CPU 端工作，之后再调用 `sample_tokens`）。

***

## 示例场景（贯穿所有 Step）

与 `_prepare_inputs` 文档使用相同批次：

| 请求    | 阶段               | token 数 | req\_idx |
| ----- | ---------------- | ------- | -------- |
| req-A | prefill          | 4       | 0        |
| req-B | decode（无 spec）   | 1       | 1        |
| req-C | spec decode（1+2） | 3       | 2        |

`_model_forward` 返回值：

    model_output = Tensor[8, 4096]   # 8 个输入 token，每个 hidden_size=4096
                                      # （单 GPU，非 EAGLE3，非 PP）

关键上下文变量（来自本轮 `execute_model` 前半段）：

    logits_indices  = tensor([3, 4, 5, 6, 7])
    # req-A 最后一个 prefill token(pos=3) + req-B(pos=4) + req-C 3个spec位置(pos=5,6,7)

    num_tokens_padded = 8     # 含 padding 后的 token 总数
    broadcast_pp_output = False   # 非 broadcast PP（常见路径）
    is_last_rank = True           # 单 GPU = 唯一也是最后一个 PP rank
    is_pooling_model = False      # 非 pooling 模型
    use_aux_hidden_state_outputs = False   # 非 EAGLE3

***

## Step 1 — 拆包 model\_output（L3229–L3236）

**目的：** `_model_forward` 可能返回单个 Tensor（普通 LLM）或 tuple（EAGLE3 返回主隐状态 + 辅助层隐状态）。此步统一解包，得到 `hidden_states` 和可选的 `aux_hidden_states`。

**代码（L3229–L3236）：**

```python
with record_function_or_nullcontext("gpu_model_runner: postprocess"):
    if self.use_aux_hidden_state_outputs:
        # EAGLE 3 专用：model 返回 (hidden, aux_list)
        hidden_states, aux_hidden_states = model_output
    else:
        # 常见路径
        hidden_states = model_output
        aux_hidden_states = None
```

**示例数据变化：**

    # use_aux_hidden_state_outputs = False（非 EAGLE3）

    model_output   = Tensor[8, 4096]   # _model_forward 的返回值

    hidden_states     = Tensor[8, 4096]  # shape: [num_tokens, hidden_size]
    aux_hidden_states = None

若为 EAGLE3（仅说明路径）：

    model_output = (Tensor[8,4096], [Tensor[8,4096], Tensor[8,4096]])
    hidden_states     = Tensor[8, 4096]
    aux_hidden_states = [Tensor[8,4096], Tensor[8,4096]]  # 中间层隐状态，供 drafter 使用

***

## Step 2 — 非 broadcast PP 路径（L3238–L3256）

**目的：** `broadcast_pp_output=False` 是常见路径（无 PP 或标准 PP）。根据当前 PP rank 的位置决定：非末尾 rank 直接返回中间张量；末尾 rank 计算 logits。

**代码（L3238–L3256）：**

```python
if not self.broadcast_pp_output:
    # Common case.
    if not get_pp_group().is_last_rank:
        # 非末尾 PP rank：返回中间隐状态，传给下一个 PP stage
        assert isinstance(hidden_states, IntermediateTensors)
        hidden_states.kv_connector_output = kv_connector_output
        self.kv_connector_output = kv_connector_output
        return hidden_states                   # ← 提前返回

    if self.is_pooling_model:
        # Pooling 模型（embedding 服务）：pooling 后直接返回
        output = self._pool(
            hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
        )
        output.kv_connector_output = kv_connector_output
        return output                          # ← 提前返回

    # 末尾 rank + 生成式模型（最常见）
    sample_hidden_states = hidden_states[logits_indices]
    logits = self.model.compute_logits(sample_hidden_states)
```

**示例数据变化（单 GPU = 唯一且是末尾 rank）：**

    broadcast_pp_output = False  → 进入此分支
    is_last_rank = True           → 跳过"提前返回"
    is_pooling_model = False      → 跳过 pooling 分支

    # 核心操作：按 logits_indices 切片隐状态
    logits_indices       = tensor([3, 4, 5, 6, 7])
    hidden_states.shape  = [8, 4096]

    sample_hidden_states = hidden_states[[3, 4, 5, 6, 7]]
    # → Tensor[5, 4096]
    # 行 0: req-A 最后一个 prefill token（pos=3）的隐状态
    # 行 1: req-B decode token（pos=4）
    # 行 2: req-C target token（pos=5）
    # 行 3: req-C draft[0]（pos=6）
    # 行 4: req-C draft[1]（pos=7）

    logits = self.model.compute_logits(sample_hidden_states)
    # → Tensor[5, 32000]
    # 对 5 个位置分别做 lm_head（Linear）映射到词表

**多 PP rank 路径（非末尾 rank，仅说明）：**

    is_last_rank = False
    hidden_states = IntermediateTensors(tensors={"hidden_states": Tensor[8,4096], ...})
    hidden_states.kv_connector_output = kv_connector_output
    → return hidden_states   # 传给 PP 下一级，本函数结束

***

## Step 3 — Broadcast PP 路径（L3257–L3285，罕见）

**目的：** 当 PP 与序列并行（SP）结合使用时，`broadcast_pp_output=True`。此时每个 PP rank 都先切片 `sample_hidden_states`，末尾 rank 计算 logits 后广播给所有其他 rank，确保所有 rank 拿到同一份 logits。

**代码（L3257–L3285）：**

```python
else:
    # Rare case: PP + SP broadcast
    assert not self.is_pooling_model

    sample_hidden_states = hidden_states[logits_indices]

    if not get_pp_group().is_last_rank:
        # 非末尾 rank：发送隐状态给末尾 rank（通过 TP all-gather）
        all_gather_tensors = {
            "residual": not is_residual_scattered_for_sp(
                self.vllm_config, num_tokens_padded
            )
        }
        get_pp_group().send_tensor_dict(
            hidden_states.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )
        logits = None                          # 非末尾 rank 无 logits
    else:
        logits = self.model.compute_logits(sample_hidden_states)  # 末尾 rank 计算

    # 末尾 rank 广播 logits 给所有 PP rank
    model_output_broadcast_data: dict[str, Any] = {}
    if logits is not None:
        model_output_broadcast_data["logits"] = logits.contiguous()

    broadcasted = get_pp_group().broadcast_tensor_dict(
        model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
    )
    assert broadcasted is not None
    logits = broadcasted["logits"]
```

**示例数据变化（此路径在示例场景中不走，仅说明）：**

    # 假设 PP=2（rank0, rank1），SP 启用，broadcast_pp_output=True

    # rank0（非末尾）：
    sample_hidden_states = hidden_states[[3,4,5,6,7]]  # Tensor[5,4096]
    get_pp_group().send_tensor_dict(hidden_states.tensors, ...)
    logits = None

    # rank1（末尾）：
    sample_hidden_states = hidden_states[[3,4,5,6,7]]
    logits = compute_logits(sample_hidden_states)       # Tensor[5,32000]
    model_output_broadcast_data = {"logits": Tensor[5,32000]}

    # broadcast（从 rank1 广播给 rank0）：
    broadcasted = {"logits": Tensor[5,32000]}
    logits = broadcasted["logits"]  # rank0 和 rank1 都拿到同一份

    # 结果：两个 rank 均有 logits = Tensor[5,32000]

***

## Step 4 — 打包 `ExecuteModelState`，返回 `None`（L3287–L3299）

**目的：** 将本轮计算的所有中间结果（logits、hidden\_states、spec\_decode\_metadata 等）打包到 `ExecuteModelState`，存入 `self.execute_model_state`。然后返回 `None`，让调度器线程有机会在 GPU 空闲时做 CPU 端工作（token scheduling、KV 管理等），之后再调用 `sample_tokens` 取出结果并采样。

**代码（L3287–L3299）：**

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,              # 调度信息（req列表、token数等）
    logits,                        # Tensor[5, 32000] — 待采样 logits
    spec_decode_metadata,          # SpecDecodeMetadata or None
    spec_decode_common_attn_metadata,  # spec decode 的 attn 元数据
    hidden_states,                 # Tensor[8, 4096] — 全部隐状态
    sample_hidden_states,          # Tensor[5, 4096] — 仅采样位置
    aux_hidden_states,             # None（非 EAGLE3）
    ec_connector_output,           # Embedding Cache Connector 输出
    cudagraph_stats,               # CUDA Graph 执行统计
)
self.kv_connector_output = kv_connector_output
return None
```

**示例数据变化：**

    self.execute_model_state = ExecuteModelState(
        scheduler_output             = <SchedulerOutput, 3 reqs>,
        logits                       = Tensor[5, 32000],   # GPU
        spec_decode_metadata         = SpecDecodeMetadata(
            target_logits_indices = tensor([1, 2, 3, 4]),  # req-B + req-C 3个
            bonus_logits_indices  = tensor([1]),            # req-B 的 bonus（无spec则自身）
            draft_token_ids       = [[600, 601]],           # req-C 的 2 个 draft
        ),
        spec_decode_common_attn_metadata = CommonAttentionMetadata(...),
        hidden_states                = Tensor[8, 4096],     # GPU
        sample_hidden_states         = Tensor[5, 4096],     # GPU
        aux_hidden_states            = None,
        ec_connector_output          = None,
        cudagraph_stats              = CUDAGraphStat{mode=FULL, ...},
    )
    self.kv_connector_output = None   # 无 KV 传输

    return None  # execute_model 结束
                 # 调度器获得控制权，可做 CPU 端调度
                 # 之后调用 sample_tokens() 完成采样

***

## 关键设计要点

| 设计点                            | 说明                                                                                  | 收益                                              |
| ------------------------------ | ----------------------------------------------------------------------------------- | ----------------------------------------------- |
| **`execute_model` 返回 `None`**  | 不返回采样结果，而是存入 `execute_model_state`；`sample_tokens` 是独立的第二个调用                        | GPU 计算与 CPU 调度解耦，支持异步调度                         |
| **`logits_indices` 切片**        | 只对"需要采样"的 token 位置（prefill 末位 + decode + spec）切出 `sample_hidden_states`，再过 lm\_head | 避免对所有 8 个 token 都做 lm\_head（词表大，开销高）            |
| **PP 非末尾 rank 提前 return**      | 非末尾 PP rank 直接返回 `IntermediateTensors`，不走 logits 计算                                 | 中间 stage 无需关心采样逻辑                               |
| **broadcast\_pp\_output（罕见）**  | PP + SP 组合时，末尾 rank 计算后广播给所有 rank，保证一致性                                             | 统一 all-rank logits，支持 SP 下的 PP 采样               |
| **EAGLE3 `aux_hidden_states`** | 多层中间隐状态随 `ExecuteModelState` 传递给 `sample_tokens`，再转给 drafter 作为 draft 输入            | drafter 利用更多层的特征提升 draft 质量                     |
| **`kv_connector_output` 双路存储** | 既存入 `execute_model_state`，也单独存 `self.kv_connector_output`（PP 非末尾 rank 提前 return 时用） | 保证 KV transfer 结果在各 PP 路径下都能传递给 `sample_tokens` |

