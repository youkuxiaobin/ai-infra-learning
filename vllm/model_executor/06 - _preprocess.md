# `_preprocess` 函数详解

> 源文件：`vllm/v1/worker/gpu_model_runner.py` L2538–L2652

***

## 函数签名与返回值

```python
def _preprocess(
    self,
    scheduler_output: "SchedulerOutput",
    num_input_tokens: int,              # padding 后的总 token 数（供 CUDA Graph 使用）
    intermediate_tensors: IntermediateTensors | None = None,  # Pipeline Parallel 中间层输入
) -> tuple[
    torch.Tensor | None,        # input_ids：纯文本路径使用，多模态时为 None
    torch.Tensor | None,        # inputs_embeds：多模态/prompt embeds 路径使用
    torch.Tensor,               # positions：token 位置编码（1D 或多维）
    IntermediateTensors | None, # intermediate_tensors：PP 非首节点使用
    dict[str, Any],             # model_kwargs：传入模型 forward 的额外参数
    ECConnectorOutput | None,   # ec_connector_output：EC KV transfer 输出
]:
```

**核心目标：** 根据模型类型（纯文本 / 多模态 / prompt embeds）和 Pipeline Parallel 位置（首节点 / 非首节点），准备模型 `forward()` 所需的 `input_ids` 或 `inputs_embeds`、`positions`，以及可能需要的中间层张量。

***

## 三条主路径总览

    _preprocess
        │
        ├── 路径 A：多模态模型（supports_mm_inputs=True, is_first_rank=True）
        │   ├── 运行 MM Encoder（视觉/音频等）
        │   ├── embed_input_ids 合并文本 + 多模态 embedding
        │   └── input_ids=None, inputs_embeds=<merged>
        │
        ├── 路径 B：启用 prompt_embeds（enable_prompt_embeds=True, is_first_rank=True）
        │   ├── 找出 token id 位置，调用 embed_input_ids 转 embedding
        │   ├── 与已有 prompt embeds 合并
        │   └── input_ids=None, inputs_embeds=<mixed>
        │
        └── 路径 C：纯文本模型（默认，性能最优）
            ├── 直接取 input_ids.gpu（embedding 层在 CUDA Graph 内部执行）
            └── input_ids=<token_ids>, inputs_embeds=None

***

## 示例场景

与 `_prepare_inputs` 文档使用**完全相同的 batch**（见 `prepare_inputs_flow.md`），`max_model_len=2048`：

| req\_idx | req\_id | 已计算 token 数 | 本次调度 token 数 | 阶段说明                                          |
| -------- | ------- | ----------- | ------------ | --------------------------------------------- |
| 0        | req-A   | 0           | 4            | chunked prefill，prompt 共 10 个 token，本次处理前 4 个 |
| 1        | req-B   | 100         | 1            | decode（无 spec），送入上一步采样的 1 个 token             |
| 2        | req-C   | 50          | 3            | decode（有 spec），1 个真实 token + 2 个 draft token  |

来自上游的输入值：

```python
# 来自 _prepare_inputs Step 9（已写入 GPU）：
input_ids.gpu[:8]   = tensor([p0,p1,p2,p3,  t100,  t50,t51,t52])
positions.gpu[:8]   = tensor([0, 1, 2, 3,   100,   50, 51, 52])

# 来自 _determine_batch_execution_and_padding：
num_input_tokens    = 8      # num_tokens_padded，PIECEWISE 无额外 padding

# 模型配置（LLaMA 纯文本模型）：
supports_mm_inputs  = False  # 无多模态输入
enable_prompt_embeds= False  # 无 prompt embeddings
is_first_rank       = True   # 单节点，无 Pipeline Parallel
is_encoder_decoder  = False  # 纯 decoder

intermediate_tensors = None  # 单节点无需中间层传递
```

***

## 各步骤详解

### Step 1 — 初始化基础状态变量

**目的：** 提取本次 batch 的基础状态（总 token 数、PP 位置、模型类型），并将 `ec_connector_output` 初始化为 `None`，作为后续 EC KV transfer 的占位。

**代码（L2551–2557）：**

```python
num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 真实 token 数（未 padding）
is_first_rank = get_pp_group().is_first_rank   # 是否是 Pipeline Parallel 的第一个节点
is_encoder_decoder = self.model_config.is_encoder_decoder  # 是否是 encoder-decoder 模型

ec_connector_output = None  # EC（External Cache）KV transfer 输出，默认无
```

**示例数据变化：**

    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens = 8
    is_first_rank        = True    # 单节点，无 PP
    is_encoder_decoder   = False   # LLaMA 是纯 decoder 模型
    ec_connector_output  = None

***

### Step 2 — 路径选择与输入准备

这一步是整个函数的核心，根据模型类型走三条路径之一。

#### 路径 A — 多模态模型（L2559–2584）

**目的：** 运行视觉/音频等多模态 encoder，将编码结果与文本 token embedding 合并，得到统一的 `inputs_embeds`。多模态模型必须在 CUDA Graph 外部做 embedding，因为输入形状因模态而异。

**代码（L2559–2584）：**

```python
if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
    with self.maybe_get_ec_connector_output(
        scheduler_output, encoder_cache=self.encoder_cache,
    ) as ec_connector_output:
        # 运行多模态 encoder（视觉/音频等），结果缓存到 encoder_cache
        self._execute_mm_encoder(scheduler_output)
        # 收集本次 batch 需要的多模态 embedding 切片
        mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

    # 将文本 token id 通过 embedding 层转换，并在多模态位置替换为 encoder 输出
    inputs_embeds_scheduled = self.model.embed_input_ids(
        self.input_ids.gpu[:num_scheduled_tokens],  # token ids（真实长度，未 padding）
        multimodal_embeddings=mm_embeds,            # 多模态 embedding 列表
        is_multimodal=is_mm_embed,                  # 每个 token 是否是多模态位置的掩码
    )
    # 拷贝到预分配 buffer（含 padding 区域）
    self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

    # 返回 padding 后的 embeds slice（供 CUDA Graph 使用固定 shape）
    input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
    # input_ids = None 或 raw token ids（部分模型需要原始 id）
    # inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]

    model_kwargs = {
        **self._init_model_kwargs(num_scheduled_tokens),
        **self._extract_mm_kwargs(scheduler_output),  # 多模态特有参数（如 image_sizes）
    }
```

**示例数据变化：**

    supports_mm_inputs = False  # LLaMA 不是多模态模型

    → 跳过路径 A

***

#### 路径 B — Prompt Embeddings（L2585–2611）

**目的：** 部分请求携带预计算的 prompt embeddings（如 RAG 场景），需要将 token id 转成 embedding 后与 prompt embeds 合并。由于并非所有请求都有 prompt embeds，且 decode 阶段不再有 embeds，此路径性能次优（embedding 层在 CUDA Graph 外部）。

**代码（L2585–2611）：**

```python
elif self.enable_prompt_embeds and is_first_rank:
    # 找出哪些位置是 token id（而非 prompt embed）
    token_ids_idx = (
        self.is_token_ids.gpu[:num_scheduled_tokens]  # 标记每个 token 是否是 token id
        .nonzero(as_tuple=False)
        .squeeze(1)
    )
    # 只对 token id 位置调用 embedding 层
    if token_ids_idx.numel() > 0:
        token_ids = self.input_ids.gpu[token_ids_idx]
        tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
        self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds  # 写入对应槽位

    inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]  # 含 padding
    model_kwargs = self._init_model_kwargs(num_input_tokens)
    input_ids = None  # 使用 embeds 路径，无需 token ids
```

**示例数据变化：**

    enable_prompt_embeds = False  # 三个请求均无 prompt embeds

    → 跳过路径 B

***

#### 路径 C — 纯文本模型（L2612–2619，默认路径）

**目的：** 纯文本模型直接将 GPU 上的 `input_ids` 切片传给模型，embedding 层包含在 CUDA Graph 内部，是性能最优路径。`inputs_embeds=None` 告诉模型走 token id → embedding 的内部路径。

**代码（L2612–2619）：**

```python
else:
    # For text-only models, we use token ids as input.
    # While it is possible to use embeddings as input just like the
    # multimodal models, it is not desirable for performance since
    # then the embedding layer is not included in the CUDA graph.
    input_ids = self.input_ids.gpu[:num_input_tokens]  # padding 后的 token id slice
    inputs_embeds = None
    model_kwargs = self._init_model_kwargs(num_input_tokens)  # 生成模型前向的额外参数
```

**示例数据变化：**

    # 走路径 C（纯文本 LLaMA）

    num_input_tokens = 8   # padding 后的 token 数

    input_ids = self.input_ids.gpu[:8]
              = tensor([p0, p1, p2, p3,  t100,  t50, t51, t52])
    #                   ←   req-A   →    req-B   ←   req-C   →

    inputs_embeds = None   # 不使用 embedding 路径

    model_kwargs = {}      # LLaMA 非 pooling 模型，_init_model_kwargs 返回空 dict

***

### Step 3 — 选择 positions 张量

**目的：** 根据模型使用的位置编码类型，取出对应的 positions 张量切片。标准 1D RoPE 取 `positions.gpu`；多维 RoPE（M-RoPE / XD-RoPE）取对应的多维张量，形状为 `[dims, num_tokens]`。

**代码（L2621–2626）：**

```python
if self.uses_mrope:
    # M-RoPE（Qwen2-VL 等）：3 维位置（时间/高/宽），shape [3, num_tokens]
    positions = self.mrope_positions.gpu[:, :num_input_tokens]
elif self.uses_xdrope_dim > 0:
    # XD-RoPE（HunYuan-VL 等）：多维位置，shape [dims, num_tokens]
    positions = self.xdrope_positions.gpu[:, :num_input_tokens]
else:
    # 标准 1D 位置编码，shape [num_tokens]
    positions = self.positions.gpu[:num_input_tokens]
```

**示例数据变化：**

    uses_mrope      = False  # LLaMA 使用标准 RoPE
    uses_xdrope_dim = 0

    # 走 else 分支：
    positions = self.positions.gpu[:8]
              = tensor([0, 1, 2, 3,  100,  50, 51, 52])
    #                   ←  req-A  →  req-B  ←  req-C →
    # 来自 _prepare_inputs Step 3 计算并在 Step 9 拷贝到 GPU 的结果

***

### Step 4 — Pipeline Parallel 中间层处理

**目的：** 在 Pipeline Parallel 场景中，非首节点不从 embedding 开始计算，而是直接接收上一个 PP 节点通过 NCCL 传来的中间激活张量（`intermediate_tensors`）。首节点清空该张量，确保从头计算。

**代码（L2628–2634）：**

```python
if is_first_rank:
    # 首节点：不需要中间层输入，清空（从 embedding 开始计算）
    intermediate_tensors = None
else:
    # 非首节点：接收上一阶段的激活，同步并切片到当前 token 范围
    assert intermediate_tensors is not None
    intermediate_tensors = self.sync_and_slice_intermediate_tensors(
        num_input_tokens,      # 当前节点处理的 token 数
        intermediate_tensors,  # 上一节点传来的激活张量
        True,                  # sync_self=True：拷贝到本地 buffer
    )
```

`sync_and_slice_intermediate_tensors` 内部（L2427–2456）：

```python
# 将上一节点的激活拷贝到本地预分配 buffer，并按 num_tokens 切片
for k, v in intermediate_tensors.items():
    is_scattered = k == "residual" and is_rs   # SP 场景 residual 是分片的
    copy_len = num_tokens // tp if is_scattered else num_tokens
    self.intermediate_tensors[k][:copy_len].copy_(v[:copy_len], non_blocking=True)

return IntermediateTensors({
    k: v[:num_tokens // tp] if k == "residual" and is_rs else v[:num_tokens]
    for k, v in self.intermediate_tensors.items()
})
```

**示例数据变化：**

    is_first_rank = True   # 单节点，无 PP

    intermediate_tensors = None   # 清空，从 embedding 层开始计算

> **PP 非首节点时（对比示例）：** 假设 2 阶段 PP，当前节点是第 2 阶段：
>
>     # 上一节点传来：intermediate_tensors = {"hidden_states": tensor([8, 4096])}
>     intermediate_tensors = sync_and_slice_intermediate_tensors(
>      num_input_tokens=8,
>      intermediate_tensors={"hidden_states": tensor([8, 4096])},
>      sync_self=True,
>     )
>     # 结果：本地 self.intermediate_tensors["hidden_states"][:8] 被填充
>     # 返回：IntermediateTensors({"hidden_states": tensor([8, 4096])})
>     # 模型 forward 从第 17 层（而非第 0 层）开始计算

***

### Step 5 — Encoder-Decoder 模型运行 Encoder（可选）

**目的：** Encoder-Decoder 模型（如 T5、Whisper）需要先运行 encoder 对输入编码，将编码结果注入 `model_kwargs`，供 decoder cross-attention 使用。纯 decoder 模型跳过。

**代码（L2636–2643）：**

```python
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    # Run the encoder, just like we do with other multimodal inputs.
    # For an encoder-decoder model, our processing here is a bit
    # simpler, because the outputs are just passed to the decoder.
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

**示例数据变化：**

    is_encoder_decoder = False   # LLaMA 是纯 decoder 模型

    → 跳过，model_kwargs 保持 {}

***

### Step 6 — 返回结果

**代码（L2645–2652）：**

```python
return (
    input_ids,             # token id 张量 或 None
    inputs_embeds,         # embedding 张量 或 None
    positions,             # 位置编码张量
    intermediate_tensors,  # PP 中间层激活 或 None
    model_kwargs,          # 模型 forward 额外参数
    ec_connector_output,   # EC KV transfer 输出 或 None
)
```

**示例最终输出（完整数据链路）：**

    # 输入（来自 _prepare_inputs 同一 batch）：
    #   req-A: prefill 4 token，req-B: decode 1 token，req-C: spec decode 3 token

    # Step 1 → num_scheduled_tokens=8, is_first_rank=True, is_encoder_decoder=False
    # Step 2 → 走路径 C（纯文本），input_ids=tensor([p0,p1,p2,p3,t100,t50,t51,t52])
    # Step 3 → positions=tensor([0,1,2,3, 100, 50,51,52])
    # Step 4 → intermediate_tensors=None（首节点，从头计算）
    # Step 5 → encoder-decoder 跳过

    返回值：
    input_ids            = tensor([p0, p1, p2, p3,  t100,  t50, t51, t52])   # shape [8]
    inputs_embeds        = None
    positions            = tensor([0,  1,  2,  3,   100,   50,  51,  52])    # shape [8]
    intermediate_tensors = None
    model_kwargs         = {}
    ec_connector_output  = None

调用方 `_model_forward`（L2824 附近）使用这些返回值：

```python
hidden_states = model(
    input_ids=input_ids,           # tensor([8])，embedding 层在 CUDA Graph 内部执行
    inputs_embeds=inputs_embeds,   # None
    positions=positions,           # tensor([8])，RoPE 用
    intermediate_tensors=intermediate_tensors,  # None
    **model_kwargs,                # {}
)
# → hidden_states: shape [8, hidden_dim]
```

***

## 关键设计要点

| 设计点                                              | 具体做法                                                                            | 收益                                                                    |
| ------------------------------------------------ | ------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **纯文本走 input\_ids 路径**                           | `input_ids=gpu_tensor, inputs_embeds=None`，embedding 层留在模型内部                    | embedding 层包含在 CUDA Graph 内，重放时无需额外调用，性能最优                            |
| **多模态走 inputs\_embeds 路径**                       | 先在 CUDA Graph 外部做 encode + embed，再传 `inputs_embeds`                             | 多模态 encoder 输出形状动态，无法固化在 CUDA Graph 中                                 |
| **num\_input\_tokens vs num\_scheduled\_tokens** | embedding 合并用 `num_scheduled_tokens`（真实长度），返回值切片用 `num_input_tokens`（padding 后） | CUDA Graph 需要固定 shape，模型 forward 接收 padding 后张量；encoder 输出不需要 padding |
| **PP 首节点清空 intermediate\_tensors**               | `is_first_rank → intermediate_tensors = None`                                   | 首节点从 embedding 开始，避免误用上一轮的残留张量                                        |
| **PP 非首节点 non\_blocking 拷贝**                     | `sync_and_slice_intermediate_tensors` 用 `non_blocking=True`                     | 与 GPU 计算重叠，隐藏 PP 节点间的激活传输延迟                                           |
| **positions 多路径选择**                              | 根据 `uses_mrope` / `uses_xdrope_dim` / 默认三路分支                                    | 不同模型的位置编码维度不同，统一接口但各取所需切片                                             |
| **EC Connector 上下文管理器**                          | `with maybe_get_ec_connector_output(...) as ec_connector_output`                | EC KV transfer 与 encoder 执行并行，`with` 块退出时自动等待结果                       |

