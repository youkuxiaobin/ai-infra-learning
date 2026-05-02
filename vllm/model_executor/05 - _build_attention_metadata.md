# `_build_attention_metadata` 函数详解

> 源文件：`vllm/v1/worker/gpu_model_runner.py` L1517–L1761

***

## 函数签名与返回值

```python
def _build_attention_metadata(
    self,
    num_tokens: int,                                      # 本次真实 token 数（未 padding）
    num_reqs: int,                                        # 本次真实请求数
    max_query_len: int,                                   # batch 中最长的 query 长度
    num_tokens_padded: int | None = None,                 # CUDA Graph padding 后 token 数
    num_reqs_padded: int | None = None,                   # CUDA Graph padding 后请求数
    ubatch_slices: UBatchSlices | None = None,            # DBO microbatch 切分（None=不切）
    logits_indices: torch.Tensor | None = None,           # 用于 kv_sharing_fast_prefill
    use_spec_decode: bool = False,                        # 是否启用投机解码
    for_cudagraph_capture: bool = False,                  # 是否处于 CUDA Graph 捕获阶段
    num_scheduled_tokens: dict[str, int] | None = None,   # encoder-decoder 使用
    cascade_attn_prefix_lens: list[list[int]] | None = None,  # 共享前缀长度
) -> tuple[
    PerLayerAttnMetadata,           # 每层 attention metadata（dict 或 list[dict]）
    CommonAttentionMetadata | None, # spec decode drafter 使用的公共 metadata
]:
```

**核心目标：** 为每个 KV cache group 的每个 attention 层构建推理所需的 `AttentionMetadata`，将 `query_start_loc`、`seq_lens`、`block_table`、`slot_mapping` 等信息组装成 attention backend（如 FlashAttention）可以直接消费的结构。

***

## 关键数据结构

```python
# gpu_model_runner.py L186
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict
# - 无 DBO：dict，key=layer_name，value=AttentionMetadata
# - 有 DBO：list[dict]，每个 microbatch 一个 dict

# vllm/v1/attention/backends/utils.py L57
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor      # [num_reqs+1] GPU，每请求 query 起始位置（cumsum）
    query_start_loc_cpu: torch.Tensor  # 同上，CPU 版
    seq_lens: torch.Tensor             # [num_reqs] GPU，每请求当前序列总长度
    num_reqs: int                      # padding 后请求数
    num_actual_tokens: int             # padding 后 token 数
    max_query_len: int                 # 最长 query 长度
    max_seq_len: int                   # 最长序列长度
    block_table_tensor: torch.Tensor   # [num_reqs, max_blocks] KV cache 块表
    slot_mapping: torch.Tensor         # [num_tokens] 每 token 对应的 KV slot
    causal: bool = True
```

***

## 示例场景

与 `_prepare_inputs` 文档使用**完全相同的 batch**（见 `prepare_inputs_flow.md`），`max_model_len=2048`，block_size=16：

| req_idx | req_id | 已计算 token 数 | 本次调度 token 数 | 阶段说明 |
| --- | --- | --- | --- | --- |
| 0 | req-A | 0 | 4 | chunked prefill，prompt 共 10 个 token，本次处理前 4 个 |
| 1 | req-B | 100 | 1 | decode（无 spec），送入上一步采样的 1 个 token |
| 2 | req-C | 50 | 3 | decode（有 spec），1 个真实 token + 2 个 draft token |

来自上游的输入值：

```python
# 来自 _prepare_inputs / _determine_batch_execution_and_padding：
num_tokens          = 8              # 4+1+3
num_reqs            = 3
max_query_len       = 4              # req-A 最长
num_tokens_padded   = 8              # PIECEWISE 模式，无额外 padding
num_reqs_padded     = 3
ubatch_slices       = None           # 单卡，无 DBO
use_spec_decode     = True           # req-C 有 spec decode
for_cudagraph_capture = False
cascade_attn_prefix_lens = None      # 无共享前缀

# 来自 _prepare_inputs Step 8（已写入 self.xxx）：
query_start_loc.gpu = [0, 4, 5, 8, 8, ...]   # cu_num_tokens 前缀
seq_lens.gpu[:3]    = [4, 101, 53]            # num_computed + num_scheduled
# 来自 _prepare_inputs Step 7（slot mapping）：
block_table[0]      = [[0,1,...], [6,...], [3,...]]  # 各请求的 block 分配
slot_mapping.gpu[:8]= [0,1,2,3, 1604, 800,801,802]  # 每 token 的 KV slot
```

***

## 各步骤详解

### Step 1 — 早退：无注意力层模型直接返回空

**目的：** 部分模型（如纯 MLP 或 SSM 架构）没有 KV cache，不需要 attention metadata，直接返回空避免无效计算。

**代码（L1535–1536）：**

```python
# Attention metadata is not needed for attention free models
if len(self.kv_cache_config.kv_cache_groups) == 0:
    return {}, None
```

**示例数据变化：**

    kv_cache_config.kv_cache_groups = [KVCacheGroup(...)]  # LLaMA 有 1 个 KV cache group
    len(...) = 1 ≠ 0

    → 不早退，继续执行

***

### Step 2 — 确定 padding 尺寸并初始化 attn_metadata 容器

**目的：** 将 `num_tokens_padded` / `num_reqs_padded` 设为默认值（若未提供则等于真实值）；根据是否有 DBO ubatch，初始化 `attn_metadata` 为单个 dict 或 list of dict。

**代码（L1538–1544）：**

```python
num_tokens_padded = num_tokens_padded or num_tokens  # 若 None 则取真实值
num_reqs_padded = num_reqs_padded or num_reqs

attn_metadata: PerLayerAttnMetadata = {}
if ubatch_slices is not None:
    # DBO 场景：每个 microbatch 一个独立 dict
    attn_metadata = [dict() for _ in range(len(ubatch_slices))]
```

**示例数据变化：**

    num_tokens_padded = 8 or 8 = 8   （已由调用方传入）
    num_reqs_padded   = 3 or 3 = 3

    ubatch_slices = None              （单卡，无 DBO）

    attn_metadata = {}                （单个 dict，非 DBO 路径）

***

### Step 3 — 确定 max_seq_len

**目的：** 计算本次 batch 中最长的序列长度，FlashAttention 等 kernel 需要用它来分配内部缓冲区和选择正确的 kernel 分支。CUDA Graph 捕获阶段使用 `max_model_len` 保证 kernel 始终选择支持最大长度的版本。

**代码（L1546–1552）：**

```python
if for_cudagraph_capture:
    # 捕获阶段用最大值，确保 sliding window 模型选择正确 kernel
    max_seq_len = self.max_model_len
else:
    max_seq_len = self.seq_lens.np[:num_reqs].max().item()
```

**示例数据变化：**

    for_cudagraph_capture = False   （正常推理，非捕获阶段）

    seq_lens.np[:3] = [4, 101, 53]
    max_seq_len = max([4, 101, 53]) = 101
    # 即 req-B：已计算 100 + 本次 1 = 101 token

***

### Step 4 — 处理 Spec Decode 的 num_accepted_tokens（可选）

**目的：** 投机解码时，某些 attention backend（如 GDN）需要知道上一步每个请求接受了多少 draft token，以便正确处理 KV cache 的写入位置。

**代码（L1554–1559）：**

```python
if use_spec_decode:
    self.num_accepted_tokens.np[:num_reqs] = (
        self.input_batch.num_accepted_tokens_cpu[:num_reqs]  # 上一步各请求接受的 token 数
    )
    self.num_accepted_tokens.np[num_reqs:].fill(1)  # 未使用槽位填 1（默认接受）
    self.num_accepted_tokens.copy_to_gpu()
```

**示例数据变化：**

    use_spec_decode = True   （req-C 有 2 个 draft token）

    # 假设上一步（初始化时）各请求接受的 token 数：
    num_accepted_tokens_cpu[:3] = [1, 1, 2]
    # req-A: prefill 不参与 spec decode，填 1
    # req-B: decode 无 spec，填 1
    # req-C: 上一步接受了 2 个 draft token

    num_accepted_tokens.np[:3] = [1, 1, 2]
    num_accepted_tokens.np[3:].fill(1)    # padding 槽位填 1
    → copy_to_gpu()：num_accepted_tokens.gpu = tensor([1, 1, 2, 1, ...])

***

### Step 5 — 获取 block_table 和 slot_mapping（内部函数）

**目的：** 根据 KV cache group id，取出对应的 block table（记录每个请求使用哪些物理 KV block）和 slot mapping（记录每个 token 写入哪个 KV slot），并将 padding 槽位填为 `-1`，防止 kernel 越界访问。

**代码（L1563–1587）：**

```python
def _get_block_table_and_slot_mapping(kv_cache_gid: int):
    kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
    if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
        # encoder-only 特殊处理：block table 和 slot mapping 全零
        blk_table_tensor = torch.zeros((num_reqs_padded, 1), ...)
        slot_mapping = torch.zeros((num_tokens_padded,), ...)
    else:
        # 标准 decoder attention：从预分配 buffer 取
        blk_table = self.input_batch.block_table[kv_cache_gid]
        blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)  # [num_reqs_padded, max_blocks]
        slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]    # [num_tokens_padded]

    # 将 padding 区域填 -1，防止 reshape_and_cache 写入无效位置
    slot_mapping[num_tokens:num_tokens_padded].fill_(-1)        # token padding 区
    blk_table_tensor[num_reqs:num_reqs_padded].fill_(-1)        # req padding 区

    return blk_table_tensor, slot_mapping

# Step 5 在函数定义后立即调用 gid=0
block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
```

**示例数据变化：**

    kv_cache_gid = 0
    kv_cache_spec = FullAttentionSpec(...)   （标准 decoder attention，非 encoder-only）

    # block_table[0]：shape [max_num_reqs, max_blocks_per_req]
    # 各请求 block 分配（block_size=16）：
    #   req-A: 位置 [0,3]，占 block 0
    #   req-B: 位置 [0,100]，占 block 0~6，当前用到 block 6（100//16=6）
    #   req-C: 位置 [0,52]，占 block 0~3，当前用到 block 3（50//16=3）
    blk_table_tensor = block_table.get_device_tensor(3)
    # shape [3, max_blocks]，例如：
    # [[block0, -1, ...],     ← req-A（只用了1个block）
    #  [block0,block1,...,block6, -1,...],  ← req-B
    #  [block0,block1,block2,block3, -1,...]] ← req-C

    # slot_mapping（来自 _prepare_inputs Step 7）：
    slot_mapping.gpu[:8] = [0,1,2,3,  1604,  800,801,802]
    #                       ← req-A →  req-B  ←  req-C  →
    # req-B token100: block6*16 + 100%16 = 6*16+4 = 100 → slot 1604（假设 block6 物理 id=100，100*16+4=1604）
    # req-C token50:  block3*16 + 50%16  = 3*16+2 = 50  → slot 800（假设 block3 物理 id=50，50*16+0=800）

    # num_tokens=8, num_tokens_padded=8 → 无 padding 区，fill_(-1) 不执行
    # num_reqs=3,   num_reqs_padded=3   → 无 padding 区，fill_(-1) 不执行

***

### Step 6 — 构建 CommonAttentionMetadata 基础对象

**目的：** 将所有请求级别的公共信息（query 位置、序列长度、block table、slot mapping）组装成 `CommonAttentionMetadata`，作为所有 KV cache group 的共享基础，后续各 group 通过浅拷贝复用。

**代码（L1590–1605）：**

```python
cm_base = CommonAttentionMetadata(
    query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],  # [4] GPU tensor
    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
    seq_lens=self.seq_lens.gpu[:num_reqs_padded],                      # [3] GPU tensor
    _seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],
    _num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs_padded],
    num_reqs=num_reqs_padded,           # 3
    num_actual_tokens=num_tokens_padded, # 8
    max_query_len=max_query_len,         # 4
    max_seq_len=max_seq_len,             # 101（Step 3 计算得到）
    block_table_tensor=block_table_gid_0,
    slot_mapping=slot_mapping_gid_0,
    causal=True,
)
```

**示例数据变化：**

    # query_start_loc.gpu[:4] 来自 _prepare_inputs Step 8：
    query_start_loc.gpu[:4] = tensor([0, 4, 5, 8])
    # 含义：req-A query 在 [0,4)，req-B 在 [4,5)，req-C 在 [5,8)

    seq_lens.gpu[:3] = tensor([4, 101, 53])
    # 含义：req-A 序列长度 4，req-B 101，req-C 53

    num_computed_tokens_cpu_tensor[:3] = tensor([0, 100, 50])
    # 含义：各请求已完成前向的 token 数

    cm_base = CommonAttentionMetadata(
        query_start_loc      = tensor([0, 4, 5, 8]),   # GPU
        query_start_loc_cpu  = tensor([0, 4, 5, 8]),   # CPU
        seq_lens             = tensor([4, 101, 53]),    # GPU
        num_reqs             = 3,
        num_actual_tokens    = 8,
        max_query_len        = 4,
        max_seq_len          = 101,
        block_table_tensor   = <shape [3, max_blocks]>,
        slot_mapping         = tensor([0,1,2,3, 1604, 800,801,802]),
        causal               = True,
    )

***

### Step 7 — DCP（Decode Context Parallelism）处理（可选）

**目的：** 在 Decode Context Parallelism 场景下，每个 rank 只处理序列的一部分，需要计算本 rank 负责的局部序列长度，让 attention kernel 知道从哪到哪做注意力计算。

**代码（L1607–1620）：**

```python
if self.dcp_world_size > 1:
    self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
        self.seq_lens.cpu[:num_reqs],
        self.dcp_world_size,
        self.dcp_rank,
        self.parallel_config.cp_kv_cache_interleave_size,
    )
    self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
    self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)
    cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
    cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[:num_reqs_padded]
```

**示例数据变化：**

    dcp_world_size = 1   （单卡，无 Decode Context Parallelism）

    → 跳过，cm_base.dcp_local_seq_lens 保持 None

***

### Step 8 — 遍历 KV cache groups，为每层构建 AttentionMetadata

**目的：** 对每个 KV cache group（混合模型如 Mamba+Attention 可能有多个 group），浅拷贝 `cm_base` 后更新 group 专属的 `block_table`/`slot_mapping`，再调用对应 backend 的 `builder.build()` 生成最终的 per-layer `AttentionMetadata`，写入 `attn_metadata[layer_name]`。

**代码（L1700–1729）：**

```python
spec_decode_common_attn_metadata = None

for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
    cm = copy(cm_base)   # 浅拷贝，只替换 group 特有字段

    # encoder seq_lens（纯 decoder 模型返回 None）
    cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
        num_scheduled_tokens or {}, kv_cache_group.kv_cache_spec, num_reqs_padded
    )

    # gid > 0 时更新 block_table 和 slot_mapping（gid=0 已在 cm_base 中）
    if kv_cache_gid > 0:
        cm.block_table_tensor, cm.slot_mapping = (
            _get_block_table_and_slot_mapping(kv_cache_gid)
        )

    # spec decode drafter 使用哪个 group 的 cm
    if self.speculative_config and spec_decode_common_attn_metadata is None:
        spec_decode_common_attn_metadata = cm

    for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
        if ubatch_slices is not None:
            # DBO：对每个 microbatch 单独 build
            for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)
        else:
            # 非 DBO：整体 build
            _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)
```

`_build_attn_group_metadata` 内部三条路径（L1637–1696）：

```python
def _build_attn_group_metadata(kv_cache_gid, attn_gid, common_attn_metadata, ubid=None):
    builder = attn_group.get_metadata_builder(ubid or 0)

    if for_cudagraph_capture:
        # 路径1：CUDA Graph 捕获专用 build
        attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)

    elif cache_key in cached_attn_metadata and builder.supports_update_block_table:
        # 路径2：复用已有 metadata，只更新 block table（hybrid model 优化）
        attn_metadata_i = builder.update_block_table(
            cached_attn_metadata[cache_key],
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.slot_mapping,
        )
    else:
        # 路径3（常规）：完整构建
        attn_metadata_i = builder.build(
            common_prefix_len=cascade_attn_prefix_len,
            common_attn_metadata=common_attn_metadata,
            **extra_attn_metadata_args,   # GDN spec decode 时传 num_accepted_tokens
        )

    # 将同一 group 内所有层共享同一个 metadata 对象
    for layer_name in attn_group.layer_names:
        attn_metadata_dict[layer_name] = attn_metadata_i
```

**示例数据变化（单 KV cache group，非 DBO）：**

    kv_cache_groups = [KVCacheGroup(layer_names=["layer.0.attn",...,"layer.31.attn"])]
    len(kv_cache_groups) = 1

    # 第 1 次（也是唯一一次）循环：kv_cache_gid=0
    cm = copy(cm_base)   # 浅拷贝，字段完全相同

    cm.encoder_seq_lens = None   （纯 decoder LLaMA，无 encoder）

    kv_cache_gid=0，不更新 block_table（gid=0 已在 cm_base 中）

    spec_decode_common_attn_metadata = cm   （spec decode 启用，记录此 cm）

    # attn_groups[0] 通常只有 1 个 attn_group（FlashAttention backend）
    # attn_gid=0：
    ubatch_slices = None → 走整体 build 路径

    builder = FlashAttentionMetadataBuilder(...)
    # 路径3（常规 build）：
    attn_metadata_i = builder.build(
        common_prefix_len=0,       # 无 cascade attention
        common_attn_metadata=cm,   # 包含 query_start_loc, seq_lens, block_table, slot_mapping
    )
    # 生成 FlashAttentionMetadata，内含：
    #   - cu_seqlens_q   = tensor([0, 4, 5, 8])   （query 累积长度）
    #   - cu_seqlens_k   = tensor([0, 4, 101, 53]) （key 累积长度）
    #   - max_seqlen_q   = 4
    #   - max_seqlen_k   = 101
    #   - block_table    = <[3, max_blocks]>
    #   - ...

    # 将所有层共享同一个 metadata：
    for layer_name in ["layer.0.attn", "layer.1.attn", ..., "layer.31.attn"]:
        attn_metadata["layer.N.attn"] = attn_metadata_i

    attn_metadata = {
        "layer.0.attn":  <FlashAttentionMetadata>,
        "layer.1.attn":  <FlashAttentionMetadata>,  # 同一对象
        ...
        "layer.31.attn": <FlashAttentionMetadata>,  # 同一对象
    }

***

### Step 9 — 裁剪 spec\_decode\_common\_attn\_metadata（可选）

**目的：** drafter（EAGLE 等）在 PIECEWISE CUDA Graph 模式下不支持 padding，需要将 `spec_decode_common_attn_metadata` 的尺寸裁剪回真实的 `(num_tokens, num_reqs)`，避免 drafter 看到 padding 的假数据。

**代码（L1751–1759）：**

```python
if spec_decode_common_attn_metadata is not None and (
    num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
):
    # drafter 只用 PIECEWISE CG，不接受 padded metadata
    spec_decode_common_attn_metadata = (
        spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
    )
```

**示例数据变化：**

    spec_decode_common_attn_metadata = cm   （Step 8 中记录）
    num_reqs         = 3
    num_reqs_padded  = 3   → 相等
    num_tokens       = 8
    num_tokens_padded= 8   → 相等

    条件：3 != 3 or 8 != 8 → False

    → 不裁剪，spec_decode_common_attn_metadata 保持 padding 前的原始值

> **有 padding 时的对比：** 若 `num_tokens_padded=16`（FULL CUDA Graph）：
>
>     条件：8 != 16 → True
>     spec_decode_common_attn_metadata = cm.unpadded(num_tokens=8, num_reqs=3)
>     # 将 query_start_loc、seq_lens、block_table、slot_mapping 切回 [:8] / [:3]

***

### Step 10 — 返回结果

**代码（L1761）：**

```python
return attn_metadata, spec_decode_common_attn_metadata
```

**示例最终输出（完整数据链路）：**

    # 输入（来自 _prepare_inputs 同一 batch）：
    #   req-A: prefill 4 token，req-B: decode 1 token，req-C: spec decode 3 token
    #
    # Step 1 → 非 attention-free 模型，不早退
    # Step 2 → num_tokens_padded=8, num_reqs_padded=3, attn_metadata={}
    # Step 3 → max_seq_len = 101（req-B 最长）
    # Step 4 → num_accepted_tokens.gpu = tensor([1,1,2,1,...])
    # Step 5 → block_table_gid_0=[3,max_blocks], slot_mapping=[0,1,2,3,1604,800,801,802]
    # Step 6 → cm_base 组装完成
    # Step 7 → DCP 跳过
    # Step 8 → builder.build() 生成 FlashAttentionMetadata，写入所有 32 层
    # Step 9 → 无 padding 差值，不裁剪

    attn_metadata = {
        "model.layers.0.self_attn":  <FlashAttentionMetadata>,
        "model.layers.1.self_attn":  <FlashAttentionMetadata>,  # 共享同一对象
        ...
        "model.layers.31.self_attn": <FlashAttentionMetadata>,
    }

    spec_decode_common_attn_metadata = CommonAttentionMetadata(
        query_start_loc = tensor([0, 4, 5, 8]),
        seq_lens        = tensor([4, 101, 53]),
        num_actual_tokens = 8,
        max_seq_len     = 101,
        block_table_tensor = <[3, max_blocks]>,
        slot_mapping    = tensor([0,1,2,3, 1604, 800,801,802]),
        ...
    )
    # drafter（EAGLE）在 propose_draft_token_ids 时使用此 cm 构建自己的 attn metadata

***

## 关键设计要点

| 设计点                                | 具体做法                                                                           | 收益                                                           |
| ---------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| **浅拷贝复用 cm\_base**                 | 每个 KV cache group `copy(cm_base)`，只替换 group 专属字段                               | 避免重复构造公共字段，多 group 混合模型（Mamba+Attn）零拷贝复用                     |
| **同 group 所有层共享 metadata**         | `for layer_name in attn_group.layer_names: dict[layer_name] = attn_metadata_i` | 同架构的层 metadata 完全相同，节省内存和构建时间                                |
| **update\_block\_table 缓存优化**      | hybrid model 中相同 spec+builder 的 group 复用 metadata，只更新 block table              | 避免 hybrid 模型重复 build 相同结构的 metadata                          |
| **padding 区域填 -1**                 | `slot_mapping[num_tokens:]` 和 `blk_table[num_reqs:]` 填 -1                      | FULL CUDA Graph 固定 shape，-1 防止 `reshape_and_cache` 写入无效 slot |
| **CUDA Graph 捕获用 max\_model\_len** | `for_cudagraph_capture=True` 时 `max_seq_len=max_model_len`                     | 确保 sliding window 模型在捕获时选择支持最大长度的 kernel 分支                  |
| **spec\_decode 单独保存 cm**           | 记录 `spec_decode_common_attn_metadata` 并按需 `unpadded()`                         | drafter 不支持 padded metadata，需独立裁剪，与主模型 metadata 解耦           |
| **DBO 切片**                         | `split_attn_metadata(ubatch_slices, cm)` 按 microbatch 切分 cm                    | 每个 microbatch 的 query/seq 范围独立，attention 分批计算与通信重叠           |

