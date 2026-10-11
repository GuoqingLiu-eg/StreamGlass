# Encoder Cache（P1-3）设计规格 — chunked 模式

> 状态：设计已定稿（2026-07-21），实现未开始。reference 模式（累计重算）仍是默认与验收基线。
> 动机（实测）：15s 长句下 `encoder_cumulative_fp16.engine` 单步从 24ms 涨到 174ms，
> max_inference 463ms，击穿 240ms 硬上限。cached 模式目标：单步 encoder 成本与句长无关。

## 1. 模型结构事实（已对照冻结源码核实）

| 项 | 值 | 出处 |
|---|---|---|
| 内部 encoder | conformer_shaw **600m**，24 层 ConformerBlock，d=1024 | unity/builder.py:167,197 |
| 注意力位置编码 | **Shaw 相对位置**，clip 左 64 / 右 8，无 rel-pos values | conformer_shaw/builder.py:59-61 |
| ConformerBlock 前向 | ffn1(×0.5)+res → self_attn → conv → ffn2(×0.5)+res → layer_norm | fairseq2 models/conformer/block.py:119-135 |
| depthwise conv | kernel **31**，padding **"same"（对称 15/15，非因果）** | fairseq2 models/conformer/convolution.py:79 |
| adaptor | 1 层，kernel 8，**stride 8** | unity/builder.py:129-131 |
| frontend | fbank80 stride-2 堆叠（395 fbank → ~197 w2v 帧 → /8 → 25 encoder 帧） | 实测帧比 + w2vbert 600m 配置 |

## 2. 关键结论

1. **Shaw 注意力允许 append-only K/V 缓存**：得分只依赖 query-key 距离（clip [−64,+8]），
   距离矩阵可由 (past_len, new_len) 生成，无绝对位置状态。
2. **对称 conv 使"精确因果流式"不可行**：每层 15 帧右依赖逐层复合（24 层 → 理论前瞻 7.2s）。
   因此采用**滑动重算窗**而非纯因果缓存。
3. adaptor 输入帧数被 /16（fbank→encoder），15s 才 ~94 帧——**adaptor 每步全量重算**，
   便宜且对其输入精确，不进缓存。

## 3. cached 模式设计：冻结前缀 + 滑动重算窗

每个 320ms step（新增 ~16 个 fbank 帧 → ~8 个 w2v 帧）：

```
w2v 帧轴: [0 .............. F-R) [F-R ........ F) [F .. F+new)
           冻结区（不再更新）      重算窗 R 帧        新帧
```

- 重算窗 R + 新帧走全部 24 层；每层注意力 query=[窗+新]，key/value=[冻结区缓存 K/V ‖ 窗+新现算 K/V]。
- 冻结区帧的输出永不更新（官方语义偏离点：它们看不到 R 之外的右上下文）。
- 窗左缘 conv 需要冻结区侧 15 帧的**逐层输入激活尾巴**（缓存）。
- 每步右移：新进入冻结区的帧把其各层 K/V 与 conv 尾巴写入缓存。
- adaptor + frontend 堆叠残余在 C++ 侧处理；adaptor 对 [全部 w2v 输出（冻结输出缓存‖本步窗输出）] 全量跑。
- R 初值 **64 w2v 帧（1.28s）**，可配置；R 越大越接近官方、越贵。步进成本 O((R+8)×24 层)＝常数。

### 跨块状态清单（每 utterance）

| 状态 | 形状 | 大小@1000 w2v 帧 |
|---|---|---|
| 每层 K/V（冻结区） | 24×2×[L_frozen,16,64] fp16 | ~98 MB 上限 |
| 每层 conv 输入尾巴 | 24×[15,1024] fp16 | 720 KB |
| w2v 输出缓存（冻结区，供 adaptor） | [L_frozen,1024] fp16 | 2 MB |
| fbank 堆叠残余 | ≤1 帧 | 忽略 |
| 帧计数/偏移 | 标量 | — |

## 4. Engine 拆分与 I/O 契约（全部新文件名，不动生产 engine）

**engine A：`encoder_w2v_chunk_fp16.engine`**（24 层 conformer，单步）
- 输入：`stacked_features [1,W,160]`（W=R+new）、`frozen_k/frozen_v [24,16,L_f,64]`、
  `conv_tail [24,15,1024]`、`frozen_len [1]`（生成 Shaw 距离矩阵用）
- 输出：`w2v_out [1,W,1024]`、`new_k/new_v [24,16,W,64]`（C++ 侧把移入冻结区的部分 append 到缓存）、
  `conv_tail_out [24,15,1024]`
- 动态轴：W ∈ [8, 128]，L_f ∈ [0, 1024]

**engine B：`encoder_adaptor_fp16.engine`**（frontend 尾部 layernorm? + adaptor 1 层 + 输出投影按官方图切分）
- 输入：`w2v_seq [1,N,1024]`（冻结输出缓存 ‖ 本步 w2v_out），输出 `encoder_output [1,⌈N/8⌉,1024]`
- 每步全量，N ≤ 2048

导出方式与现有一致：官方权重加载 → wrapper 在 torch 层重排 forward（复用官方子模块，NaiveSDPA 替换）→
ONNX opset 17 → AGX docker（streamglass/seamless-s0:0.1）内 trtexec FP16。
新脚本：`deploy/export/export_encoder_chunked.py` + `deploy/trt/build_encoder_chunked.sh`。

## 5. 实现步骤

1. torch 参考实现（wrapper 内先做 python 版滑窗前向），与官方整段输出对比：
   冻结区 cosine、窗内 cosine、末帧 cosine 随 R 的曲线 → 选定 R。
2. ONNX 导出 + trtexec 构建（两个 engine）。
3. C++ `EncoderRuntime` 增加 chunked 模式（新类或模式开关）+ 状态缓冲；`StreamingPipelineOptions.cached_encoder`。
4. A/B 门（待办规则 10）：逐 step encoder 输出 cosine ≥0.999（窗内）/记录冻结区漂移、
   READ/WRITE 决策一致率、token/unit/译文对比、长句 soak 无饿死（吸取 decoder probe 教训：
   **EMMA p_choose 对输入扰动敏感**，encoder 冻结区漂移可能同样引起决策偏移，必须长句实测）。
5. 性能验收：15s 长句单步 encoder ≤40ms 且与句长无关；max_inference 回到 240ms 内。

## 6. 已知风险

- 冻结区右上下文缺失是不可消除的语义偏离；R 是唯一调节旋钮。
- p_choose 敏感性（decoder probe 实测教训）可能放大 encoder 偏差 → 决策时机漂移；
  兜底：与 decoder probe 相同的"偏差可控 + 强制校验"思路不适用于 encoder（无廉价 reference），
  只能靠加大 R 或回退 reference 模式。
- ONNX 导出 Shaw 距离矩阵生成需用张量 op 表达（避免 python 分支进图）。
