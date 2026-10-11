# Encoder Cache 研究结论（P1-3，2026-07-21）：按当前 parity 门槛否决

## 方法
`deploy/export/chunked_encoder_reference.py`（AGX S0 容器内运行）：官方权重 + 官方子模块，
手工分层实现"冻结前缀 + 滑动重算窗 R"前向（设计见 docs/encoder_cache_design.md），
对 15s 真实音频（4×sample_spa，1585 fbank 帧）逐 320ms step 与官方累计输出对比。

## 结果（/opt/streamglass/tmp/encoder-chunk-study/study_long_spa.json）

| R (w2v 帧) | min cos_all | min cos_last8 | max_abs |
|---|---|---|---|
| ∞（sanity，等价全量重算） | **0.9999998** | 0.9999998 | 0.016 |
| 128 (2.56s) | 0.9881 | 0.9826 | 2.19 |
| 64 (1.28s) | 0.9874 | 0.9788 | 1.97 |
| 32 (0.64s) | 0.9624 | 0.9730 | 2.38 |

- sanity 档证明分层复刻精确（ffn1/attn/conv/ffn2/norm 顺序、Shaw 末尾对齐、adaptor 尾部均无误），数据可信。
- R 64→128 几乎无改善（0.9874→0.9881）：偏差是**结构性天花板**——冻结区帧永久缺失
  全局右上下文（对称 conv 15 帧/层 + Shaw 右 clip 8/层，24 层复合），非窗口尺寸问题。

## 判定
- 0.988 << 0.999 parity 门；且 decoder probe 实测已证明 EMMA p_choose 对输入扰动极端敏感
  （KV 冻结导致概率漂移 0.42），该量级 encoder 偏差几乎必然扭曲 READ/WRITE 决策与质量。
- 按待办红线"任何 cached 优化不能以速度换官方语义"：**不建 chunked engine，cached encoder 记为负结果**。

## 生产替代路径（精确、已验证）
15s 长句实时节拍实测（reference 模式，realtime pacing，2 utterance）：
max_inference 459.7ms、deadline_misses 10、**audio_queue_high_water 1、max_capture_queue 139.9ms、
coalesced_batches 0**——单步超时被相邻快步完全吸收，队列无积压增长。
结论：**reference encoder + 合并调度兜底（process_audio_batch 已支持）+ 20s 硬上限即为生产姿态**，
验收口径用"队列零斜率"，而非"每步 <240ms"。

## 重启本课题的条件（任一满足）
1. 未来出现官方训练的流式/因果 encoder 变体（如 chunk-trained 或 emformer 式）；
2. 产品要求 >20s 无切分连续输入且实测队列发散；
3. 有人完成 R=128 全链路质量 A/B（BLEU/听感）证明 0.988 偏差实际可接受——需重新过
   待办规则 10 的逐 step 决策/token/unit/质量门。
