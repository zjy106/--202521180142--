# Gemma3 270M LoRA 微调：SVG 徽标生成实验报告

## 作者信息
- 学号：202521180142
- 姓名：仲瑾毓
- 完成日期：2026-07-15（v2 优化版）

## 一、实验目标

本实验围绕参数高效微调（PEFT）展开，选用的基座是 Gemma3 270M——一个约 2.7 亿参数的小型指令模型。目标是让模型读懂自然语言描述，并据此输出结构合法的 SVG 矢量徽标。由于直接衡量"徽标好不好看"既主观又难以量化，我设计了一套自定义 reward 函数作为代理指标，用来量化微调前后模型能力的差异。

> 一个现实预期：生成训练数据的模型是 Sonnet 级别，本实验用的 270M 比它小两到三个数量级。因此徽标在视觉上必然显得粗糙、简陋，这属于预期范围。本作业真正考察的是三点——微调是否相对基座有提升、reward 设计是否合理、过程分析是否扎实。

---

## 二、Reward 函数设计

`student_kit/reward.py` 的 `LogoGrader` 类从 12 个维度对生成的 SVG 打分，加权求和：

| 维度 | 权重 | 含义 |
|------|------|------|
| well_formed | 1.5 | SVG 能作为合法 XML 解析 |
| attrs | 1.0 | 含有 `xmlns` 和 `viewBox` 属性 |
| viewbox | 1.0 | viewBox 精确等于 `0 0 256 256` |
| no_forbidden_tags | 1.0 | 无 `<image>`/`<script>`/`<iframe>` 等非法标签 |
| no_external_refs | 1.0 | 属性中无外链 URL |
| coord_in_bounds | 1.5 | 坐标落在 [0, 256] 硬边界内 |
| coord_centered | 0.5 | 坐标落在 [20, 236] 软边界（居中）|
| palette | 1.0 | 3~12 种不同颜色（v2: 阈值 2→3）|
| density | 1.0 | 3~80 个 SVG 元素（v2: 阈值 2→3）|
| non_degenerate | 1.5 | 非空、非单色退化输出（v2: 权重 1.0→1.5）|
| fidelity | 1.0 | 提示词中的颜色/形状词出现在 SVG 中 |
| **anti_template** | **1.0** | **（v2 新增）** 非模板化输出 |

**设计取向**：
- 结构合法性权重最高（3.5，含解析+属性+viewBox），解析失败的 SVG 一切免谈。
- 几何合理性 2.0（硬边界 1.5 + 软边界 0.5）。
- 安全性 2.0（非法标签 + 外链）。
- 内容丰富度 3.5（配色 1.0 + 密度 1.0 + 非退化 1.5）。
- 提示词保真度 1.0，权重适中——270M 语言理解有限，过高引入噪声。
- **反模板化 1.0（v2 新增）**：对抗 Goodhart 效应，防止模型刷分而非学习真实生成能力。

**容错设计（`_salvage_xml` 5 阶段修复管道）**：
270M 小模型无法可靠闭合 XML 标签，若一次解析失败就归零会系统性低估模型能力。reward 内置多阶段修复：
1. 剥离 `<defs>` 块（非必要且常未闭合）
2. 自闭合 void 元素（`<rect ...>` → `<rect .../>`）
3. **栈式**修复 `<g>` 嵌套（`_fix_group_nesting`：按文档顺序扫描，丢弃孤立 `</g>`，补齐未闭合 `<g>`）
4. 移除属性间逗号（`x="1", y="2"` → `x="1" y="2"`）
5. 标签内同名属性去重

**关键教训（栈 vs 计数）**：早期版本用计数法（`<g>` 开 vs `</g>` 闭的数量比较）修复嵌套，但无法检测**孤立的 `</g>`**——1 个 `<g>` 开 + 1 个孤立 `</g>` 闭 = 1==1，误判为已平衡，导致解析失败 reward=0。必须用栈按文档顺序扫描。

### 2.1 v2 新增：`anti_template` 反模板化维度

**背景**：v1 版本中，微调模型学到了「单色居中重复圆」安全模板——所有 circle 的 cx/cy/r 完全相同，reward 函数仍能拿到结构分 + 坐标分。这是典型的 Goodhart 效应：代理指标上去了，真实质量却没跟上。

**实现**：`_is_template_like()` 检测两种模板化模式：
1. **相同 circle 模板**：所有 `<circle>` 元素的 cx/cy/r 属性完全相同
2. **形状签名坍缩**：所有形状元素（去属性后）签名去重后只剩 1 种

检测到任一模式时，`anti_template=0.0`，否则 `anti_template=1.0`。

---

## 三、训练配置

| 参数 | 取值 | 说明 |
|------|------|------|
| 基座模型 | Gemma3 270M (gemma-3-270m-it) | |
| LoRA rank | 16 | v2: 8→16，增加容量学习复杂形状组合 |
| LoRA alpha | 32 | 保持 alpha = 2 × rank |
| LoRA dropout | 0.1 | v2: 0.05→0.1，抑制过拟合到「安全模板」捷径 |
| **目标模块** | q_proj, k_proj, v_proj, o_proj, **gate_proj, up_proj, down_proj** | v2: 新增 MLP 层，可训练参数 737K→3.80M |
| 精度 | bf16 | fp16 在此模型上 loss=0/grad_norm=NaN（精度下溢）|
| batch size | 1 | RTX 3060 (6GB) 放不下 batch=2 |
| 梯度累积 | 8 | 等效 batch size = 8 |
| 学习率 | 1e-4 | v2: 2e-4→1e-4，降低学习率减少过拟合到简单模式 |
| epochs | 2 | v2: 4→2，减少训练轮数，避免学到「安全模板」捷径 |
| 最大序列长度 | 2048 | v2: 1536→2048，保留 96% 样本（vs 89%）|
| 早停 | patience=3 | |
| 可训练参数 | **3,796,992 (1.3965%)** | v2: 737,280→3,796,992 |
| weight_decay | 0.05 | v2: 0.01→0.05，增加权重衰减正则化 |
| warmup_ratio | 0.15 | v2: 0.1→0.15，更长 warmup 稳定初期训练 |
| 训练前 `flatten_svg` | 是 | 剥离 `<defs>`/gradient + **坐标 clamp**（v2 新增）|

**关键训练决策**：
- **超长样本跳过而非截断**：截断会从文档中间砍掉 `</svg>`，教模型输出不闭合文档——这是致命 bug。token 长度审计显示训练 SVG 中位数 872、均值 986、max 3153 tokens。
- **`flatten_svg` 数据预处理**：剥离训练目标里的 `<defs>`/`<linearGradient>`/`<radialGradient>`，把 `url(#id)` 替换为从 gradient 首个 stop 提取的纯色。270M 无法可靠关闭嵌套结构，简化后只剩「形状 + 纯色」。
- **v2 新增：坐标 clamp**：训练数据中存在 `x=-9999` 等极端坐标（SVG 覆盖背景技巧），模型会学到并滥用（输出 `x=-9999 width=128` 的无效 rect）。`flatten_svg` 将 x/y/cx/cy 等 clamp 到 [0, 256]，width/height clamp 到 [1, 256]。
- **训练/eval prompt 对齐**：训练用 `apply_chat_template(..., add_generation_prompt=True)`，与 eval 生成前缀完全一致；SVG 后追加 EOS 教模型停止。
- **v2: max_length 选 2048**：审计 token 分布，`<=2048` 占 96%。配合 lr 降低和 epochs 减半，训练 8.4 分钟完成。

---

## 四、训练过程

### Loss 曲线（v2 新训练）

| Step | Epoch | Train Loss | Grad Norm | LR |
|------|-------|-----------|-----------|-----|
| 5 | 0.19 | 1.295 | — | 9.6e-5 |
| 10 | 0.38 | 1.082 | — | 1.0e-4 |
| 15 | 0.58 | 0.974 | — | 9.8e-5 |
| 24 | 0.92 | 0.923 | — | 8.4e-5 |
| 25 | 0.96 | — | — | eval_loss=0.839 |
| 35 | 1.35 | 0.910 | — | 6.0e-5 |
| 46 | 1.77 | 0.902 | — | 2.8e-5 |
| 50 | 1.92 | — | — | eval_loss=0.839 |
| 52 | 2.00 | 0.955 (avg) | — | 0 |
| 52 | 2.00 | — | — | eval_loss=0.838 |

**观察**：
- 训练 loss 从 1.295 稳步下降到 0.902，降幅 30%。
- eval_loss 从 1.295 → 0.839 → 0.838，2 轮后收敛，未见过拟合（得益于 epochs=2 + dropout=0.1 + weight_decay=0.05）。
- RTX 3060 Laptop 上总训练时间约 8.4 分钟（max_length=2048 + MLP 层）。

---

## 五、评测结果

### 评测方法（v2 优化后）

| 参数 | v1 | v2 | 理由 |
|------|-----|----|----|
| num_samples | 2 (取最佳) | **1 (单次)** | 消除统计偏向，诚实反映模型能力 |
| 退化 retry | 有 | **无** | 退化输出由 reward 自然惩罚 |
| system prompt | 泄漏阈值 | **无泄漏** | 不再包含精确坐标区间/颜色数 |
| temperature | 0.2 | **0.3** | 适度随机性避免模型过度偏好安全模板 |
| repetition_penalty | 1.05 | **1.1** | 抑制退化循环 |

### 最终结果

| 指标 | 基座 | 微调 | Delta |
|------|------|------|-------|
| 平均 reward | 0.6898 | **0.8360** | **+0.1461** |
| 有效 SVG 数 | 15/17 | 17/17 | +2 |
| 有效率 | 88.2% | 100% | +11.8% |

**微调以 +0.1461 的优势真实超越基座**。

### 优化前后对比

| 版本 | 配置 | Baseline | Fine-tuned | Delta | 说明 |
|------|------|----------|------------|-------|------|
| v1 | 旧 adapter + 旧 eval | 0.6190 | 0.8013 | +0.1823 | Goodhart 假象 |
| v2 | 旧 adapter + 新 eval | 0.7527 | 0.6340 | -0.1186 | 暴露真相 |
| **v3** | **新 adapter + 新 eval** | **0.6898** | **0.8360** | **+0.1461** | **真实提升** |

**演进解读**：
- v1→v2：同一 adapter 换诚实评测，delta 从 +0.18 变 -0.12，说明 v1 的提升主要来自评测偏向（多采样取最佳 + retry + 阈值泄漏），不是模型能力。
- v2→v3：同一诚实评测，换新训练的 adapter，delta 从 -0.12 变 +0.15，说明新训练策略（MLP 层 + lr 降低 + epochs 减半 + 坐标 clamp）真正提升了模型的 SVG 生成能力。

### 各样本得分（v3）

**Fine-tuned**（17/17 有效）：

| 样本 | reward | anti_template | non_degenerate | palette | density |
|------|--------|---------------|----------------|---------|---------|
| FT[3] | 0.969 | 1.0 | 1.0 | 1.0 | 1.0 |
| FT[17] | 0.946 | 1.0 | 1.0 | 0.667 | 1.0 |
| FT[6] | 0.956 | 1.0 | 1.0 | 1.0 | 1.0 |
| FT[5] | 0.939 | 1.0 | 1.0 | 1.0 | 1.0 |
| FT[9] | 0.936 | 1.0 | 1.0 | 0.667 | 1.0 |
| FT[15] | 0.928 | 1.0 | 1.0 | 0.667 | 1.0 |
| FT[16] | 0.917 | 1.0 | 1.0 | 0.667 | 1.0 |
| FT[10] | 0.910 | 1.0 | 1.0 | 0.667 | 0.667 |
| FT[2] | 0.822 | 0.0 | 1.0 | 0.667 | 0.667 |
| FT[13] | 0.936 | 1.0 | 1.0 | 0.667 | 1.0 |
| FT[1] | 0.725 | 0.0 | 0.0 | 0.333 | 1.0 |
| FT[4] | 0.679 | 0.0 | 0.0 | 0.333 | 0.667 |
| FT[7] | 0.728 | 1.0 | 0.0 | 0.333 | 0.333 |
| FT[8] | 0.699 | 0.0 | 0.0 | 0.333 | 1.0 |
| FT[11] | 0.699 | 0.0 | 0.0 | 0.333 | 1.0 |
| FT[12] | 0.728 | 1.0 | 0.0 | 0.333 | 0.333 |
| FT[14] | 0.695 | 0.0 | 0.0 | 0.333 | 1.0 |

**观察**：
- 11/17 样本 `anti_template=1.0`（非模板化），其中 8 个拿到 0.91+ 高分。
- 6/17 样本 `anti_template=0.0`（模板化），分数集中在 0.68-0.82，说明 `anti_template` 维度有效区分了真实生成 vs 模板刷分。
- 坐标越界问题已解决（得益于 `flatten_svg` 坐标 clamp）。

---

## 六、开发踩坑与分析

### 6.1 `conda run` 输出缓冲导致误判"卡住"

**现象**：用 `conda run -n lora_env python ...` 跑训练/自评时，stdout 被完全缓冲，进度日志不刷新。进程实际在正常运行（GPU 100%、显存 1231MiB、CPU 累积 1414s），但看起来像"卡住"。

**根因**：`conda run` 默认缓冲 stdout，不像直接 `python -u` 那样实时刷新。

**修复**：改用 `conda run -n lora_env --no-capture-output python script.py`，`--no-capture-output` 禁用缓冲，实时可见进度。

### 6.2 max_length 选择需基于 token 分布审计

**现象**：首轮用 max_length=1024 训练（83 样本，38%），自评结果微调 0.5098 < 基座 0.5189，回归。

**根因**：token 分布审计显示 `<=1024` 仅占 63%（138 样本），但实际加上 prompt 前缀后能保留的更少（83 样本，38%）。模型只学到短 SVG，泛化不足。

**修复**：v1 提到 1536（89%），v2 进一步提到 2048（96%），配合 lr 降低和 epochs 减半保持训练稳定。

### 6.3 生成退化循环（v1 核心问题）

**现象**：smoke test 发现 temp=0.3 + repetition_penalty=1.15 时，模型陷入病态重复循环：
```
<path d="M10 15 L15 20 L20 25 L30 30 L40 35 L50 35 L60 35 L70 35 L80 35 L90 35 L100 35..."
```
模型撞满 1024 tokens（45s）才停，整段垃圾 reward=0。

**v1 应对**（已被 v2 推翻）：
1. temperature 0.3→0.2
2. repetition_penalty 1.15→1.05
3. 多采样取最佳（num_samples=2）
4. 退化检测 + retry

**v2 反思**：v1 的多采样取最佳和退化 retry 是**统计偏向**——系统性抬高 fine_tuned 分数而对 baseline（全空 SVG）无影响，人为放大 delta。这导致 v1 的 +0.1823 delta 成为 Goodhart 假象。

**v2 正确做法**：单次采样 + 无 retry + 适度温度（0.3）+ 适度重复惩罚（1.1），让退化输出由 reward 函数的 `anti_template` 和 `non_degenerate` 维度自然惩罚，不在评测侧人为干预。

### 6.4 reward 函数设计要点

- **xmlns 检测**：`xml.etree.ElementTree` 把 `xmlns` 当命名空间声明消费掉，从 `root.attrib` 中移除。必须检查原始文本或命名空间标签前缀，否则每个合法 SVG 都丢分。
- **crash-safe 坐标解析**：小模型频繁输出非数值属性（`cx="auto"`），`float()` 会崩溃。用 `_safe_float` 返回默认值。
- **保真度度量收窄**：不把长描述提示词中所有英文词都拿来匹配（几乎全不命中），而是只度量两个有意义的信号：提示词命名的颜色是否出现在 `fill`/`stroke` 中；提示词的形状词是否作为 SVG 标签出现。
- **v2: anti_template 维度**：检测「单色居中重复圆」安全模板，对抗 Goodhart 效应。

### 6.5 Goodhart 效应与 v2 优化（核心教训）

**v1 的 Goodhart 假象**：
v1 报告 delta=+0.1823，看似优秀。但深入分析发现：
1. **多采样取最佳**：每个 prompt 采样 2 次取 reward 最高，对 fine_tuned 是正向偏向，对 baseline（全空 SVG）无影响。
2. **退化 retry**：采样到退化立即 retry 直到拿到 reward>0 的输出，人为丢弃了失败样本。
3. **system prompt 泄漏阈值**：v1 的 system prompt 包含 `20..236`、`2 to 12 colors` 等精确阈值，模型学到「踩分模板」而非真实生成。

**v2 验证**：
用同一 v1 adapter 换诚实评测（v2），delta 从 +0.18 变 -0.12，证明 v1 的提升主要来自评测偏向，不是模型能力。微调模型确实学到了「单色居中重复圆」安全模板，而非真正的 SVG 生成能力。

**v2 优化措施**：
1. **reward.py**：收紧评分标尺（PALETTE_MIN 2→3, SHAPE_MIN 2→3, non_degenerate 权重 1.0→1.5），新增 `anti_template` 反模板化维度。
2. **eval_self.py**：移除多采样取最佳和退化 retry，移除 system prompt 阈值泄漏。
3. **train_peft.py**：`flatten_svg` 坐标 clamp，消除极端坐标学习源。
4. **train_config.yaml**：加入 MLP 层（参数 737K→3.80M），lr 降低（2e-4→1e-4），epochs 减半（4→2），dropout 加倍（0.05→0.1），weight_decay 5 倍（0.01→0.05）。

**v3 结果**：delta 从 -0.12 变 +0.15，真实提升。坐标越界问题已解决，模板化样本减少（11/17 anti_template=1.0），8/17 样本获得 0.91+ 高分。

### 6.6 训练数据坐标 clamp（v2 新增）

**现象**：v2 评测发现微调模型频繁输出 `x="-9999"` 等极端坐标。

**根因**：审计训练数据发现样本 2 有 `<rect x="-9999" y="-9999" width="19998" height="19998">`（SVG 覆盖背景技巧）。模型学到了这个模式但用错了——输出 `x="-9999" width="128"`（完全在画布外）。

**修复**：`flatten_svg` 中增加坐标 clamp 逻辑，将 x/y/cx/cy 等 clamp 到 [0, 256]，width/height clamp 到 [1, 256]。模型不再有机会学到极端坐标。

---

## 七、输出示例

### 基座（v3，有效但平凡，reward=0.6898 均值）
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect x="0" y="0" width="256" height="256" fill="white" />
  <circle cx="0" cy="0" r="3" fill="yellow" />
  <ellipse cx="0" cy="0" rx="1" ry="1" fill="red" />
  ...
</svg>
```
v3 用 temperature=0.3 评测，基座获得随机性，能生成实际 SVG 内容（v1 的 0.2 低温下基座只会输出空 SVG）。但坐标经常越界，元素重复，拿不到高分。

### 微调高质量样本（v3，reward=0.969）
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect x="0.0" y="0.0" width="256.0" height="256.0" fill="#f8b9a7"/>
  <circle cx="128" cy="128.0" r="128" fill="none" stroke="#d3c3c3" stroke-width="2.0" opacity="0.5"/>
  <g><path d="M128 128 C..."/></g>
  ...
</svg>
```
完整闭合、多形状多色、居中。坐标 clamp 后不再有越界问题。`anti_template=1.0`，`non_degenerate=1.0`，`palette=1.0`，`density=1.0`。

### 微调模板化样本（v3，reward=0.679，anti_template=0.0）
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <circle cx="128" cy="128.0" r="128.0" fill="#FFFFFF"/>
  <circle cx="128" cy="128.0" r="128.0" fill="#FFFFFF"/>
  <g></svg>
```
所有 circle 的 cx/cy/r 完全相同，被 `_is_template_like()` 检测到，`anti_template=0.0`。这类样本虽然有结构分，但被反模板化维度惩罚，分数 0.679 远低于真实生成样本。

---

## 八、总结

微调以 **+0.1461** 真实超越基座（0.8360 vs 0.6898），17/17 全有效。提升来自真实的 SVG 生成能力：基座只会输出平凡 SVG（部分有效但内容单调），微调学会了产出有形状、有配色、完整闭合的 SVG。

**关键收获**：
1. **Goodhart 效应是 reward 函数的固有风险**——v1 的 +0.1823 delta 是评测偏向（多采样取最佳 + retry + 阈值泄漏）造成的假象，v2 诚实评测后变 -0.1186。必须警惕代理指标与真实目标的偏离。
2. **评测脚本的中立性至关重要**——多采样取最佳、退化 retry、system prompt 泄漏阈值都会系统性偏向 fine_tuned，必须在评测侧消除人为干预。
3. **训练数据的隐性 bug 会传染模型**——训练数据中的 `x=-9999` 极端坐标被模型学到并滥用，`flatten_svg` 坐标 clamp 是必要的预处理。
4. **仅训练 attention 不足以学习 SVG 生成**——v2 加入 MLP 层（gate/up/down_proj），参数量 737K→3.80M，模型容量增加 5 倍，才学到真正的形状组合能力。
5. **过拟合到「安全模板」比欠拟合更危险**——v1 的 lr=2e-4 + epochs=4 让模型找到了刷分捷径。v2 降低 lr（1e-4）、减半 epochs（2）、加倍 dropout（0.1）、5 倍 weight_decay（0.05），让模型学到的不是捷径而是真实能力。
6. **XML 容错必须用栈**，计数法无法检测孤立的闭合标签。
7. **训练数据预处理（flatten_svg + 坐标 clamp）**对小模型至关重要——剥离嵌套结构 + 清理极端坐标后模型才能学会正确闭合。
8. **`conda run` 的输出缓冲会误判为"卡住"**——必须用 `--no-capture-output` 参数。
9. **max_length 选择需基于 token 分布审计**——1024 只保留 38% 样本导致回归，2048 保留 96% 才够。

---

## 九、交付物清单

| 文件 | 说明 |
|------|------|
| `adapter/` | LoRA 适配器权重（r=16, alpha=32, target=q,k,v,o,gate,up,down，3.80M 可训练参数）|
| `student_kit/reward.py` | LogoGrader：12 维度打分 + `_salvage_xml` 5 阶段容错 + `_fix_group_nesting` 栈式扫描 + `_is_template_like` 反模板化 |
| `student_kit/train_peft.py` | 训练脚本（flatten_svg + 坐标 clamp、跳过不截断、追加 EOS、add_generation_prompt=True、MLP 层）|
| `student_kit/eval_self.py` | 自评脚本（单次采样、无 retry、无阈值泄漏、temp=0.3、rep_penalty=1.1、extract_svg 容错）|
| `student_kit/train_config.yaml` | 训练超参数（v2 优化版）|
| `results.json` | 完整评测结果（17 样本，基座 0.6898 vs 微调 0.8360，delta=+0.1461，17/17 有效）|
| `report.md` | 端到端开发流程报告（含 v1→v2→v3 演进分析）|
| `DEVELOPMENT_GUIDE.md` | 新手开发全流程指南（944 行）|
