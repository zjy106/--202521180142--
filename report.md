# Gemma3 270M LoRA 微调：SVG 徽标生成实验报告

## 作者信息
- 学号：202521180142
- 姓名：仲瑾毓
- 完成日期：2026-07-09
- v2 优化日期：2026-07-15

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
| palette | 1.0 | 3~12 种不同颜色（v2: 2→3）|
| density | 1.0 | 3~80 个 SVG 元素（v2: 2→3）|
| non_degenerate | 1.5 | 非空、非单色退化输出（v2: 1.0→1.5）|
| anti_template | 1.0 | 非模板化输出（**v2 新增**）|
| fidelity | 1.0 | 提示词中的颜色/形状词出现在 SVG 中 |

**设计取向**：
- 结构合法性权重最高（3.5，含解析+属性+viewBox），解析失败的 SVG 一切免谈。
- 几何合理性 2.0（硬边界 1.5 + 软边界 0.5）。
- 安全性 2.0（非法标签 + 外链）。
- 内容丰富度 4.5（配色 + 密度 + 非退化 + 反模板化）。
- 提示词保真度 1.0，权重适中——270M 语言理解有限，过高引入噪声。

### v2 新增：反模板化维度 `anti_template`

**动机**：v1 评测发现微调模型学到了"单色居中重复圆"安全模板——所有 `<circle>` 的 cx/cy/r 完全相同，重复 2-3 次刷分。这是典型的 Goodhart 效应：reward 分数上去了，真实生成能力却没跟上。

**检测逻辑** `_is_template_like()`：
- 收集所有 `<circle>` 的 (cx, cy, r) 三元组，若去重后只剩 1 种 → 判定为模板化
- 收集所有形状元素的签名（标签+属性序列），若去重后只剩 1 种 → 判定为模板化
- 命中任一条件，`anti_template=0.0`，否则 `anti_template=1.0`

**容错设计（`_salvage_xml` 5 阶段修复管道）**：
270M 小模型无法可靠闭合 XML 标签，若一次解析失败就归零会系统性低估模型能力。reward 内置多阶段修复：
1. 剥离 `<defs>` 块（非必要且常未闭合）
2. 自闭合 void 元素（`<rect ...>` → `<rect .../>`）
3. **栈式**修复 `<g>` 嵌套（`_fix_group_nesting`：按文档顺序扫描，丢弃孤立 `</g>`，补齐未闭合 `<g>`）
4. 移除属性间逗号（`x="1", y="2"` → `x="1" y="2"`）
5. 标签内同名属性去重

**关键教训（栈 vs 计数）**：早期版本用计数法（`<g>` 开 vs `</g>` 闭的数量比较）修复嵌套，但无法检测**孤立的 `</g>`**——1 个 `<g>` 开 + 1 个孤立 `</g>` 闭 = 1==1，误判为已平衡，导致解析失败 reward=0。必须用栈按文档顺序扫描。

---

## 三、训练配置

| 参数 | 取值 | 说明 |
|------|------|------|
| 基座模型 | Gemma3 270M (gemma-3-270m-it) | |
| LoRA rank | 16 | v2: 8→16，增加 LoRA 容量学习复杂形状组合 |
| LoRA alpha | 32 | 保持 alpha = 2 × rank |
| LoRA dropout | 0.1 | v2: 0.05→0.1，增强正则化抑制"安全模板"捷径 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, **gate_proj, up_proj, down_proj** | v2 新增 MLP 层 |
| 精度 | bf16 | fp16 在此模型上 loss=0/grad_norm=NaN（精度下溢）|
| batch size | 1 | RTX 3060 (6GB) 放不下 batch=2 |
| 梯度累积 | 8 | 等效 batch size = 8 |
| 学习率 | 1e-4 | v2: 2e-4→1e-4，降低学习率减少过拟合到简单模式 |
| epochs | 2 | v2: 4→2，减少训练轮数避免学到"安全模板"捷径 |
| 最大序列长度 | 2048 | v2: 1536→2048，保留 96% 样本（vs 89%）|
| 早停 | patience=3 | |
| 可训练参数 | 3,796,992 (1.3965%) | v2: 737,280→3,796,992（加入 MLP 层后 5 倍增长）|
| weight_decay | 0.05 | v2: 0.01→0.05，增加权重衰减正则化 |
| warmup_ratio | 0.15 | v2: 0.1→0.15，更长 warmup 稳定初期训练 |
| 训练前 `flatten_svg` | 是 | 剥离 `<defs>`/gradient + **v2: 坐标 clamp** |

### v2 关键训练决策

- **超长样本跳过而非截断**：截断会从文档中间砍掉 `</svg>`，教模型输出不闭合文档——这是致命 bug。token 长度审计显示训练 SVG 中位数 872、均值 986、max 3153 tokens。
- **`flatten_svg` 数据预处理**：剥离训练目标里的 `<defs>`/`<linearGradient>`/`<radialGradient>`，把 `url(#id)` 替换为从 gradient 首个 stop 提取的纯色。270M 无法可靠关闭嵌套结构，简化后只剩「形状 + 纯色」。
- **v2 新增：坐标 clamp**：审计发现训练数据中存在 `<rect x="-9999" y="-9999" width="19998">`（SVG 覆盖背景技巧），v1 模型学到了这个 `x=-9999` 模式但滥用为 `x="-9999" width="128"`（完全在画布外）。v2 在 `flatten_svg` 中将 x/y/cx/cy 等坐标 clamp 到 [0, 256]，width/height clamp 到 [1, 256]，从源头消除极端坐标学习源。
- **v2 新增：扩展 target_modules 到 MLP 层**：仅训练 attention 的 q/k/v/o 不足以学习 SVG 生成，加入 MLP 的 gate/up/down 投影后可训练参数从 737K 增至 3.80M（5 倍增长），模型容量大幅提升。
- **训练/eval prompt 对齐**：训练用 `apply_chat_template(..., add_generation_prompt=True)`，与 eval 生成前缀完全一致；SVG 后追加 EOS 教模型停止。

---

## 四、训练过程

### v2 训练 Loss 曲线

| Step | Epoch | Train Loss | Grad Norm | LR |
|------|-------|-----------|-----------|-----|
| 初始 | 0 | 1.295（前向测试）| — | — |
| 5 | 0.19 | ~1.10 | — | 8.6e-5 |
| 25 | 0.96 | ~0.95 | — | 6.0e-5 |
| 52 | 2.00 | 0.9555 | — | 0 |
| 52 | 2.00 | — | — | eval_loss=0.8382 |

**观察**：
- 训练 loss 从 1.295 下降到 0.9555，降幅 26%。
- eval_loss 0.8382，相比 v1 的 0.946 下降 11.5%。
- RTX 3060 Laptop 上总训练时间约 8 分 26 秒（max_length=2048 + MLP 层）。
- v1 训练 4.8 分钟，v2 训练 8.4 分钟——时间增长来自更大的模型容量（MLP 层）和更长的序列（2048 vs 1536）。

---

## 五、评测结果

### 三版本对比（核心结果）

| 指标 | v1 (旧 adapter + 旧 eval) | v2 (旧 adapter + 新 eval) | v3 (新 adapter + 新 eval) |
|------|---------------------------|---------------------------|---------------------------|
| Baseline avg | 0.6190 | 0.7527 | 0.6898 |
| Fine-tuned avg | 0.8013 | 0.6340 | **0.8360** |
| Delta | +0.1823 (Goodhart 假象) | -0.1186 (暴露真相) | **+0.1461 (真实提升)** |
| Baseline valid | 17/17 | 16/17 | 15/17 |
| Fine-tuned valid | 17/17 | 17/17 | 17/17 |

### v3 最终结果

| 指标 | 基座 | 微调 | Delta |
|------|------|------|-------|
| 平均 reward | 0.6898 | **0.8360** | **+0.1461** |
| 有效 SVG 数 | 15/17 | 17/17 | +2 |
| 有效率 | 88.2% | 100% | +11.8% |

**微调以 +0.1461 的优势真实超越基座**。这是在消除所有评测偏向（多采样取最佳、退化 retry、阈值泄漏）后的纯净结果。

### 生成参数（v2 优化后）

- temperature: 0.3（v1: 0.2→v2: 0.3，适度随机性避免模板化）
- top_p: 0.9
- repetition_penalty: 1.1（v1: 1.05→v2: 1.1，抑制退化循环）
- max_new_tokens: 1024
- do_sample: true
- **num_samples: 1**（v1: 2→v2: 1，单次采样诚实反映模型能力）
- **无退化 retry**（v2 移除，由 reward 自然惩罚）

### v3 各样本得分分析

**Fine-tuned（17/17 有效，8/17 获 0.9+ 高分）**：

| 样本 | reward | anti_template | non_degenerate | 说明 |
|------|--------|---------------|---------------|------|
| FT[3] | 0.969 | 1.0 | 1.0 | 最高分，多形状多色 |
| FT[17] | 0.946 | 1.0 | 1.0 | |
| FT[6] | 0.956 | 1.0 | 1.0 | |
| FT[5] | 0.939 | 1.0 | 1.0 | |
| FT[15] | 0.928 | 1.0 | 1.0 | |
| FT[16] | 0.917 | 1.0 | 1.0 | |
| FT[9,13] | 0.936 | 1.0 | 1.0 | |
| FT[10] | 0.910 | 1.0 | 1.0 | |
| FT[1,4,7,8,11,14] | 0.68-0.73 | 0.0 | 0.0 | 仍有模板化倾向 |

**仍存在的问题（诚实记录）**：6/17 样本 `anti_template=0.0`，模型在这些样本上仍退化到重复圆模式。部分样本颜色多样性有限（palette=0.333）。这是 270M 容量限制和训练数据复杂度的客观瓶颈。

---

## 六、开发踩坑与分析

### 6.1 `conda run` 输出缓冲导致误判"卡住"

**现象**：用 `conda run -n lora_env python ...` 跑训练/自评时，stdout 被完全缓冲，进度日志不刷新。进程实际在正常运行（GPU 100%、显存 1231MiB、CPU 累积 1414s），但看起来像"卡住"。

**根因**：`conda run` 默认缓冲 stdout，不像直接 `python -u` 那样实时刷新。

**修复**：改用 `conda run --no-capture-output` 或直接 python.exe 路径调用 `& C:\Users\dtft\miniconda3\envs\lora_env\python.exe -u script.py`，禁用缓冲，实时可见进度。

### 6.2 max_length=1024 导致训练数据不足

**现象**：首轮用 max_length=1024 训练（83 样本，38%），自评结果微调 0.5098 < 基座 0.5189，回归。

**根因**：token 分布审计显示 `<=1024` 仅占 63%（138 样本），但实际加上 prompt 前缀后能保留的更少（83 样本，38%）。模型只学到短 SVG，泛化不足。

**修复**：max_length 提到 1536（v1）/ 2048（v2），保留 89%/96% 样本，eval_loss 从 1.106 降到 0.946（v1）/ 0.8382（v2）。

### 6.3 生成退化循环（v1 核心问题，v2 重新审视）

**v1 现象**：smoke test 发现 temp=0.3 + repetition_penalty=1.15 时，模型陷入病态重复循环。17 样本 × 2 阶段 × 45s ≈ 25 分钟，全程 GPU 满载但产出无效。

**v1 修复（事后证明是 Goodhart 风险）**：
1. `temperature` 0.3 → 0.2
2. `repetition_penalty` 1.15 → 1.05
3. **多采样取最佳**：每个 prompt 采样 2 次取 reward 最高
4. **退化检测 + retry**：`_is_degenerate()` 检测退化并重采样

**v2 重新审视（关键教训）**：v1 的"修复"实际上是引入了三层 Goodhart 偏向：
- **多采样取最佳**系统性抬高 fine_tuned 分数（baseline 全空 SVG 无 benefit）
- **退化 retry** 丢弃了模型真实产出的退化样本，只保留"运气好"的
- **system prompt 泄漏 reward 阈值**（`20..236`、`2 to 12 colors`）让模型学到"踩分模板"

v2 评测移除全部三层偏向后，v1 adapter 的真实分数暴露：delta 从 +0.1823 跌到 -0.1186，**微调实际上比基座差**。这说明 v1 的"成功"是评测造假的产物。

### 6.4 reward 函数设计要点

- **xmlns 检测**：`xml.etree.ElementTree` 把 `xmlns` 当命名空间声明消费掉，从 `root.attrib` 中移除。必须检查原始文本或命名空间标签前缀，否则每个合法 SVG 都丢分。
- **crash-safe 坐标解析**：小模型频繁输出非数值属性（`cx="auto"`），`float()` 会崩溃。用 `_safe_float` 返回默认值。
- **保真度度量收窄**：不把长描述提示词中所有英文词都拿来匹配（几乎全不命中），而是只度量两个有意义的信号：提示词命名的颜色是否出现在 `fill`/`stroke` 中；提示词的形状词是否作为 SVG 标签出现。

### 6.5 Goodhart 效应与 v2 优化（核心教训）

**Goodhart 定律**："当一个度量成为目标时，它就不再是好的度量。"

**v1 的 Goodhart 三层偏向**：
1. **多采样取最佳**（num_samples=2）：每个 prompt 采样 2 次取 reward 最高。这系统性抬高 fine_tuned 分数而对 baseline（全空 SVG）无影响，人为放大 delta。
2. **退化检测 + retry**：丢弃模型真实产出的退化样本，重采样直到拿到有效输出。这把"模型不会做"伪装成"模型会做"。
3. **system prompt 泄漏 reward 阈值**：system prompt 包含 `20..236`、`2 to 12 colors` 等精确阈值，模型学到"踩分模板"而非真实生成能力。

**v2 优化措施**：

| 层面 | v1 | v2 |
|------|-----|-----|
| **reward.py** | 颜色阈值 2，元素阈值 2，non_degenerate 权重 1.0 | 颜色阈值 3，元素阈值 3，non_degenerate 权重 1.5，新增 anti_template 维度 |
| **eval_self.py** | num_samples=2 + retry + 阈值泄漏 | num_samples=1，无 retry，无阈值泄漏 |
| **train_peft.py** | flatten_svg 无坐标处理 | flatten_svg 坐标 clamp 到 [0,256] |
| **train_config.yaml** | r=8, dropout=0.05, lr=2e-4, epochs=4, 仅 attention | r=16, dropout=0.1, lr=1e-4, epochs=2, 加入 MLP 层 |

**v2 结果验证**：
- v2（旧 adapter + 新 eval）：delta -0.1186 → 暴露 v1 adapter 的真实质量
- v3（新 adapter + 新 eval）：delta +0.1461 → 证明 v2 训练优化有效

**核心教训**：评测脚本的"善意"（retry、多采样、阈值提示）会系统性地制造假阳性。诚实的评测应该让模型一次产出、一次打分，reward 函数本身负责惩罚退化。训练优化应该针对真实能力，而非针对评测漏洞。

---

## 七、输出示例

### v3 基座（有效但内容有限，reward=0.6898 平均）

基座在 temperature=0.3 下偶尔生成有内容的 SVG，但仍以平凡输出为主：
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect x="0" y="0" width="256" height="256" fill="white"/>
  <circle cx="0" cy="0" r="3" fill="yellow"/>
  ...
</svg>
```
部分样本因坐标越界或结构错误被判无效（15/17 valid）。

### v3 微调高质量样本（reward=0.969，FT[3]）

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect x="0.0" y="0.0" width="256.0" height="256.0" fill="#f8b9a7"/>
  <circle cx="128" cy="128.0" r="128" fill="none" stroke="#d3c3c3" stroke-width="2.0" opacity="0.5"/>
  <g><path d="M128 128 C..."/>...</g>
</svg>
```
完整闭合、坐标在界、多形状多色、非模板化（anti_template=1.0）。**坐标 clamp 生效**——所有坐标都在 [0, 256] 内，不再出现 v1 的 `x=-9999` 问题。

### v3 微调模板化样本（reward=0.725，FT[1]）

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect x="0.0" y="0.0" width="256.0" height="256.0"/>
  <circle cx="128" cy="128.0" r="128.0" fill="#FFFFFF"/>
  <circle cx="128" cy="128.0" r="128.0" fill="#ffffff"/>
  <circle cx="128" cy="128.0" r="128.0" fill="#FFFFFF"/>
  <path d="M128 128 L128 96 L128 100..." fill="#FFFFFF"/>
</svg>
```
被判 `anti_template=0.0`（多个相同 circle）和 `non_degenerate=0.0`（单色 #FFFFFF）。reward 函数正确识别了模板化输出并惩罚。**这是 v2 反模板化维度的设计目标**——诚实度量而非虚假高分。

---

## 八、总结

### 三版本演进总结

| 版本 | Delta | 评价 |
|------|-------|------|
| v1 | +0.1823 | Goodhart 假象——评测偏向（多采样+retry+阈值泄漏）人为放大分数 |
| v2 | -0.1186 | 真相暴露——移除评测偏向后，v1 adapter 实际比基座差 |
| v3 | **+0.1461** | **真实提升**——新 adapter（MLP 层+坐标 clamp+正则化）+ 诚实评测 |

**关键收获**：
1. **Goodhart 效应是真实风险**——评测脚本的"善意"（retry、多采样、阈值提示）会系统性制造假阳性。v1 的 +0.1823 完全是评测造假的产物。
2. **诚实评测是基础**——v2 移除三层偏向后暴露 v1 adapter 真实质量（-0.1186），这是优化的起点。
3. **训练数据质量决定模型上限**——`flatten_svg` 的坐标 clamp 消除了 `x=-9999` 滥用，从源头提升模型质量。
4. **模型容量需要匹配任务复杂度**——仅训练 attention 的 q/k/v/o（737K 参数）不足以学习 SVG 生成，加入 MLP 层（3.80M 参数）后模型展现出真实的多形状多色生成能力。
5. **过拟合到"安全模板"是隐性风险**——v1 的 lr=2e-4 + epochs=4 导致模型学到"单色居中重复圆"捷径，v2 通过 lr 减半、epochs 减半、dropout/weight_decay 加倍来抑制。
6. **reward 函数需要持续演进**——v2 新增的 `anti_template` 维度能检测模板化输出，这是对抗 Goodhart 效应的最后一道防线。
7. **`conda run` 的输出缓冲会误判为"卡住"**——必须用 `--no-capture-output` 或 `python -u`。
8. **max_length 选择需基于 token 分布审计**——1024 只保留 38% 样本导致回归，2048 保留 96% 才够。
9. **XML 容错必须用栈**，计数法无法检测孤立的闭合标签。

---

## 九、交付物清单

| 文件 | 说明 |
|------|------|
| `adapter/` | LoRA 适配器权重（v2: r=16, alpha=32, target=q,k,v,o,gate,up,down，3.80M 参数）|
| `student_kit/reward.py` | LogoGrader：12 维度打分（含 v2 新增 `anti_template`）+ `_salvage_xml` 5 阶段容错 + `_fix_group_nesting` 栈式扫描 |
| `student_kit/train_peft.py` | 训练脚本（flatten_svg 预处理 + **v2: 坐标 clamp**、跳过不截断、追加 EOS、add_generation_prompt=True）|
| `student_kit/eval_self.py` | 自评脚本（**v2: 单次采样、无 retry、无阈值泄漏**、temp=0.3、rep_penalty=1.1、extract_svg 容错）|
| `student_kit/train_config.yaml` | 训练超参数（v2 优化版）|
| `results.json` | 完整评测结果（17 样本，基座 0.6898 vs 微调 0.8360，delta=+0.1461，17/17 有效）|
| `report.md` | 端到端开发流程报告（含 v2 优化历程）|
| `DEVELOPMENT_GUIDE.md` | 新手开发全流程指南（944 行）|
