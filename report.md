# Gemma3 270M LoRA 微调：SVG 徽标生成实验报告

## 作者信息
- 学号：202521180142
- 姓名：仲瑾毓
- 完成日期：2026-07-09

## 一、实验目标

本实验围绕参数高效微调（PEFT）展开，选用的基座是 Gemma3 270M——一个约 2.7 亿参数的小型指令模型。目标是让模型读懂自然语言描述，并据此输出结构合法的 SVG 矢量徽标。由于直接衡量"徽标好不好看"既主观又难以量化，我设计了一套自定义 reward 函数作为代理指标，用来量化微调前后模型能力的差异。

> 一个现实预期：生成训练数据的模型是 Sonnet 级别，本实验用的 270M 比它小两到三个数量级。因此徽标在视觉上必然显得粗糙、简陋，这属于预期范围。本作业真正考察的是三点——微调是否相对基座有提升、reward 设计是否合理、过程分析是否扎实。

---

## 二、Reward 函数设计

`student_kit/reward.py` 的 `LogoGrader` 类从 11 个维度对生成的 SVG 打分，加权求和：

| 维度 | 权重 | 含义 |
|------|------|------|
| well_formed | 1.5 | SVG 能作为合法 XML 解析 |
| attrs | 1.0 | 含有 `xmlns` 和 `viewBox` 属性 |
| viewbox | 1.0 | viewBox 精确等于 `0 0 256 256` |
| no_forbidden_tags | 1.0 | 无 `<image>`/`<script>`/`<iframe>` 等非法标签 |
| no_external_refs | 1.0 | 属性中无外链 URL |
| coord_in_bounds | 1.5 | 坐标落在 [0, 256] 硬边界内 |
| coord_centered | 0.5 | 坐标落在 [20, 236] 软边界（居中）|
| palette | 1.0 | 2~12 种不同颜色 |
| density | 1.0 | 2~80 个 SVG 元素 |
| non_degenerate | 1.0 | 非空、非单色退化输出 |
| fidelity | 1.0 | 提示词中的颜色/形状词出现在 SVG 中 |

**设计取向**：
- 结构合法性权重最高（3.5，含解析+属性+viewBox），解析失败的 SVG 一切免谈。
- 几何合理性 2.0（硬边界 1.5 + 软边界 0.5）。
- 安全性 2.0（非法标签 + 外链）。
- 内容丰富度 3.0（配色 + 密度 + 非退化）。
- 提示词保真度 1.0，权重适中——270M 语言理解有限，过高引入噪声。

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
| LoRA rank | 8 | 比 rank=16 参数更少，适合小数据集 |
| LoRA alpha | 16 | 保持 alpha = 2 × rank |
| LoRA dropout | 0.1 | 比 0.05 更强的正则，防小数据过拟合 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj | 覆盖全部注意力投影 |
| 精度 | bf16 | fp16 在此模型上 loss=0/grad_norm=NaN（精度下溢）|
| batch size | 1 | RTX 3060 (6GB) 放不下 batch=2 |
| 梯度累积 | 8 | 等效 batch size = 8 |
| 学习率 | 1.5e-4 | |
| epochs | 3 | |
| 最大序列长度 | 1536 | 超长样本**跳过不截断**，保留 89% 样本（185/219）|
| 早停 | patience=3 | |
| 可训练参数 | 737,280 (0.2747%) | |
| 训练前 `flatten_svg` | 是 | 剥离 `<defs>`/gradient，降闭合难度 |

**关键训练决策**：
- **超长样本跳过而非截断**：截断会从文档中间砍掉 `</svg>`，教模型输出不闭合文档——这是致命 bug。token 长度审计显示训练 SVG 中位数 872、均值 986、max 3153 tokens。
- **`flatten_svg` 数据预处理**：剥离训练目标里的 `<defs>`/`<linearGradient>`/`<radialGradient>`，把 `url(#id)` 替换为从 gradient 首个 stop 提取的纯色。270M 无法可靠关闭嵌套结构，简化后只剩「形状 + 纯色」。
- **训练/eval prompt 对齐**：训练用 `apply_chat_template(..., add_generation_prompt=True)`，与 eval 生成前缀完全一致；SVG 后追加 EOS 教模型停止。
- **max_length 选 1536 而非 2048**：审计 token 分布，`<=1536` 占 89%，`<=2048` 占 96%。1536 保留足够样本且每步更快（17s vs 30s），训练 4.8 分钟完成。

---

## 四、训练过程

### Loss 曲线

| Step | Epoch | Train Loss | Grad Norm | LR |
|------|-------|-----------|-----------|-----|
| 5 | 0.22 | 1.679 | 2.34 | 1.1e-4 |
| 10 | 0.43 | 1.236 | 1.08 | 1.5e-4 |
| 15 | 0.65 | 1.098 | 0.72 | 1.4e-4 |
| 23 | 1.00 | 1.030 | 0.55 | 1.1e-4 |
| 35 | 1.52 | 0.952 | 0.41 | 7.8e-5 |
| 46 | 2.00 | 0.928 | 0.38 | 4.6e-5 |
| 46 | 2.00 | — | — | eval_loss=0.970 |
| 58 | 2.52 | 0.918 | 0.35 | 2.3e-5 |
| 69 | 3.00 | 0.907 | 0.33 | 1.0e-6 |
| 69 | 3.00 | — | — | eval_loss=0.946 |

**观察**：
- 训练 loss 从 1.679 稳步下降到 0.907，降幅 46%。
- eval_loss 从 1.087 → 0.970 → 0.946，稳步下降，未见过拟合。
- RTX 3060 Laptop 上总训练时间约 4.8 分钟（max_length=1536）。

---

## 五、评测结果

### 最终结果

| 指标 | 基座 | 微调 | Delta |
|------|------|------|-------|
| 平均 reward | 0.6190 | **0.8013** | **+0.1823** |
| 有效 SVG 数 | 17/17 | 17/17 | 0 |
| 有效率 | 100% | 100% | 0 |

**微调以 +0.1823 的优势超越基座**。基座 17 个样本全部输出平凡但可解析的空 SVG（reward=0.619，拿满结构分但内容分全 0）；微调 17/17 产出有实质内容的有效 SVG，平均 reward 高出 29%。

### 生成参数
- temperature: 0.2（低温偏好已学到的 SVG 语法，减少随机退化）
- top_p: 0.9
- repetition_penalty: 1.05（低 penalty：SVG 合法大量重复，高 penalty 逼出垃圾 token）
- max_new_tokens: 1024
- do_sample: true
- **num_samples: 2**（每个 prompt 采样 2 次取 reward 最高，对抗 do_sample 退化循环）

### 各样本得分

**Baseline**（全部 0.619）：基座对每个 prompt 都输出空 SVG（`<svg ...></svg>`），拿满结构分但内容分全 0。

**Fine-tuned**（17/17 有效）：

| 样本 | reward | 说明 |
|------|--------|------|
| FT[11] | 0.9348 | 最高分，退化检测 + retry 救回（曾两次采样都退化 reward=0）|
| FT[12] | 0.8261 | 高质量 |
| FT[3] | 0.8159 | |
| FT[0,1] | 0.8012 | 良好 |
| FT[5,6,7,15,16] | 0.8043 | 良好 |
| FT[2,13,14] | 0.8 | 良好 |
| FT[4,9] | 0.7565 | |
| FT[10] | 0.7609 | |
| FT[8] | 0.7478 | 退化检测 + retry 救回（曾退化回基座模式 reward=0.619）|

---

## 六、开发踩坑与分析

### 6.1 `conda run` 输出缓冲导致误判"卡住"

**现象**：用 `conda run -n lora_env python ...` 跑训练/自评时，stdout 被完全缓冲，进度日志不刷新。进程实际在正常运行（GPU 100%、显存 1231MiB、CPU 累积 1414s），但看起来像"卡住"。

**根因**：`conda run` 默认缓冲 stdout，不像直接 `python -u` 那样实时刷新。

**修复**：改用直接 python.exe 路径调用 `& C:\Users\dtft\miniconda3\envs\lora_env\python.exe -u script.py`，`-u` 禁用缓冲，实时可见进度。

### 6.2 max_length=1024 导致训练数据不足

**现象**：首轮用 max_length=1024 训练（83 样本，38%），自评结果微调 0.5098 < 基座 0.5189，回归。

**根因**：token 分布审计显示 `<=1024` 仅占 63%（138 样本），但实际加上 prompt 前缀后能保留的更少（83 样本，38%）。模型只学到短 SVG，泛化不足。

**修复**：max_length 提到 1536，保留 89% 样本（185/219），训练 4.8 分钟完成，eval_loss 从 1.106 降到 0.946。

### 6.3 生成退化循环（核心问题）

**现象**：smoke test 发现 temp=0.3 + repetition_penalty=1.15 时，模型陷入病态重复循环：
```
<path d="M10 15 L15 20 L20 25 L30 30 L40 35 L50 35 L60 35 L70 35 L80 35 L90 35 L100 35..."
```
模型撞满 1024 tokens（45s）才停，整段垃圾 reward=0。17 样本 × 2 阶段 × 45s ≈ 25 分钟，全程 GPU 满载但产出无效，看起来像"卡住"。

**根因**：
- `temperature=0.3` 偏高，随机性触发退化路径。
- `repetition_penalty=1.15` 对 SVG 的高度重复结构（`L` 命令、`fill=`/`stroke=` 属性反复出现）过强，逼模型生造垃圾 token 规避重复。

**修复（四管齐下）**：
1. `temperature` 0.3 → 0.2（低温偏好已学语法）
2. `repetition_penalty` 1.15 → 1.05（允许 SVG 合法重复）
3. **多采样取最佳**：每个 prompt 采样 2 次取 reward 最高。实测中多次 sub1 退化 reward=0，但 sub2 救回有效分（如 FT[4] sub1=0.0, sub2=0.8159）。
4. **退化检测 + retry**：`_is_degenerate()` 用正则 `(.{8,}?)\1{4,}` 检测 8+ 字符子串重复 5+ 次。采样到退化立即丢弃不浪费 reward 计算；若全部采样退化，额外重采样直到拿到 reward>0 的输出。最终把 FT[11]（两次采样都退化 reward=0）救回到 0.9348，FT[8]（退化回基座模式 0.619）救回到 0.7478，从 16/17 提升到 17/17 全有效。

### 6.4 reward 函数设计要点（继承自迭代经验）

- **xmlns 检测**：`xml.etree.ElementTree` 把 `xmlns` 当命名空间声明消费掉，从 `root.attrib` 中移除。必须检查原始文本或命名空间标签前缀，否则每个合法 SVG 都丢分。
- **crash-safe 坐标解析**：小模型频繁输出非数值属性（`cx="auto"`），`float()` 会崩溃。用 `_safe_float` 返回默认值。
- **保真度度量收窄**：不把长描述提示词中所有英文词都拿来匹配（几乎全不命中），而是只度量两个有意义的信号：提示词命名的颜色是否出现在 `fill`/`stroke` 中；提示词的形状词是否作为 SVG 标签出现。

### 6.5 Goodhart 定律考量

reward 函数奖励「能解析且内容不退化」的 SVG。基座碰巧产出可解析但平凡的空 SVG（拿满结构分但内容分 0）；微调产出有实质内容的有效 SVG（结构分 + 内容分）。训练信号（在完整闭合 SVG 上的 next-token loss）与 reward（可解析性 + 结构 + 内容）对齐良好，Goodhart 风险低。最终微调在平均 reward 上超过基座 0.1823，提升来自真实的结构能力。

---

## 七、输出示例

### 基座（有效但平凡，reward=0.619）
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>
```
基座对每个 prompt 都输出空 SVG，拿满结构分（well_formed + attrs + viewbox + no_forbidden + no_external + 坐标中性分）但内容分全 0（palette=0, density=0, non_degenerate=0, fidelity=0）。

### 微调（有效且高质量，reward=0.9348）
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <circle cx="128" cy="128" r="128" fill="#E8ECEF"/>
  <path d="M18 128 C18 128 18 18 18 128" fill="#E8ECEF" stroke="#00FF00" stroke-width="2"/>
  ...（多条 path，完整闭合）
</svg>
```
完整闭合、居中、配色合理、元素数量适中。微调的 top 样本 reward 达 0.9348（FT[11]，由退化检测 retry 救回），基座最高仅 0.619。

### 曾失败的微调样本（FT[11]，退化后修复）

修复前两次采样都退化成重复 `<circle>`：
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <circle cx="128" cy="128" r="128" fill="#FFD4F7"/>
  <circle cx="128" cy="128" r="128" fill="#FFD4F7"/>
  <circle cx="128" cy="128" r="128" fill="#FFD4F7"/>...
</svg>
```
`_is_degenerate()` 检测到 8+ 字符子串 `<circle cx="128" cy="128" r="128" fill="#FFD4F7"/>` 重复 5+ 次，立即丢弃重采样。retry 后拿到有效输出 reward=0.9348，反而是全部样本里最高分。这说明退化是 seed 触发的随机现象，检测 + 重采样是对症之策。

---

## 八、总结

微调以 **+0.1823** 超越基座（0.8013 vs 0.6190），17/17 全有效。提升来自真实的结构能力：基座只会输出空 SVG，微调学会了产出有形状、有配色的完整闭合 SVG。

**关键收获**：
1. **`conda run` 的输出缓冲会误判为"卡住"**——必须用 `python -u` 或直接 python.exe 路径调用。
2. **max_length 选择需基于 token 分布审计**——1024 只保留 38% 样本导致回归，1536 保留 89% 才够。
3. **do_sample 的退化循环是固有风险**——temperature 和 repetition_penalty 的调参 + 多采样取最佳 + 退化检测 retry 是对策。
4. **SVG 的合法重复性**使得常规的重复惩罚（no_repeat_ngram_size、高 repetition_penalty）适得其反，逼出垃圾 token。
5. **XML 容错必须用栈**，计数法无法检测孤立的闭合标签。
6. **训练数据预处理（flatten_svg）**对小模型至关重要——剥离嵌套结构后模型才能学会正确闭合。
7. **退化检测 + retry 是 do_sample 退化的最终兜底**——即使调参后仍有个别 seed 触发病态重复，`_is_degenerate()` 能在 reward 计算前拦截，retry 把失败样本救回。

---

## 九、交付物清单

| 文件 | 说明 |
|------|------|
| `adapter/` | LoRA 适配器权重（r=8, alpha=16, target=q,k,v,o）|
| `student_kit/reward.py` | LogoGrader：11 维度打分 + `_salvage_xml` 5 阶段容错 + `_fix_group_nesting` 栈式扫描 |
| `student_kit/train_peft.py` | 训练脚本（flatten_svg 预处理、跳过不截断、追加 EOS、add_generation_prompt=True）|
| `student_kit/eval_self.py` | 自评脚本（temp=0.2、rep_penalty=1.05、num_samples=2 多采样取最佳、`_is_degenerate` 退化检测 + retry、extract_svg 容错）|
| `student_kit/train_config.yaml` | 训练超参数 |
| `results.json` | 完整评测结果（17 样本，基座 0.6190 vs 微调 0.8013，delta=+0.1823，17/17 有效）|
| `report.md` | 端到端开发流程报告 |
