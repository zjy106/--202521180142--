# LoRA 微调 Gemma3 270M 生成 SVG 徽标

## 学生信息

| 项目 | 内容 |
|------|------|
| 学号 | 202521180142 |
| 姓名 | 仲瑾毓 |
| 完成日期 | 2026-07-09 |

---

## 任务简介

通过 LoRA 微调 Gemma3 270M（约 2.7 亿参数的小型指令模型），使其在给定详细文字提示词后能生成有效的 SVG 徽标。以自定义 reward 函数作为代理指标，衡量微调相对基座模型的提升。

---

## 快速复现

### 环境要求
- Windows 11 + Miniconda
- 一块 6GB+ 显存的 GPU（如 RTX 3060 Laptop）
- 约 1.5GB 磁盘空间（基座模型 511MB + 数据 + 适配器）

### 复现步骤

```powershell
# 1. 克隆本仓库
git clone <repo-url>
cd <repo-name>

# 2. 创建 conda 环境
conda create -n lora_env python=3.10 -y
conda activate lora_env
pip install transformers peft accelerate torch pyyaml modelscope

# 3. 下载基座模型（从 ModelScope，不要用 HuggingFace，被墙）
python scripts/download_model.py
# → 模型存到 models/google/gemma-3-270m-it/

# 4. 克隆数据集
git clone https://github.com/roboticcam/logo-detailed-prompt
# → 数据存到 logo-detailed-prompt/{train,valid}.jsonl

# 5. 训练适配器（约 30 分钟 / RTX 3060）
conda run -n lora_env python student_kit/train_peft.py `
  --train_data ./logo-detailed-prompt/train.jsonl `
  --valid_data ./logo-detailed-prompt/valid.jsonl `
  --model_name_or_path ./models/google/gemma-3-270m-it `
  --output_dir ./adapter

# 6. 运行自评（生成 results.json）
conda run -n lora_env python student_kit/eval_self.py `
  --valid_data ./logo-detailed-prompt/valid.jsonl `
  --model_name_or_path ./models/google/gemma-3-270m-it `
  --adapter_path ./adapter `
  --output_file ./results.json `
  --max_new_tokens 1024 `
  --temperature 0.3

# 7. 查看结果
# results.json 的 summary 字段给出「基座 vs 微调」对比
```

超参数见 `student_kit/train_config.yaml`。

---

## 目录结构

```
.
├── adapter/                       # LoRA 适配器权重（提交产物）
│   ├── adapter_config.json        # LoRA 配置 (r=8, alpha=16, target=q,k,v,o)
│   ├── adapter_model.safetensors  # LoRA 权重
│   └── tokenizer 配置
├── student_kit/
│   ├── reward.py                  # 【核心】奖励函数 (LogoGrader, 11 维度打分 + XML salvage)
│   ├── train_peft.py              # 训练脚本 (含 flatten_svg 数据预处理)
│   ├── eval_self.py               # 自评脚本 (含 extract_svg 容错)
│   └── train_config.yaml          # 超参数
├── scripts/
│   └── download_model.py          # 下载基座模型
├── results.json                   # 自评结果：基座 vs 微调
├── report.md                      # 分析报告（主要评分产物）
├── task.md                        # 老师任务手册
└── .gitignore                     # 排除 models/、数据集、checkpoint、日志
```

---

## 关键产物说明

| 产物 | 文件 | 说明 |
|------|------|------|
| LoRA 适配器 | `adapter/` | 老师加载它即可复现微调效果 |
| 奖励函数 | `student_kit/reward.py` | LogoGrader 类，11 维度打分 + `_salvage_xml` 容错管道 |
| 超参数 | `student_kit/train_config.yaml` | 可复现性 |
| 自评结果 | `results.json` | 基座 vs 微调对比 |
| 分析报告 | `report.md` | 根因分析 + 迭代过程 |

---

## 技术亮点

1. **reward 容错设计**：内置 `_salvage_xml` 5 阶段 XML 修复（剥离 defs → 自闭合 void 元素 → 栈式修复 `<g>` → 去逗号 → 去重属性），诚实度量模型输出能力，而非因单个语法错误全盘归零。
2. **训练数据拍平**：`flatten_svg` 剥离 `<defs>`/gradient，把嵌套结构降为「形状 + 纯色」，让 270M 能学会正确闭合。
3. **栈式嵌套修复**：`_fix_group_nesting` 按文档顺序扫描 `<g>`，丢弃孤立闭合标签、补齐未闭合开标签——计数法无法识别顺序错误的孤立 `</g>`。
4. **完整 pipeline 对齐**：训练/eval prompt 用 `add_generation_prompt=True` 保持一致；超长样本跳过而非截断；SVG 后追加 EOS 教模型停止。
5. **反模板化评分（v2 优化）**：新增 `anti_template` 维度（权重 1.0），检测「单色居中重复圆」安全模板——当所有 circle 的 cx/cy/r 完全相同或所有形状元素签名去重后只剩 1 种时，该维度归零。对抗 Goodhart 效应，防止模型刷分而非学习真实生成能力。

---

## v2 优化说明（2026-07-15）

基于对比分析（见 `../分析报告.html`），本版本针对以下问题进行了优化：

### reward.py 优化
- **颜色阈值上调**：`PALETTE_MIN` 2→3，单色输出不再能拿满分 palette 维度
- **元素阈值上调**：`SHAPE_MIN` 2→3，2 个重复圆不再能拿满分 density 维度
- **非退化权重恢复**：1.0→1.5，加强对退化输出的惩罚
- **新增反模板化维度**：`_is_template_like()` 检测完全相同元素重复（权重 1.0），对抗 Goodhart 效应

### eval_self.py 优化
- **移除多采样取最佳**：num_samples=2→1，单次采样诚实反映模型能力，消除统计偏向
- **移除退化 retry 机制**：退化输出由 reward 函数自然惩罚，不在评测侧人为丢弃
- **移除 reward 阈值泄漏**：system prompt 不再包含精确的坐标区间、颜色数、标签白名单等评分阈值
- **温度上调**：temperature 0.2→0.3，适度随机性避免模型过度偏好安全模板
- **重复惩罚上调**：repetition_penalty 1.05→1.1，抑制退化循环

### train_config.yaml 优化
- **rank 上调**：8→16，增加 LoRA 容量学习复杂形状组合
- **dropout 上调**：0.05→0.1，增加正则化，抑制过拟合到「安全模板」捷径
- **learning rate 下调**：2e-4→1e-4，降低学习率减少过拟合到简单模式
- **epochs 下调**：4→2，减少训练轮数，避免模型学到「安全模板」捷径
- **max_length 上调**：1536→2048，保留 96% 样本（vs 89%）
- **target_modules 扩展**：新增 gate_proj/up_proj/down_proj，加入 MLP 层增加模型容量（可训练参数 737K→3.80M）
- **weight_decay 上调**：0.01→0.05，增加权重衰减正则化
- **warmup_ratio 上调**：0.1→0.15，更长 warmup 稳定初期训练

### train_peft.py 优化
- **flatten_svg 坐标 clamp**：将训练数据中 x/y/cx/cy 等坐标 clamp 到 [0, 256]，width/height clamp 到 [1, 256]，消除模型从训练数据学到 x=-9999 等极端坐标并滥用的风险

### 评测结果（优化前后对比）

| 指标 | v1 (旧 adapter + 旧 eval) | v2 (旧 adapter + 新 eval) | v3 (新 adapter + 新 eval) |
|------|---------------------------|---------------------------|---------------------------|
| Baseline | 0.6190 | 0.7527 | 0.6898 |
| Fine-tuned | 0.8013 | 0.6340 | **0.8360** |
| Delta | +0.1823 (Goodhart 假象) | -0.1186 (真实差距) | **+0.1461 (真实提升)** |

---

## 联系方式

- 学号：202521180142
- 姓名：仲瑾毓
- 邮箱：709996150@qq.com
