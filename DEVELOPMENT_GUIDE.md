# LoRA 微调 SVG 徽标生成 - 新手开发全流程指南

> 本指南面向首次接触 LoRA 微调的开发者，从原理到代码逐层展开，覆盖从项目搭建到最终评测的完整流程。所有代码引用均对应本仓库实际文件。

---

## 目录

- [一、项目概述与目标](#一项目概述与目标)
- [二、核心原理](#二核心原理)
  - [2.1 LoRA 低秩分解原理](#21-lora-低秩分解原理)
  - [2.2 PEFT 框架](#22-peft-框架)
  - [2.3 指令微调与 chat template](#23-指令微调与-chat-template)
  - [2.4 SVG 生成任务的特点](#24-svg-生成任务的特点)
- [三、环境准备](#三环境准备)
- [四、Reward 函数设计](#四reward-函数设计)
  - [4.1 设计哲学](#41-设计哲学)
  - [4.2 11 维度评分体系](#42-11-维度评分体系)
  - [4.3 XML 容错管道](#43-xml-容错管道)
  - [4.4 栈式嵌套修复（关键教训）](#44-栈式嵌套修复关键教训)
- [五、训练数据预处理](#五训练数据预处理)
  - [5.1 flatten_svg 原理](#51-flatten_svg-原理)
  - [5.2 token 分布审计](#52-token-分布审计)
  - [5.3 max_length 选择](#53-max_length-选择)
- [六、训练脚本设计](#六训练脚本设计)
  - [6.1 LoRA 配置](#61-lora-配置)
  - [6.2 训练参数与超参数](#62-训练参数与超参数)
  - [6.3 损失计算（mask prompt）](#63-损失计算mask-prompt)
  - [6.4 早停与梯度检查点](#64-早停与梯度检查点)
- [七、自评脚本设计](#七自评脚本设计)
  - [7.1 解码参数选择](#71-解码参数选择)
  - [7.2 extract_svg 容错](#72-extract_svg-容错)
  - [7.3 退化检测 + retry 机制](#73-退化检测--retry-机制)
- [八、训练与评测流程](#八训练与评测流程)
  - [8.1 启动训练](#81-启动训练)
  - [8.2 监控训练](#82-监控训练)
  - [8.3 运行自评](#83-运行自评)
- [九、踩坑与解决方案](#九踩坑与解决方案)
- [十、交付物清单](#十交付物清单)

---

## 一、项目概述与目标

### 1.1 任务背景

给定一个由强模型（Sonnet）生成的（详细提示词 → SVG 徽标）配对数据集，目标是：

1. **设计一个 reward 函数**：用程序化方式定义"什么样的 SVG 徽标是合格的"——有效性、闭合性、配色、元素数量、坐标边界、提示词覆盖度等。
2. **用 LoRA 微调 Gemma3 270M**：让小模型能根据自然语言描述生成有效的 SVG 徽标。
3. **衡量提升**：微调后是否比基座 270M 有提升？

### 1.2 关键预期

- **270M 是极小的模型**（比生成训练数据的 Sonnet 小几百倍），徽标必然粗糙简陋。
- **评分依据**：相对基座的提升、reward 设计质量、分析质量——**不是徽标好不好看**。
- 一个能稳定产出"有效但朴素"徽标、并被充分分析解释的模型，就是强结果。

### 1.3 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 基座模型 | Gemma3 270M (gemma-3-270m-it) | 任务指定，小到笔记本可训 |
| 微调方法 | LoRA (PEFT) | 参数高效，几 MB 权重 |
| 框架 | transformers + PEFT | 灵活可控 |
| 精度 | bf16 | fp16 会 loss=0/NaN |
| 数据 | logo-detailed-prompt | GitHub 开源 |

---

## 二、核心原理

### 2.1 LoRA 低秩分解原理

**问题**：全量微调一个 270M 模型需要更新 270M 参数，显存和存储都吃紧。

**LoRA 的核心思想**：冻结原始权重 W，只训练一个低秩增量 ΔW = B·A。

#### 数学推导

原始线性层的前向传播：
```
h = W·x        # W ∈ R^(d×k), x ∈ R^k, h ∈ R^d
```

全量微调会更新 W → W + ΔW，其中 ΔW ∈ R^(d×k) 有 d×k 个参数。

**LoRA 假设**：微调过程中的权重更新 ΔW 具有低内在秩（low intrinsic rank）。于是把 ΔW 分解为两个小矩阵的乘积：

```
ΔW = B·A       # A ∈ R^(r×k), B ∈ R^(d×r), r << min(d,k)
```

前向传播变为：
```
h = W·x + B·A·x
```

#### 参数量对比

以 Gemma3 270M 的注意力投影层为例（d = k = 2560）：

| 方法 | 可训练参数 | 占比 |
|------|-----------|------|
| 全量微调 | 2560×2560 = 6,553,600 / 层 | 100% |
| LoRA (r=8) | 8×2560 + 2560×8 = 40,960 / 层 | 0.625% |

四个投影层（q/k/v/o）总计：737,280 参数，**仅占模型 0.2747%**。

#### 为什么有效

1. **存储**：adapter 权重只有 2.9 MB（vs 全量微调的 1 GB+）
2. **显存**：梯度只对 A、B 计算，激活值用梯度检查点压缩
3. **效果**：低秩假设在很多下游任务上经验有效——微调不需要改变模型的全部能力，只需要在子空间里调整

#### 关键超参数

- **rank (r)**：低秩矩阵的秩。r 越大表达能力越强但参数越多。r=8 是小数据集的甜点。
- **alpha**：缩放因子，实际增量 = (alpha/r)·B·A。**经验法则：alpha = 2×rank**，让 (alpha/r) = 2，使 LoRA 增量与原始权重的尺度匹配。
- **target_modules**：对哪些层应用 LoRA。注意力投影（q/k/v/o）是标准选择。

### 2.2 PEFT 框架

PEFT（Parameter-Efficient Fine-Tuning）是 HuggingFace 的库，封装了 LoRA 等方法。

**核心 API**：

```python
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 1. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # 秩
    lora_alpha=16,                # 缩放因子
    lora_dropout=0.1,             # dropout 防过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",                  # 不训练 bias
    inference_mode=False,         # 训练模式
)

# 2. 应用 LoRA 到基座模型
model = AutoModelForCausalLM.from_pretrained(...)
model = get_peft_model(model, lora_config)

# 3. 训练...
# 4. 保存 adapter（只保存 LoRA 权重，几 MB）
model.save_pretrained("./adapter")

# 5. 推理时加载：基座 + adapter
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, "./adapter")
model = model.merge_and_unload()   # 合并权重用于推理
```

**为什么用 `merge_and_unload`**：合并后模型与全量微调效果一致，但只增加推理时的矩阵乘法，无额外开销。

### 2.3 指令微调与 chat template

#### Chat 格式

Gemma3 是指令模型，训练数据是 chat 格式：
```json
{
  "messages": [
    {"role": "system", "content": "You are a logo designer..."},
    {"role": "user", "content": "Draw a sun badge..."},
    {"role": "assistant", "content": "<svg>...</svg>"}
  ]
}
```

#### Chat Template 的作用

tokenizer 有一个 Jinja2 模板，把 messages 渲染成模型期望的文本格式：

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True   # 加上 assistant 轮的开头标记
)
```

渲染后大致是：
```
<start_of_turn>user
Draw a sun badge...<end_of_turn>
<start_of_turn>assistant
```

#### 关键陷阱：训练/eval 对齐

**训练时**用 `add_generation_prompt=True`，结尾带 `<start_of_turn>assistant`。
**eval 生成时**也必须用 `add_generation_prompt=True`。

如果不一致，模型在训练时学到的前缀和推理时看到的前缀不同，分布不匹配，效果会大幅退化。

### 2.4 SVG 生成任务的特点

#### 与普通文本生成的区别

| 特性 | 普通文本 | SVG 徽标 |
|------|---------|---------|
| 结构 | 自然语言，容错 | 严格 XML，必须闭合 |
| 长度 | 几十到几百 tokens | 500-3000 tokens |
| 重复性 | 词汇多样 | 大量 `<circle>`、`fill=` 重复 |
| 评判 | 语义相似度 | 程序化检查（合法性、配色、坐标）|

#### 对小模型的挑战

1. **闭合标签**：270M 无法可靠地关闭嵌套结构（`<defs>`、`<g>`）
2. **退化循环**：do_sample 时会陷入"重复同一个坐标"的死循环
3. **长度**：完整 SVG 动辄上千 tokens，容易撞 token 预算

这些特点决定了 reward 函数和预处理的设计。

---

## 三、环境准备

### 3.1 硬件需求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | 任意 CUDA GPU（4GB+）| RTX 3060 6GB+ |
| 内存 | 8GB | 16GB |
| 训练时间 | 5-15 分钟 | 5 分钟（max_length=1536）|

### 3.2 软件环境

```bash
# 创建 conda 环境
conda create -n lora_env python=3.10
conda activate lora_env

# 安装依赖
pip install torch transformers peft accelerate
pip install pyyaml  # 读 train_config.yaml
```

### 3.3 模型与数据下载

```bash
# 数据集（直接 git clone，无需 LFS）
git clone https://github.com/roboticcam/logo-detailed-prompt

# 基座模型（从 ModelScope，不要用 HuggingFace——被墙/慢）
pip install modelscope
modelscope download --model <gemma-3-270m repo> --local_dir ./models/google/gemma-3-270m-it
```

### 3.4 项目目录结构

```
lora_2/
├── adapter/                    # LoRA 适配器权重（训练产物）
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer_config.json
├── models/google/gemma-3-270m-it/  # 基座模型
├── logo-detailed-prompt/       # 数据集
│   ├── train.jsonl
│   └── valid.jsonl
├── student_kit/
│   ├── reward.py               # reward 函数
│   ├── train_peft.py           # 训练脚本
│   ├── eval_self.py            # 自评脚本
│   └── train_config.yaml       # 超参数
├── results.json                # 评测结果
├── report.md                   # 实验报告
└── README.md
```

---

## 四、Reward 函数设计

reward 函数是整个项目的**头等大事**——它既是训练的代理目标，也是评分项。代码见 [student_kit/reward.py](student_kit/reward.py)。

### 4.1 设计哲学

**核心策略：先抢救再评分**

270M 小模型无法可靠闭合 XML 标签。如果一次解析失败就归零，会系统性低估模型的真实形状/配色能力。

因此 reward 函数采用「多阶段容错」：
1. 先尝试严格解析
2. 失败则进入 `_salvage_xml` 修复管道
3. 还失败则兜底截断

只有在容错后仍无法解析时才归零。

### 4.2 11 维度评分体系

```python
# 权重分配思路
# 结构合法性（well_formed + attrs + viewbox）合计 3.5，最高
# 几何合理性（coord_hard + coord_soft）合计 2.0
# 安全性（forbidden_tags + external_refs）合计 2.0
# 内容丰富度（palette + density + non_degenerate）合计 3.0
# 提示词保真度（fidelity）1.0，权重适中
```

| 维度 | 权重 | 含义 | 评判方式 |
|------|------|------|---------|
| well_formed | 1.5 | SVG 能作为合法 XML 解析 | `ET.fromstring` 成功 |
| attrs | 1.0 | 含 `xmlns` 和 `viewBox` 属性 | 检查原始文本/命名空间 |
| viewbox | 1.0 | viewBox 精确等于 `0 0 256 256` | 字符串比较 |
| no_forbidden_tags | 1.0 | 无 `<image>`/`<script>` 等非法标签 | 遍历元素 |
| no_external_refs | 1.0 | 属性中无外链 URL | 检查属性值前缀 |
| coord_in_bounds | 1.5 | 坐标落在 [0, 256] 硬边界 | 收集所有坐标 |
| coord_centered | 0.5 | 坐标落在 [20, 236] 软边界 | 同上 |
| palette | 1.0 | 2~12 种不同颜色 | 收集 fill/stroke |
| density | 1.0 | 2~80 个 SVG 元素 | 计数形状标签 |
| non_degenerate | 1.0 | 非空、非单色退化 | 检查 shapes/colors |
| fidelity | 1.0 | 提示词中的颜色/形状词出现在 SVG 中 | 词汇匹配 |

**权重分配的理由**：
- **结构合法性权重最高**：解析失败的 SVG 一切免谈
- **保真度权重适中**：270M 语言理解有限，过高引入噪声
- **安全性 2.0**：防止生成危险标签（脚本注入）

### 4.3 XML 容错管道

`_salvage_xml` 是 5 阶段修复管道：

```python
@staticmethod
def _salvage_xml(text: str) -> str:
    out = text
    
    # 阶段 1：移除 <defs> 块（非必要，且常未闭合）
    out = re.sub(r'<defs\b[^>]*>.*?</defs\s*>', '', out, flags=re.DOTALL)
    out = re.sub(r'<defs\b[^>]*>(?:(?!</svg>).)*$', '', out, flags=re.DOTALL)
    
    # 阶段 2：未自闭合的 void 元素补上 />
    for tag in ["rect", "circle", "ellipse", "line", "path", "polygon", "polyline", "stop", "use", "image"]:
        out = re.sub(rf'<{tag}\b([^>]*?)(?<!/)>', rf'<{tag}\1/>', out)
    
    # 阶段 3：<g> 嵌套栈式修复
    out = LogoGrader._fix_group_nesting(out)
    
    # 阶段 4：属性间逗号 → 空格
    out = re.sub(r'"\s*,\s*', '" ', out)
    
    # 阶段 5：同标签内同名属性去重
    def _dedupe_attrs(m):
        head = m.group(1)
        attrs = m.group(2)
        seen = set()
        kept = []
        for am in re.finditer(r'(\w[\w-]*)=["\'][^"\']*["\']', attrs):
            name = am.group(1)
            if name not in seen:
                seen.add(name)
                kept.append(am.group(0))
        return f'<{head} {" ".join(kept)}/>'
    
    out = re.sub(r'<(\w+)\s+([^>]*?)/>', _dedupe_attrs, out)
    return out
```

**每阶段解决的问题**：
1. **剥离 `<defs>`**：`<defs>` 内是渐变定义，非必要且小模型常不闭合
2. **自闭合 void 元素**：小模型常写 `<rect ...>` 而非 `<rect .../>`
3. **修复 `<g>` 嵌套**：见下节
4. **逗号转空格**：小模型偶尔输出 `x="1", y="2"`
5. **属性去重**：防止 `<circle cx="1" cx="2"/>` 破坏解析

### 4.4 栈式嵌套修复（关键教训）

这是 reward 函数里**最隐蔽的 bug**。

#### 错误的计数法

```python
# 错误：计数法
open_count = text.count("<g>")
close_count = text.count("</g>")
if open_count > close_count:
    text += "</g>" * (open_count - close_count)
```

**为什么错**：计数法无法检测**孤立的 `</g>`**。

考虑这个例子：
```xml
<g>      <!-- 1 个开 -->
</g>     <!-- 孤立的闭 -->
</g>     <!-- 孤立的闭 -->
```
计数法：开=1，闭=2，差=-1，不做任何处理。但实际有两个孤立 `</g>` 破坏解析。

#### 正确的栈式扫描

```python
@staticmethod
def _fix_group_nesting(text: str) -> str:
    out = []
    depth = 0
    i = 0
    n = len(text)
    open_re = re.compile(r'<g\b[^>]*?(?<!/)>')
    close_re = re.compile(r'</g\s*>')
    
    while i < n:
        mo = open_re.match(text, i)
        if mo:
            out.append(text[i:mo.end()])
            depth += 1
            i = mo.end()
            continue
        mc = close_re.match(text, i)
        if mc:
            if depth > 0:
                out.append(text[i:mc.end()])
                depth -= 1
            # 栈空时遇到的 </g> 是孤立的，丢弃
            i = mc.end()
            continue
        out.append(text[i])
        i += 1
    
    if depth == 0:
        return ''.join(out)
    # 仍有未闭合 <g>：在 </svg> 前补齐
    closing = '</g>' * depth
    merged = ''.join(out)
    svg_close = merged.rfind('</svg>')
    if svg_close != -1:
        merged = merged[:svg_close] + closing + merged[svg_close:]
    else:
        merged = merged + closing
    return merged
```

**栈式扫描的逻辑**：
- 遇到 `<g>`：压栈，depth++
- 遇到 `</g>` 且栈非空：弹栈，depth--
- 遇到 `</g>` 但栈空：**丢弃**（这是孤立的闭标签）
- 扫描完仍有未闭合 `<g>`：在 `</svg>` 前补齐

这样能正确处理任何顺序的嵌套错误。

---

## 五、训练数据预处理

### 5.1 flatten_svg 原理

**问题**：训练数据由 Sonnet 生成，大量使用 `<defs>` 内的 `<linearGradient>`/`<radialGradient>` 与 `<stop>` 子元素。270M 的 LoRA 模型无法可靠地关闭这种深层嵌套——它会自闭合 `<linearGradient .../>` 再追加子元素，或干脆忘了 `</defs>`，导致整个文档不可解析。

**解决方案**：彻底移除 `<defs>`，把 `fill="url(#id)"` 替换成从对应渐变首个 `<stop>` 取到的纯色。

```python
def flatten_svg(svg_text: str) -> str:
    # 1. 从每个渐变里 harvest 首个 stop-color，按 id 索引
    grad_map = {}
    for m in re.finditer(
        r'<(?:linear|radial)Gradient\b[^>]*\bid=["\']([^"\']+)["\'][^>]*>(.*?)</(?:linear|radial)Gradient\s*>',
        svg_text, flags=re.DOTALL,
    ):
        gid, body = m.group(1), m.group(2)
        stop_m = re.search(r'stop-color=["\']([^"\']+)["\']', body)
        if stop_m:
            grad_map[gid] = stop_m.group(1)
    
    # 2. 删除所有 <defs>...</defs> 块
    out = re.sub(r'<defs\b[^>]*>.*?</defs\s*>', '', out, flags=re.DOTALL)
    out = re.sub(r'<defs\b[^>]*>(?:(?!</svg>).)*$', '', out, flags=re.DOTALL)
    
    # 3. 把 url(#id) 引用替换成 harvest 到的纯色
    def _swap_url(m):
        gid = m.group(1)
        return grad_map.get(gid, _FALLBACK_PALETTE[abs(hash(gid)) % len(_FALLBACK_PALETTE)])
    
    out = re.sub(r'url\(#([^)]+)\)', _swap_url, out)
    return out
```

**效果**：
- 训练目标从"深层嵌套结构"变成"形状 + 纯色"
- 模型只需要学会闭合 `<circle>`、`<path>` 等简单标签
- 大幅提升可解析率

### 5.2 token 分布审计

**为什么必须审计**：max_length 选太小会丢大量样本，选太大会拖慢训练。

```python
# 审计脚本（伪代码）
lengths = []
for item in train_data:
    tokens = tokenizer.encode(item["messages"]...)
    lengths.append(len(tokens))

print(f"中位数: {np.median(lengths)}")
print(f"均值: {np.mean(lengths)}")
print(f"max: {max(lengths)}")
print(f"<=1024: {sum(l <= 1024 for l in lengths) / len(lengths):.1%}")
print(f"<=1536: {sum(l <= 1536 for l in lengths) / len(lengths):.1%}")
print(f"<=2048: {sum(l <= 2048 for l in lengths) / len(lengths):.1%}")
```

**实际审计结果**：
```
中位数: 872
均值: 986
max: 3153
<=1024: 63%   (138 样本)
<=1536: 89%   (185 样本)
<=2048: 96%
```

### 5.3 max_length 选择

| max_length | 保留样本 | 占比 | 训练时间 | eval_loss | 结果 |
|-----------|---------|------|---------|-----------|------|
| 1024 | 83 | 38% | 2 分钟 | 1.106 | 微调 0.5098 < 基座 0.5189，回归 |
| **1536** | **185** | **89%** | **4.8 分钟** | **0.946** | **微调 0.8013 > 基座 0.6190** |
| 2048 | 210 | 96% | 8 分钟 | — | 边际收益小 |

**选择 1536**：保留 89% 样本，训练 4.8 分钟，eval_loss 0.946。1024 只保留 38% 导致回归——这是必须审计的根本原因。

---

## 六、训练脚本设计

训练脚本见 [student_kit/train_peft.py](student_kit/train_peft.py)。

### 6.1 LoRA 配置

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # 秩：小数据集的甜点
    lora_alpha=16,                # alpha = 2 × rank
    lora_dropout=0.1,             # 比 0.05 更强的正则，防小数据过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 覆盖全部注意力投影
    bias="none",                  # 不训练 bias
    inference_mode=False,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 737,280 || all params: 268,065,664 || trainable%: 0.2747%
```

**超参数选择理由**：
- **r=8**：比 r=16 参数更少，适合小数据集（185 样本），防过拟合
- **alpha=16**：保持 alpha = 2 × rank，让缩放因子 (alpha/r) = 2
- **dropout=0.1**：小数据集需要更强正则
- **target=q,k,v,o**：覆盖全部注意力投影，标准选择

### 6.2 训练参数与超参数

```python
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=1,           # RTX 3060 6GB 放不下 batch=2
    gradient_accumulation_steps=8,           # 等效 batch size = 8
    learning_rate=1.5e-4,
    num_train_epochs=3,
    max_length=1536,                          # 基于 token 审计
    logging_steps=5,
    eval_steps=25,
    save_steps=25,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,                               # 不能用 fp16
    bf16=True,                                # 必须用 bf16
    gradient_checkpointing=True,              # 省显存
    weight_decay=0.01,
    warmup_ratio=0.1,
)
```

**关键决策**：
- **batch_size=1 + gradient_accumulation=8**：6GB 显存放不下 batch=2，用梯度累积模拟 batch=8
- **bf16 而非 fp16**：fp16 在 Gemma3 上会 loss=0/grad_norm=NaN（精度下溢）
- **gradient_checkpointing**：用计算换显存，让 6GB 卡能训 270M

### 6.3 损失计算（mask prompt）

**核心设计**：损失只在 SVG 部分计算，不对 prompt 计算 loss。

```python
class LogoDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=2048):
        # ...
        for line in f:
            # ...
            prompt_ids = tokenizer(prompt_text)["input_ids"]
            target_ids = tokenizer(svg_target, add_special_tokens=False)["input_ids"]
            target_ids = target_ids + [eos_id]   # 教模型停止
            
            input_ids = prompt_ids + target_ids
            if len(input_ids) > max_length:
                skipped += 1
                continue   # 跳过而非截断！
            
            self.records.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                # 关键：prompt 部分的 labels 设为 -100（不计入 loss）
                "labels": [-100] * len(prompt_ids) + target_ids,
            })
```

**为什么 mask prompt**：
- 训练目标是"给定 prompt，生成 SVG"
- 如果对 prompt 也计算 loss，模型会花大量梯度学习"如何生成提示词"——这不是我们要的
- `labels = -100` 是 PyTorch 的约定，CrossEntropyLoss 会跳过这些位置

**为什么跳过而非截断**：
- 截断会从 SVG 中间砍掉 `</svg>`，教模型输出不闭合文档
- 这是致命 bug，会让微调回归

### 6.4 早停与梯度检查点

```python
# 早停：关注 eval_loss，小数据很快过拟合
callbacks=[EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001
)]

# 梯度检查点：省显存
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
```

**早停**：patience=3 表示 eval_loss 连续 3 次不下降就停。防止过拟合。
**梯度检查点**：前向时只保存部分激活值，反向时重新计算。用约 30% 额外计算换 50%+ 显存节省。

---

## 七、自评脚本设计

自评脚本见 [student_kit/eval_self.py](student_kit/eval_self.py)。

### 7.1 解码参数选择

```python
def generate_svg(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.2):
    # ...
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.2,           # 低温
        top_p=0.9,
        repetition_penalty=1.05,   # 低 penalty
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
```

**参数选择理由（来自实测 smoke test）**：

| 参数 | 值 | 理由 |
|------|-----|------|
| temperature | 0.2 | temp=0.3 触发"L 18 L 18..."重复死循环 |
| top_p | 0.9 | 标准选择 |
| repetition_penalty | 1.05 | SVG 合法大量重复，penalty=1.15 会逼出垃圾 token |
| max_new_tokens | 1024 | 多数 SVG 超 512 tokens |
| do_sample | True | 贪心解码质量差，但 sampling 有退化风险 |

**为什么不用 no_repeat_ngram_size**：SVG 合法地大量重复（许多 `<circle>`、`<path>`），禁重复只会逼出损坏属性。

### 7.2 extract_svg 容错

**小模型最常见的失败模式**：输出了开标签 `<svg>` 但永远不输出闭标签 `</svg>`（通常撞到 token 预算）。

```python
def extract_svg(text: str) -> str:
    text = text.strip()
    
    # 剥掉 markdown 代码围栏
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1:
            text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    
    svg_start = text.find("<svg")
    if svg_start == -1:
        return text   # 完全没有 svg 开标签，原样返回让 reward 报告无效
    
    svg_end = text.rfind("</svg>")
    if svg_end != -1 and svg_end > svg_start:
        return text[svg_start:svg_end + 6]
    
    # 有开标签但没闭标签：截到最后一个 '>'，再补 </svg>
    fragment = text[svg_start:]
    last_gt = fragment.rfind(">")
    if last_gt != -1 and last_gt < len(fragment) - 1:
        fragment = fragment[:last_gt + 1]
    return fragment + "</svg>"
```

**策略**：追加一个闭标签，让文档至少能被解析，而不是整段被判 0 分。

### 7.3 退化检测 + retry 机制

**问题**：do_sample 在 270M 小模型上，某些 prompt 会触发病态重复循环：
- 同一个 `<circle .../>` 完全重复几十次
- path `d="M128 128 L128 128 L128 128..."` 坐标死循环

**检测策略**：

```python
def _is_degenerate(text: str) -> bool:
    import re
    # 检测 8+ 字符子串重复 5+ 次
    if re.search(r'(.{8,}?)\1{4,}', text):
        return True
    # 检测同一个元素标签重复 5+ 次
    if re.search(r'(<\w+[^>]*?/>\s*)\1{4,}', text):
        return True
    return False
```

**为什么 8 字符 + 5 次重复**：合法 SVG 也会重复相似元素，但坐标/颜色不同——完全相同的子串重复 5 次几乎只出现在退化输出里。

**retry 机制**：

```python
def evaluate_model(model, tokenizer, samples, label, num_samples=2):
    results = []
    for sample in samples:
        prompt = extract_prompt(sample)
        best = None
        
        # 多采样取最佳
        for k in range(num_samples):
            svg = generate_svg(model, tokenizer, prompt)
            if _is_degenerate(svg):
                log(f"      sub {k+1}: DEGENERATE, 跳过")
                continue
            reward = compute_reward(svg, prompt)
            if best is None or reward["total"] > best["reward"]["total"]:
                best = {"svg": svg, "reward": reward}
        
        # 全部退化的兜底：额外重采样
        if best is None:
            for _ in range(num_samples):
                svg = generate_svg(model, tokenizer, prompt)
                reward = compute_reward(svg, prompt)
                best = {"svg": svg, "reward": reward}
                if reward["total"] > 0:
                    break
        
        results.append(best)
    return results
```

**效果**：把 FT[11] 从 0.0（两次采样都退化）救回到 0.9348。

---

## 八、训练与评测流程

### 8.1 启动训练

**重要：用 `python.exe -u` 直接调用，不要用 `conda run`**（输出缓冲会误判"卡住"）。

```powershell
cd c:\Users\dtft\Desktop\project_file\course\model_math\lora_2\student_kit

# 直接调用 python.exe（-u 禁用缓冲）
& C:\Users\dtft\miniconda3\envs\lora_env\python.exe -u train_peft.py --max_length 1536
```

### 8.2 监控训练

训练开始后会先做 sanity check：

```
=== Dataset Sanity Check ===
Train samples: 185
Valid samples: 17
Sample 0 - input_ids len: 872, labels len: 872
Sample 0 - valid (non -100) labels: 520
Sample 0 - first 10 valid labels: [<s>, '<svg', ' xmlns', ...]
=== Forward Pass Test ===
Test loss: 1.679
=== Forward Pass OK ===
```

然后开始训练，每 5 步打印一次 loss：

```
{'loss': 1.679, 'grad_norm': 2.34, 'learning_rate': 1.1e-4, 'epoch': 0.22}
{'loss': 1.236, 'grad_norm': 1.08, 'learning_rate': 1.5e-4, 'epoch': 0.43}
...
{'eval_loss': 0.970, 'epoch': 2.0}
{'loss': 0.918, 'grad_norm': 0.35, 'learning_rate': 2.3e-5, 'epoch': 2.52}
{'loss': 0.907, 'grad_norm': 0.33, 'learning_rate': 1.0e-6, 'epoch': 3.0}
{'eval_loss': 0.946, 'epoch': 3.0}
```

**正常指标**：
- train_loss 从 1.679 稳步下降到 0.907
- eval_loss 从 1.087 → 0.970 → 0.946，稳步下降未过拟合
- 训练时间约 4.8 分钟

### 8.3 运行自评

```powershell
& C:\Users\dtft\miniconda3\envs\lora_env\python.exe -u eval_self.py --num_samples 2
```

自评流程：
1. 加载基座模型，对 17 个 valid 样本生成 SVG，用 reward 打分
2. 释放基座模型显存
3. 加载基座 + adapter，合并权重，对同样的 17 个样本生成打分
4. 输出对比结果到 results.json

```
=== Phase 1: Baseline Evaluation ===
Loading baseline model...
  [baseline] Sample 1/17 (n=2)...
      sub 1/2: total=0.619 valid=True
      sub 2/2: total=0.619 valid=True
  ...

=== Phase 2: Fine-tuned Evaluation ===
Loading fine-tuned model...
  [fine-tuned] Sample 1/17 (n=2)...
      sub 1/2: total=0.8013 valid=True
  ...

=== Summary ===
Baseline avg reward: 0.6190
Fine-tuned avg reward: 0.8013
Delta: 0.1823
Baseline valid: 17/17
Fine-tuned valid: 17/17
```

---

## 九、踩坑与解决方案

### 9.1 `conda run` 输出缓冲误判"卡住"

**现象**：用 `conda run -n lora_env python ...` 跑训练时，stdout 被完全缓冲，进度日志不刷新。进程实际在正常运行（GPU 100%、显存 1231MiB），但看起来像"卡住"。

**根因**：`conda run` 默认缓冲 stdout。

**修复**：改用 `python.exe -u` 直接调用，`-u` 禁用缓冲。

### 9.2 max_length=1024 导致训练数据不足

**现象**：首轮用 max_length=1024 训练（83 样本，38%），自评结果微调 0.5098 < 基座 0.5189，回归。

**根因**：token 分布审计显示 `<=1024` 仅占 63%，但实际加上 prompt 前缀后能保留的更少（83 样本，38%）。

**修复**：max_length 提到 1536，保留 89% 样本（185/219）。

### 9.3 fp16 精度下溢

**现象**：用 fp16 训练时 loss=0、grad_norm=NaN。

**根因**：Gemma3 的某些数值在 fp16 下溢出。

**修复**：改用 bf16。bf16 的指数位与 fp32 相同，动态范围更大。

### 9.4 生成退化循环

**现象**：temp=0.3 + repetition_penalty=1.15 时，模型陷入病态重复循环：
```
<path d="M10 15 L15 20 L20 25 L30 30 L40 35 L50 35 L60 35 L70 35 L80 35 L90 35 L100 35..."
```
模型撞满 1024 tokens（45s）才停，整段垃圾 reward=0。

**根因**：
- temperature=0.3 偏高，随机性触发退化路径
- repetition_penalty=1.15 对 SVG 的高度重复结构过强，逼模型生造垃圾 token

**修复（四管齐下）**：
1. temperature 0.3 → 0.2
2. repetition_penalty 1.15 → 1.05
3. 多采样取最佳（num_samples=2）
4. 退化检测 + retry

### 9.5 XML 嵌套修复的栈 vs 计数

**现象**：用计数法修复 `<g>` 嵌套时，某些样本仍解析失败 reward=0。

**根因**：计数法无法检测孤立的 `</g>`（1 开 + 1 孤立闭 = 1==1，误判为平衡）。

**修复**：改用栈式扫描，按文档顺序处理。

### 9.6 训练/eval prompt 不对齐

**现象**：训练 loss 下降但自评 reward 不升反降。

**根因**：训练时 prompt 拼装方式与 eval 生成前缀不一致。

**修复**：训练和 eval 都用 `apply_chat_template(..., add_generation_prompt=True)`。

---

## 十、交付物清单

| 文件 | 说明 |
|------|------|
| `adapter/` | LoRA 适配器权重（r=8, alpha=16, target=q,k,v,o）|
| `student_kit/reward.py` | LogoGrader：11 维度打分 + `_salvage_xml` 5 阶段容错 + `_fix_group_nesting` 栈式扫描 |
| `student_kit/train_peft.py` | 训练脚本（flatten_svg 预处理、跳过不截断、追加 EOS、add_generation_prompt=True）|
| `student_kit/eval_self.py` | 自评脚本（temp=0.2、rep_penalty=1.05、num_samples=2 多采样取最佳、`_is_degenerate` 退化检测 + retry、extract_svg 容错）|
| `student_kit/train_config.yaml` | 训练超参数 |
| `results.json` | 完整评测结果（17 样本，基座 0.6190 vs 微调 0.8013，delta=+0.1823，17/17 有效）|
| `report.md` | 端到端开发流程报告 |

---

## 附录：关键命令速查

```powershell
# 1. 训练（用 python.exe -u，不要用 conda run）
cd student_kit
& C:\Users\dtft\miniconda3\envs\lora_env\python.exe -u train_peft.py --max_length 1536

# 2. 自评
& C:\Users\dtft\miniconda3\envs\lora_env\python.exe -u eval_self.py --num_samples 2

# 3. 检查 GPU 状态
nvidia-smi

# 4. 检查进程
Get-Process python | Select-Object Id, CPU, WorkingSet

# 5. 终止卡住的进程
Stop-Process -Id <PID> -Force
```

---

## 附录：进一步学习资源

- **LoRA 原始论文**：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **PEFT 文档**：https://huggingface.co/docs/peft
- **Transformers Trainer**：https://huggingface.co/docs/transformers/main_classes/trainer
- **Gemma3 模型**：https://www.modelscope.cn/models/google/gemma-3-270m-it

---

*本指南基于实际项目开发经验编写，所有代码引用均对应仓库实际文件。*
