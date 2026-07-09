import re
import xml.etree.ElementTree as ET
from collections import Counter


class LogoGrader:
    """SVG 徽标评分器。

    设计取向（与一般“先解析再扣分”的写法不同）：
    采用「先抢救再评分」策略——270M 小模型无法可靠闭合 XML 标签，
    若一次解析失败就归零，会系统性低估模型的真实形状/配色能力。
    因此 parse 失败时进入多阶段 salvage 流程，尽量把有效结构捞出来。
    """

    # 视为非法/危险的标签（安全维度）
    FORBIDDEN_TAGS = {"image", "script", "foreignObject", "iframe", "use"}
    # 允许出现的结构标签
    SUPPORTED_TAGS = {
        "svg", "defs", "g", "path", "circle", "ellipse", "rect",
        "polygon", "polyline", "line", "text", "linearGradient",
        "radialGradient", "stop", "clipPath", "pattern", "filter",
    }

    TARGET_VIEWBOX = "0 0 256 256"
    # 画布外硬边界（超出即越界）
    HARD_MIN = 0
    HARD_MAX = 256
    # 内容居中软边界（落在此区间内奖励满分子项）
    SOFT_MIN = 20
    SOFT_MAX = 236

    # 配色数量上下限（比常见设定更宽松，鼓励多样性）
    PALETTE_MIN = 2
    PALETTE_MAX = 12
    # 元素数量上下限（上限收紧，避免噪声刷分）
    SHAPE_MIN = 2
    SHAPE_MAX = 80

    HEX_COLOR_RE = re.compile(r'^#([0-9a-fA-F]{3}){1,2}$')
    NAME_COLOR_RE = re.compile(r'^[a-zA-Z]+$')

    # 提示词中可能出现的颜色词（仅收录无歧义的颜色名，避免把形状词误判为颜色）
    COLOR_VOCAB = {
        "black", "white", "red", "green", "blue", "yellow", "orange",
        "purple", "pink", "brown", "gray", "grey", "navy", "teal", "lime",
        "cyan", "magenta", "maroon", "olive", "silver", "gold", "coral",
        "salmon", "crimson", "amber", "ivory", "cream", "charcoal", "sage",
        "violet", "indigo", "turquoise", "beige", "tan", "plum", "khaki",
        "lavender", "peach", "mint", "ruby", "emerald", "azure", "slate",
        "copper", "bronze", "mustard", "rose", "sky", "forest", "sand",
        "pearl", "wine", "jade", "ochre", "rust", "apricot", "chocolate",
    }

    # 形状词到 SVG 标签的映射（用于形状保真度度量）
    SHAPE_LEXICON = {
        "circle": "circle", "circular": "circle", "round": "circle",
        "square": "rect", "rectangle": "rect", "rectangular": "rect",
        "ellipse": "ellipse", "oval": "ellipse",
        "hexagon": "polygon", "hexagonal": "polygon",
        "triangle": "polygon", "triangular": "polygon",
        "star": "polygon", "diamond": "polygon", "pentagon": "polygon",
        "polygon": "polygon",
        "line": "line", "ray": "line", "bar": "line",
        "path": "path", "curve": "path", "arc": "path",
    }

    # 需要自闭合的 void 元素（小模型常写成未闭合容器）
    _SELF_CLOSE_TAGS = {
        "rect", "circle", "ellipse", "line", "path", "polygon",
        "polyline", "stop", "use", "image",
    }

    @staticmethod
    def _fix_group_nesting(text: str) -> str:
        """栈式修复 <g> 容器嵌套。

        小模型两种典型错误：
        - 孤立 </g>：闭合标签前没有匹配的 <g>。直接丢弃（留着会破坏解析）。
        - 未闭合 <g>：开标签后从未出现 </g>。在 </svg> 前补齐。

        关键点：必须用栈按文档顺序扫描，而不能用「开 vs 闭计数」——
        计数法无法识别顺序错误的孤立闭合标签（1 开 + 1 孤立闭 = 1==1，
        误判为已平衡）。这是 reward 侧最后一个致命 bug。
        """
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

    @staticmethod
    def _salvage_xml(text: str) -> str:
        """尽力修复小模型常见的 XML 语法错误。

        策略是「逐项剥离问题源」，而不是整体丢弃：
        1. 移除 <defs> 块（非必要，且常未闭合）。
        2. 把未自闭合的 void 元素补上 />。
        3. 栈式修复 <g> 嵌套。
        4. 去掉属性间的逗号（x="1", y="2" → x="1" y="2"）。
        5. 标签内同名属性去重（保留首个）。
        """
        out = text

        # 1. 剥离 <defs>...</defs>，以及无闭合标签的孤儿 <defs>
        out = re.sub(r'<defs\b[^>]*>.*?</defs\s*>', '', out, flags=re.DOTALL)
        out = re.sub(r'<defs\b[^>]*>(?:(?!</svg>).)*$', '', out, flags=re.DOTALL)

        # 2. 未自闭合的 void 元素补上 />
        for tag in LogoGrader._SELF_CLOSE_TAGS:
            out = re.sub(
                rf'<{tag}\b([^>]*?)(?<!/)>',
                rf'<{tag}\1/>',
                out,
            )

        # 3. <g> 嵌套栈式修复（替代旧的计数法）
        out = LogoGrader._fix_group_nesting(out)

        # 4. 属性间逗号 → 空格
        out = re.sub(r'"\s*,\s*', '" ', out)

        # 5. 同标签内同名属性去重
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

    @staticmethod
    def parse_svg(svg_text):
        """解析 SVG，失败时走渐进式 salvage。

        阶段 0：严格解析。
        阶段 1：完整 salvage 后重试。
        阶段 2：兜底——截掉 <defs> 之后所有内容，只保留前面的形状。
        """
        # 阶段 0：严格解析
        try:
            return ET.fromstring(svg_text)
        except Exception:
            pass

        # 阶段 1：完整修复后重试
        salvaged = LogoGrader._salvage_xml(svg_text)
        if salvaged != svg_text:
            try:
                return ET.fromstring(salvaged)
            except Exception:
                pass

        # 阶段 2：兜底，只保留 <defs> 之前的部分
        truncated = re.sub(r'<defs\b.*', '', svg_text, flags=re.DOTALL)
        truncated = truncated.rstrip()
        if '</svg>' not in truncated and '<svg' in truncated:
            truncated += '</svg>'
        if truncated != svg_text:
            try:
                return ET.fromstring(truncated)
            except Exception:
                pass

        return None

    @staticmethod
    def _safe_float(value, default=0.0):
        """安全解析数值属性，失败返回默认值。

        小模型经常输出非数值属性（如 cx="auto"），不做防护会让整个
        reward 计算崩溃。
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _viewbox_ok(root):
        return root.attrib.get("viewBox", "") == LogoGrader.TARGET_VIEWBOX

    @staticmethod
    def _has_namespace_and_viewbox(root, raw_text=""):
        has_viewbox = "viewBox" in root.attrib
        # xmlns 会被 ElementTree 当作命名空间声明消费掉，
        # 因此不能从 root.attrib 里找，要看原始文本或命名空间标签前缀。
        has_xmlns = "xmlns" in raw_text or root.tag.startswith("{")
        return has_viewbox and has_xmlns

    @staticmethod
    def _scan_forbidden_tags(root):
        found = []
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag in LogoGrader.FORBIDDEN_TAGS:
                found.append(tag)
        return found

    @staticmethod
    def _scan_external_refs(root):
        refs = []
        for elem in root.iter():
            for attr, value in elem.attrib.items():
                if value.startswith(("http://", "https://", "//")):
                    refs.append(f"{elem.tag}:{attr}={value}")
        return refs

    @staticmethod
    def _collect_coords(root):
        coords = []
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == "circle":
                cx = LogoGrader._safe_float(elem.attrib.get("cx", "0"))
                cy = LogoGrader._safe_float(elem.attrib.get("cy", "0"))
                coords.extend([cx, cy])
            elif tag == "rect":
                x = LogoGrader._safe_float(elem.attrib.get("x", "0"))
                y = LogoGrader._safe_float(elem.attrib.get("y", "0"))
                w = LogoGrader._safe_float(elem.attrib.get("width", "0"))
                h = LogoGrader._safe_float(elem.attrib.get("height", "0"))
                coords.extend([x, y, x + w, y + h])
            elif tag == "ellipse":
                cx = LogoGrader._safe_float(elem.attrib.get("cx", "0"))
                cy = LogoGrader._safe_float(elem.attrib.get("cy", "0"))
                coords.extend([cx, cy])
            elif tag == "line":
                x1 = LogoGrader._safe_float(elem.attrib.get("x1", "0"))
                y1 = LogoGrader._safe_float(elem.attrib.get("y1", "0"))
                x2 = LogoGrader._safe_float(elem.attrib.get("x2", "0"))
                y2 = LogoGrader._safe_float(elem.attrib.get("y2", "0"))
                coords.extend([x1, y1, x2, y2])
            elif tag in ("polygon", "polyline"):
                pts = elem.attrib.get("points", "")
                for v in pts.replace(',', ' ').split():
                    f = LogoGrader._safe_float(v, None)
                    if f is not None:
                        coords.append(f)
            elif tag == "path":
                d = elem.attrib.get("d", "")
                nums = re.findall(r'-?\d+\.?\d*', d)
                coords.extend([LogoGrader._safe_float(n) for n in nums])
        return coords

    @staticmethod
    def _collect_colors(root):
        colors = []
        for elem in root.iter():
            for attr, value in elem.attrib.items():
                if attr.lower() in ("fill", "stroke") and value not in ("none", ""):
                    if LogoGrader.HEX_COLOR_RE.match(value):
                        colors.append(value.lower())
                    elif LogoGrader.NAME_COLOR_RE.match(value):
                        colors.append(value.lower())
        return colors

    @staticmethod
    def _count_shapes(root):
        count = 0
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag in {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line"}:
                count += 1
        return count

    @staticmethod
    def _is_degenerate(root):
        colors = LogoGrader._collect_colors(root)
        shapes = LogoGrader._count_shapes(root)
        if shapes == 0:
            return True, "empty"
        if len(set(colors)) <= 1:
            return True, "monochrome"
        return False, None


def compute_reward(svg_text, prompt=None):
    """对一个生成的 SVG 打分，返回总分与各维度细分。

    评分采用加权求和。权重分配思路：
    - 结构合法性（well_formed + attrs + viewbox）合计 3.5，最高，因为
      解析失败的 SVG 一切免谈；
    - 几何合理性（coord_hard + coord_soft）合计 2.0；
    - 安全性（forbidden_tags + external_refs）合计 2.0；
    - 内容丰富度（palette + density + non_degenerate）合计 3.0；
    - 提示词保真度（fidelity）1.5，权重适中——270M 语言理解有限，
      过高会引入噪声。
    """
    dims = {}
    score = 0.0
    wsum = 0.0

    root = LogoGrader.parse_svg(svg_text)

    if root is None:
        return {
            "valid": False,
            "total": 0.0,
            "reason": "SVG parse failed",
            "breakdown": {}
        }

    # 维度 1：XML 结构合法（1.5）
    dims["well_formed"] = 1.0
    score += 1.5
    wsum += 1.5

    # 维度 2：含 xmlns + viewBox（1.0）
    if LogoGrader._has_namespace_and_viewbox(root, svg_text):
        dims["attrs"] = 1.0
        score += 1.0
    else:
        dims["attrs"] = 0.0
    wsum += 1.0

    # 维度 3：viewBox 精确等于 0 0 256 256（1.0）
    if LogoGrader._viewbox_ok(root):
        dims["viewbox"] = 1.0
        score += 1.0
    else:
        dims["viewbox"] = 0.0
    wsum += 1.0

    # 维度 4：无非法标签（1.0）
    forbidden = LogoGrader._scan_forbidden_tags(root)
    if not forbidden:
        dims["no_forbidden_tags"] = 1.0
        score += 1.0
    else:
        dims["no_forbidden_tags"] = 0.0
    wsum += 1.0

    # 维度 5：无外链引用（1.0）
    ext_refs = LogoGrader._scan_external_refs(root)
    if not ext_refs:
        dims["no_external_refs"] = 1.0
        score += 1.0
    else:
        dims["no_external_refs"] = 0.0
    wsum += 1.0

    # 维度 6/7：坐标硬边界 + 软边界（居中度）
    coords = LogoGrader._collect_coords(root)
    if coords:
        cmin = min(coords)
        cmax = max(coords)
        # 硬边界：是否落在 [0, 256]
        if LogoGrader.HARD_MIN <= cmin and cmax <= LogoGrader.HARD_MAX:
            dims["coord_in_bounds"] = 1.0
            score += 1.5
        else:
            overflow = max(abs(cmin), abs(cmax - 256))
            dims["coord_in_bounds"] = max(0.0, 1 - overflow / 256)
            score += dims["coord_in_bounds"]
        wsum += 1.5

        # 软边界：内容是否落在居中区间 [20, 236]
        if LogoGrader.SOFT_MIN <= cmin and cmax <= LogoGrader.SOFT_MAX:
            dims["coord_centered"] = 1.0
            score += 0.5
        else:
            slack = max(LogoGrader.SOFT_MIN - cmin, cmax - LogoGrader.SOFT_MAX, 0)
            dims["coord_centered"] = max(0.0, 1 - slack / 100)
            score += dims["coord_centered"]
        wsum += 0.5
    else:
        # 没有可解析坐标（如纯 text），给中性分避免一边倒
        dims["coord_in_bounds"] = 0.5
        dims["coord_centered"] = 0.5
        score += 1.0
        wsum += 1.0

    # 维度 8：配色数量
    colors = LogoGrader._collect_colors(root)
    n_colors = len(set(colors))
    if LogoGrader.PALETTE_MIN <= n_colors <= LogoGrader.PALETTE_MAX:
        dims["palette"] = 1.0
        score += 1.0
    elif n_colors < LogoGrader.PALETTE_MIN:
        dims["palette"] = n_colors / LogoGrader.PALETTE_MIN
        score += dims["palette"]
    else:
        dims["palette"] = max(0.0, 1 - (n_colors - LogoGrader.PALETTE_MAX) / 20)
        score += dims["palette"]
    wsum += 1.0

    # 维度 9：元素密度
    n_shapes = LogoGrader._count_shapes(root)
    if LogoGrader.SHAPE_MIN <= n_shapes <= LogoGrader.SHAPE_MAX:
        dims["density"] = 1.0
        score += 1.0
    elif n_shapes < LogoGrader.SHAPE_MIN:
        dims["density"] = n_shapes / LogoGrader.SHAPE_MIN
        score += dims["density"]
    else:
        dims["density"] = max(0.0, 1 - (n_shapes - LogoGrader.SHAPE_MAX) / 200)
        score += dims["density"]
    wsum += 1.0

    # 维度 10：非退化（非空、非单色）
    degenerate, reason = LogoGrader._is_degenerate(root)
    if not degenerate:
        dims["non_degenerate"] = 1.0
        score += 1.0
    else:
        dims["non_degenerate"] = 0.0
        dims["degenerate_reason"] = reason
    wsum += 1.0

    # 维度 11：提示词保真度（颜色 + 形状）
    if prompt:
        prompt_lower = prompt.lower()
        svg_colors = set(LogoGrader._collect_colors(root))

        # (a) 颜色保真：提示词里的颜色是否出现在 fill/stroke
        prompt_hex = set(re.findall(r'#[0-9a-fA-F]{6}\b', prompt_lower))
        prompt_named = {
            w for w in re.findall(r'[a-zA-Z]+', prompt_lower)
            if LogoGrader.NAME_COLOR_RE.match(w) and w in LogoGrader.COLOR_VOCAB
        }
        prompt_colors = {c.lower() for c in (prompt_hex | prompt_named)}
        color_hit = (len(prompt_colors & svg_colors) / len(prompt_colors)
                     if prompt_colors else None)

        # (b) 形状保真：提示词里的形状词是否对应到实际 SVG 标签
        present_tags = {
            elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            for elem in root.iter()
        }
        matched_shapes = []
        for word, svg_tag in LogoGrader.SHAPE_LEXICON.items():
            if word in prompt_lower and svg_tag in present_tags:
                matched_shapes.append(word)
        prompt_shapes = [w for w in LogoGrader.SHAPE_LEXICON if w in prompt_lower]
        shape_hit = (len(matched_shapes) / len(prompt_shapes)
                     if prompt_shapes else None)

        parts = [p for p in (color_hit, shape_hit) if p is not None]
        dims["fidelity"] = (sum(parts) / len(parts)) if parts else 0.5
        dims["fidelity_detail"] = {
            "color_hit": round(color_hit, 3) if color_hit is not None else None,
            "shape_hit": round(shape_hit, 3) if shape_hit is not None else None,
        }
        score += dims["fidelity"]
    else:
        dims["fidelity"] = 0.5
        score += 0.5
    wsum += 1.0

    total = score / wsum if wsum > 0 else 0.0

    return {
        "valid": True,
        "total": round(total, 4),
        "breakdown": dims,
        "details": {
            "color_count": n_colors,
            "element_count": n_shapes,
            "coord_range": f"[{min(coords):.1f}, {max(coords):.1f}]" if coords else "N/A",
            "forbidden_tags": forbidden,
            "external_refs": ext_refs
        }
    }


if __name__ == "__main__":
    demo_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
        <circle cx="128" cy="128" r="100" fill="#1B3A5C"/>
        <path d="M100 150 L128 100 L156 150 Z" fill="#F2A93B"/>
    </svg>'''

    demo_prompt = "A circular badge with a triangle in golden orange color"

    result = compute_reward(demo_svg, demo_prompt)
    print("Reward:", result["total"])
    print("Breakdown:", result["breakdown"])
    print("Details:", result["details"])
