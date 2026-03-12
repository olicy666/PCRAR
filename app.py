from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from collections import Counter
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from raven3d.pcrar_dataset import PCRARDatasetGenerator, PCRARConfig
from raven3d.pcrar_rules import RuleTemplate

RECORDS_PATH = Path("exam_records.csv")
RESULTS_DIR = Path("results")
TOTAL_QUESTIONS = 10
PLY_HEIGHT = 320
BIG_PLY_HEIGHT = 720
OPTION_PLY_HEIGHT = 210
POINTS_PER_CLOUD = 8192
BIG_VIEW_POINT_SIZE_SCALE = 6.0
BIG_VIEW_GRID_SPACING = 4.2
BIG_VIEW_ROW_DEPTH = 1.8
BIG_VIEW_CAMERA_FIT_MARGIN = 1.0
BIG_VIEW_MAX_RENDER_POINTS_PER_CLOUD = 3200

# 规则名称映射
RULE_NAMES = {
    RuleTemplate.PROGRESSION: "递进规则",
    RuleTemplate.CYCLE: "Distribute-three规则",
    RuleTemplate.COPY: "Distribute-three规则",
    RuleTemplate.COUNT: "增减规则",
    RuleTemplate.CONSERVATION: "守恒规则",
    RuleTemplate.PERMUTATION: "置换规则",
    RuleTemplate.SYMMETRY: "对称规则",
}

RULE_SHORT_NAMES = {
    RuleTemplate.PROGRESSION: "递进",
    RuleTemplate.CYCLE: "Distribute-three",
    RuleTemplate.COPY: "Distribute-three",
    RuleTemplate.COUNT: "增减",
    RuleTemplate.CONSERVATION: "守恒",
    RuleTemplate.PERMUTATION: "置换",
    RuleTemplate.SYMMETRY: "对称",
}

ERROR_TYPE_KEYS = ["analogical_wrong_relation", "perceptual_plausible", "irrelevant"]
ATTRIBUTE_LABELS = {
    "shape": "形状",
    "size": "尺寸",
    "density": "密度",
    "color": "颜色",
    "count": "数量",
    "rotation": "姿态",
    "position": "位置",
}
ATTRIBUTE_ORDER = list(ATTRIBUTE_LABELS.keys())
PERTURBATION_TYPE_KEYS = ["density", "jitter", "quantize", "outlier"]
PERTURBATION_LABELS = {
    "density": "密度降采样",
    "jitter": "坐标抖动",
    "quantize": "坐标量化",
    "outlier": "离群点替换",
}

# Streamlit 端展示的规则集合：Copy 已并入 Distribute-three，不再单独出题。
MODE_RULE_TEMPLATES: List[RuleTemplate] = [
    RuleTemplate.PROGRESSION,
    RuleTemplate.CYCLE,
    RuleTemplate.COUNT,
    RuleTemplate.CONSERVATION,
    RuleTemplate.PERMUTATION,
    RuleTemplate.SYMMETRY,
]


def canonical_template(template: RuleTemplate) -> RuleTemplate:
    """规则模板规范化（兼容历史 Copy->Distribute-three）。"""
    return RuleTemplate.CYCLE if template == RuleTemplate.COPY else template

# 题型名称（固定为 matrix）
TASK_TYPE_NAMES = {
    "matrix_3x3": "3x3矩阵推理",
}

# 生成规则 ID 列表：矩阵题-规则编号
def generate_mode_list() -> List[str]:
    """生成模式列表：矩阵推理-规则1, ..."""
    modes = []
    for idx, template in enumerate(MODE_RULE_TEMPLATES, 1):
        del template
        mode_id = f"矩阵推理-规则{idx}"
        modes.append(mode_id)
    return modes

# 模式 ID 到 (rule_template) 的映射
def parse_mode(mode: str) -> RuleTemplate:
    """解析模式 ID，返回 rule_template"""
    rule_num = int(mode.split("规则")[1]) - 1
    if rule_num < 0 or rule_num >= len(MODE_RULE_TEMPLATES):
        return MODE_RULE_TEMPLATES[0]
    return MODE_RULE_TEMPLATES[rule_num]

def get_mode_description(mode: str) -> str:
    """获取模式描述"""
    template = parse_mode(mode)
    rule_name = RULE_NAMES[template]
    return f"{rule_name} ({template.value})"


def mode_display_name(mode: str) -> str:
    """把 mode 统一转换为管理员后台展示名称。"""
    if not isinstance(mode, str):
        return str(mode)
    text = mode.strip()
    if not text:
        return ""
    if text in RULE_SHORT_NAMES.values():
        return text
    if text.startswith("矩阵推理-规则"):
        try:
            return RULE_SHORT_NAMES[canonical_template(parse_mode(text))]
        except Exception:
            return text
    for template, short_name in RULE_SHORT_NAMES.items():
        if text == template.value:
            return short_name
    return text


def normalize_template_value(template_name: str, default: RuleTemplate) -> str:
    """把 meta 中的规则名标准化为当前语义（Copy 归并到 Distribute-three）。"""
    try:
        tpl = RuleTemplate.from_any(template_name)
    except Exception:
        tpl = default
    return canonical_template(tpl).value


def axis_to_attribute(axis: Optional[str]) -> Optional[str]:
    """把规则轴映射到可统计的属性。"""
    if not axis:
        return None
    key = str(axis).strip()
    direct_map = {
        "r": "size",
        "R": "rotation",
        "p": "position",
        "count": "count",
        "size_conservation": "size",
        "slot_permutation": "position",
        "distribute_three_shape": "shape",
        "distribute_three_size": "size",
        "distribute_three_density": "density",
        "distribute_three_color": "color",
        "cycle_shape_distribute3": "shape",
        "cycle_size_distribute3": "size",
        "cycle_density_distribute3": "density",
        "cycle_color_distribute3": "color",
        "copy_shape_cycle": "shape",
        "copy_size_cycle": "size",
        "copy_density_cycle": "density",
        "shape": "shape",
        "size": "size",
        "density": "density",
        "color": "color",
    }
    if key in direct_map:
        return direct_map[key]
    if "shape" in key:
        return "shape"
    if "size" in key:
        return "size"
    if "density" in key:
        return "density"
    if "color" in key:
        return "color"
    if "count" in key:
        return "count"
    if "slot" in key or "position" in key:
        return "position"
    if "rot" in key or "pose" in key:
        return "rotation"
    return None


def template_default_attribute(template_name: Optional[str]) -> Optional[str]:
    if not template_name:
        return None
    mapping = {
        RuleTemplate.COUNT.value: "count",
        RuleTemplate.CONSERVATION.value: "size",
        RuleTemplate.PERMUTATION.value: "position",
    }
    return mapping.get(str(template_name))


def extract_question_attributes(entry: Dict) -> List[str]:
    """提取题目涉及属性（含横/纵两个关系参数）。"""
    attrs = set()
    rule = entry.get("rule", {})
    param_candidates = [
        rule.get("params", {}) if isinstance(rule, dict) else {},
        rule.get("vertical_params", {}) if isinstance(rule, dict) else {},
        entry.get("rule_params", {}),
        entry.get("rule_params_vertical", {}),
    ]
    for params in param_candidates:
        if not isinstance(params, dict):
            continue
        attr = axis_to_attribute(params.get("axis"))
        if attr:
            attrs.add(attr)
    template_name = entry.get("rule_template") or (rule.get("template") if isinstance(rule, dict) else None)
    fallback_attr = template_default_attribute(template_name)
    if fallback_attr:
        attrs.add(fallback_attr)
    return [name for name in ATTRIBUTE_ORDER if name in attrs]


def extract_perturbation_type(entry: Dict) -> Optional[str]:
    """提取题目点云扰动类型。"""
    perturb = entry.get("point_cloud_row_perturbation", {})
    if not isinstance(perturb, dict):
        return None
    selected_type = perturb.get("selected_type")
    if not isinstance(selected_type, str):
        return None
    key = selected_type.strip().lower()
    return key if key in PERTURBATION_TYPE_KEYS else None

MODE_IDS = generate_mode_list()
DIFFICULTY_IDS = ["easy"]
DIFFICULTY_LABELS = {
    "easy": "简单（Easy）",
}
RECORD_COLUMNS = ["username", "mode", "score", "total", "accuracy", "reason"]


def render_theme_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245, 158, 11, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.14), transparent 26%),
                linear-gradient(180deg, #f8fafc 0%, #edf4f7 100%);
        }
        html, body, [class*="css"]  {
            font-family: "Trebuchet MS", "Segoe UI", "PingFang SC", "Hiragino Sans GB", sans-serif;
        }
        .main .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #10263d 0%, #173d61 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: #ecf5ff;
        }
        [data-testid="stSidebar"] .stCaption {
            color: #d3e7f9;
        }
        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 24px;
            padding: 1.15rem 1rem 0.55rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(8px);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            padding: 0.35rem 0.6rem;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            min-height: 2.85rem;
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #0f4c81 0%, #1f87c9 100%);
            color: #ffffff;
            font-weight: 700;
            box-shadow: 0 12px 24px rgba(15, 76, 129, 0.18);
        }
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(135deg, #0c406b 0%, #176ea7 100%);
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
            border-radius: 16px !important;
            border: 1px solid rgba(15, 23, 42, 0.12) !important;
            background: rgba(255, 255, 255, 0.9) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stTextInput"] input,
        [data-testid="stSidebar"] div[data-testid="stNumberInput"] input,
        [data-testid="stSidebar"] div[data-baseweb="select"] *,
        [data-testid="stSidebar"] div[data-testid="stMultiSelect"] div[data-baseweb="select"] * {
            color: #0f172a !important;
        }
        div[role="radiogroup"] {
            gap: 0.45rem;
        }
        div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 23, 42, 0.1);
            border-radius: 999px;
            padding: 0.3rem 0.85rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stAlert"] {
            border-radius: 20px;
            border: 1px solid rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 22px;
            overflow: hidden;
        }
        div[data-testid="stCheckbox"] label > div:first-child {
            transform: scale(1.28);
            transform-origin: left center;
        }
        div[data-testid="stCheckbox"] label {
            align-items: center;
        }
        .app-hero {
            padding: 1.35rem 1.5rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.88) 0%, rgba(237, 246, 255, 0.96) 55%, rgba(255, 248, 235, 0.92) 100%);
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 24px 50px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }
        .app-hero-eyebrow {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #b45309;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .app-hero h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.1;
            color: #0f172a;
        }
        .app-hero p {
            margin: 0.6rem 0 0;
            font-size: 1rem;
            line-height: 1.6;
            color: #334155;
        }
        .app-hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.95rem;
        }
        .meta-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.34rem 0.8rem;
            border-radius: 999px;
            font-size: 0.88rem;
            font-weight: 700;
            color: #0f4c81;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(15, 76, 129, 0.12);
        }
        .section-panel {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.9rem;
        }
        .section-kicker {
            color: #0f4c81;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .section-title {
            color: #0f172a;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .section-text {
            color: #475569;
            line-height: 1.65;
            font-size: 0.95rem;
        }
        .section-panel.compact {
            min-height: 150px;
        }
        .question-facts {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.7rem;
        }
        .question-fact {
            min-width: 132px;
            padding: 0.7rem 0.85rem;
            border-radius: 18px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid rgba(15, 23, 42, 0.08);
        }
        .question-fact-label {
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 0.15rem;
        }
        .question-fact-value {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_banner(title: str, subtitle: str, eyebrow: str, chips: Optional[List[str]] = None) -> None:
    chip_html = "".join(f'<span class="meta-chip">{escape(chip)}</span>' for chip in (chips or []))
    st.markdown(
        f"""
        <div class="app-hero">
          <div class="app-hero-eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(title)}</h1>
          <p>{escape(subtitle)}</p>
          <div class="app-hero-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pl_component(ply_content_str: str, height: int = PLY_HEIGHT, reset_nonce: int = 0) -> None:
    import uuid

    container_id = f"pc_{reset_nonce}_{uuid.uuid4().hex}"
    overlay_id = f"{container_id}_overlay"
    ply_json = json.dumps(ply_content_str)
    html = f"""
    <div style="width:100%; height:{height}px; position:relative; background:#fff;">
      <div id="{container_id}" style="width:100%; height:100%;"></div>
      <div id="{overlay_id}" style="
        position:absolute; left:8px; top:8px; font-size:12px; color:#333;
        background:rgba(255,255,255,0.85); border:1px solid rgba(0,0,0,0.08);
        padding:6px 8px; border-radius:8px; pointer-events:none;">
        加载中…
      </div>
    </div>
    <!-- reset:{reset_nonce} -->
    <script type="importmap">
      {{
        "imports": {{
          "three": "https://unpkg.com/three@0.161.0/build/three.module.js"
        }}
      }}
    </script>
    <script type="module">
      import * as THREE from "three";
      import {{ OrbitControls }} from "https://unpkg.com/three@0.161.0/examples/jsm/controls/OrbitControls.js";
      import {{ PLYLoader }} from "https://unpkg.com/three@0.161.0/examples/jsm/loaders/PLYLoader.js";

      const container = document.getElementById("{container_id}");
      const overlay = document.getElementById("{overlay_id}");
      function setOverlay(text) {{
        if (overlay) overlay.textContent = text;
      }}
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xffffff);

      const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1e9);
      camera.position.set(4, 3, 6);

      const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
      renderer.setPixelRatio(1);
      renderer.setClearColor(0xffffff, 1);
      container.appendChild(renderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.screenSpacePanning = true;

      scene.add(new THREE.AmbientLight(0xffffff, 1.0));

      const loader = new PLYLoader();
      const plyText = {ply_json};
      const blob = new Blob([plyText], {{ type: "text/plain" }});
      const url = URL.createObjectURL(blob);

      function fitCamera(geometry, material) {{
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        if (!box) return;
        const size = new THREE.Vector3();
        const center = new THREE.Vector3();
        box.getSize(size);
        box.getCenter(center);

        const radius = size.length() * 0.5;
        controls.target.copy(center);

        const fov = THREE.MathUtils.degToRad(camera.fov);
        let dist = radius / Math.tan(fov / 2);
        dist *= {BIG_VIEW_CAMERA_FIT_MARGIN};

        const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
        if (dir.lengthSq() < 1e-12) dir.set(0, 0, 1);
        camera.position.copy(center).addScaledVector(dir, dist);
        camera.near = Math.max(dist / 10000, 0.0001);
        camera.far = Math.max(dist * 1000, camera.near + 1);
        camera.updateProjectionMatrix();

        controls.minDistance = dist * 0.35;
        controls.maxDistance = dist * 4.0;

        const sizeVal = Math.max(radius * 0.002, 0.001) * 3.0;
        material.size = sizeVal;
      }}

      loader.load(url, (geometry) => {{
        URL.revokeObjectURL(url);
        if (geometry.computeVertexNormals) {{
          geometry.computeVertexNormals();
        }}
        const material = new THREE.PointsMaterial({{
          size: 1.0,
          vertexColors: true,
          sizeAttenuation: true,
          transparent: true,
          opacity: 1.0
        }});
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        fitCamera(geometry, material);
        setOverlay("加载完成");
      }}, undefined, (err) => {{
        URL.revokeObjectURL(url);
        console.error(err);
        setOverlay("加载失败，请检查控制台错误");
      }});

      function resize() {{
        const width = container.clientWidth || 300;
        const height = container.clientHeight || {height};
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
      }}

      const observer = new ResizeObserver(resize);
      observer.observe(container);
      resize();

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();
    </script>
    """
    components.html(html, height=height)


def pl_multi_component(
    ply_contents: List[str],
    labels: List[str],
    offsets: Optional[List[List[float]]] = None,
    height: int = BIG_PLY_HEIGHT,
    reset_nonce: int = 0,
) -> None:
    import uuid

    container_id = f"pc_multi_{reset_nonce}_{uuid.uuid4().hex}"
    overlay_id = f"{container_id}_overlay"
    offset_list = offsets if offsets is not None else [[0.0, 0.0, 0.0] for _ in labels]
    items = [
        {"label": label, "content": content, "offset": offset}
        for label, content, offset in zip(labels, ply_contents, offset_list)
    ]
    items_json = json.dumps(items)
    html = f"""
    <div style="width:100%; height:{height}px; position:relative; background:#fff;">
      <div id="{container_id}" style="width:100%; height:100%;"></div>
      <div id="{overlay_id}" style="
        position:absolute; left:8px; top:8px; font-size:12px; color:#333;
        background:rgba(255,255,255,0.85); border:1px solid rgba(0,0,0,0.08);
        padding:6px 8px; border-radius:8px; pointer-events:none;">
        加载中…
      </div>
    </div>
    <!-- reset:{reset_nonce} -->
    <script type="importmap">
      {{
        "imports": {{
          "three": "https://unpkg.com/three@0.161.0/build/three.module.js"
        }}
      }}
    </script>
    <script type="module">
      import * as THREE from "three";
      import {{ OrbitControls }} from "https://unpkg.com/three@0.161.0/examples/jsm/controls/OrbitControls.js";
      import {{ PLYLoader }} from "https://unpkg.com/three@0.161.0/examples/jsm/loaders/PLYLoader.js";

      const container = document.getElementById("{container_id}");
      const overlay = document.getElementById("{overlay_id}");
      function setOverlay(text) {{
        if (overlay) overlay.textContent = text;
      }}

      const items = {items_json};
      if (!items.length) {{
        setOverlay("未选择点云");
      }}

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xffffff);

      const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1e9);
      camera.position.set(0, 0, 5);

      const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
      renderer.setPixelRatio(1);
      renderer.setClearColor(0xffffff, 1);
      container.appendChild(renderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.screenSpacePanning = true;

      scene.add(new THREE.AmbientLight(0xffffff, 1.0));

      const loader = new PLYLoader();
      const group = new THREE.Group();
      scene.add(group);
      const materials = [];
      const box = new THREE.Box3();
      let hasBox = false;
      let pending = 0;
      const maxRenderPoints = {BIG_VIEW_MAX_RENDER_POINTS_PER_CLOUD};

      function downsampleGeometry(geometry, maxPoints) {{
        const posAttr = geometry.getAttribute("position");
        if (!posAttr) return geometry;
        const count = posAttr.count || 0;
        if (!maxPoints || count <= maxPoints) return geometry;

        const step = count / maxPoints;
        const sampledPos = new Float32Array(maxPoints * 3);
        const colorAttr = geometry.getAttribute("color");
        const sampledColor = colorAttr ? new Float32Array(maxPoints * 3) : null;

        for (let i = 0; i < maxPoints; i++) {{
          const src = Math.min(count - 1, Math.floor(i * step));
          sampledPos[i * 3] = posAttr.array[src * 3];
          sampledPos[i * 3 + 1] = posAttr.array[src * 3 + 1];
          sampledPos[i * 3 + 2] = posAttr.array[src * 3 + 2];
          if (sampledColor) {{
            sampledColor[i * 3] = colorAttr.array[src * 3];
            sampledColor[i * 3 + 1] = colorAttr.array[src * 3 + 1];
            sampledColor[i * 3 + 2] = colorAttr.array[src * 3 + 2];
          }}
        }}

        const out = new THREE.BufferGeometry();
        out.setAttribute("position", new THREE.BufferAttribute(sampledPos, 3));
        if (sampledColor) {{
          out.setAttribute("color", new THREE.BufferAttribute(sampledColor, 3));
        }}
        out.computeBoundingBox();
        return out;
      }}

      function fitCamera(bounds) {{
        const size = new THREE.Vector3();
        const center = new THREE.Vector3();
        bounds.getSize(size);
        bounds.getCenter(center);

        const radius = size.length() * 0.5;
        controls.target.copy(center);

        const fov = THREE.MathUtils.degToRad(camera.fov);
        let dist = radius / Math.tan(fov / 2);
        dist *= 1.15;

        const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
        if (dir.lengthSq() < 1e-12) dir.set(0, 0, 1);
        camera.position.copy(center).addScaledVector(dir, dist);
        camera.near = Math.max(dist / 10000, 0.0001);
        camera.far = Math.max(dist * 1000, camera.near + 1);
        camera.updateProjectionMatrix();

        controls.minDistance = dist * 0.85;
        controls.maxDistance = dist * 1.35;

        const sizeVal = Math.max(radius * 0.002, 0.001) * {BIG_VIEW_POINT_SIZE_SCALE};
        materials.forEach((material) => {{
          material.size = sizeVal;
        }});
      }}

      function addItem(item) {{
        pending += 1;
        const blob = new Blob([item.content], {{ type: "text/plain" }});
        const url = URL.createObjectURL(blob);
        loader.load(url, (geometry) => {{
          URL.revokeObjectURL(url);
          geometry = downsampleGeometry(geometry, maxRenderPoints);
          if (geometry.computeVertexNormals) {{
            geometry.computeVertexNormals();
          }}
          const material = new THREE.PointsMaterial({{
            size: 1.0,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 1.0
          }});
          materials.push(material);
          const points = new THREE.Points(geometry, material);
          const offset = item.offset || [0, 0, 0];
          points.position.set(offset[0] || 0, offset[1] || 0, offset[2] || 0);
          points.name = item.label || "";
          group.add(points);

          geometry.computeBoundingBox();
          if (geometry.boundingBox) {{
            const shifted = geometry.boundingBox.clone();
            shifted.translate(points.position);
            if (!hasBox) {{
              box.copy(shifted);
              hasBox = true;
            }} else {{
              box.union(shifted);
            }}
          }}

          pending -= 1;
          setOverlay("已加载 " + group.children.length + "/" + items.length);
          if (pending === 0 && hasBox) {{
            fitCamera(box);
            setOverlay("已加载 " + items.length + " 个点云");
          }}
        }}, undefined, (err) => {{
          URL.revokeObjectURL(url);
          console.error(err);
          pending -= 1;
          setOverlay("加载失败，请检查控制台错误");
        }});
      }}

      if (items.length) {{
        items.forEach(addItem);
      }}

      function resize() {{
        const width = container.clientWidth || 300;
        const height = container.clientHeight || {height};
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
      }}

      const observer = new ResizeObserver(resize);
      observer.observe(container);
      resize();

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();
    </script>
    """
    components.html(html, height=height)


def stable_seed(username: str, mode: str, difficulty: str) -> int:
    key = f"{username}|{mode}|{difficulty}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def init_state() -> None:
    defaults = {
        "logged_in": False,
        "is_admin": False,
        "username": "",
        "mode": MODE_IDS[0] if MODE_IDS else "",
        "difficulty": "easy",
        "question_index": 0,
        "answers": {},
        "exam_generated": False,
        "temp_dir_obj": None,
        "exam_dir": "",
        "exam_meta": [],
        "result_ready": False,
        "result_json": "",
        "result_saved": False,
        "score": 0,
        "seed": None,
        "viewer_reset_nonce": 0,
        "show_big_view": True,
        "big_view_selection": ["grid[0,0]", "grid[0,1]"],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if MODE_IDS and st.session_state.get("mode") not in MODE_IDS:
        st.session_state["mode"] = MODE_IDS[0]
    if st.session_state.get("difficulty") not in DIFFICULTY_IDS:
        st.session_state["difficulty"] = "easy"


def reset_exam_state() -> None:
    temp_dir_obj = st.session_state.get("temp_dir_obj")
    if temp_dir_obj is not None:
        try:
            temp_dir_obj.cleanup()
        except Exception:
            pass
    st.session_state.temp_dir_obj = None
    st.session_state.exam_dir = ""
    st.session_state.exam_meta = []
    st.session_state.answers = {}
    st.session_state.exam_generated = False
    st.session_state.question_index = 0
    st.session_state.result_ready = False
    st.session_state.result_json = ""
    st.session_state.result_saved = False
    st.session_state.score = 0
    st.session_state.seed = None
    st.session_state.viewer_reset_nonce = 0
    st.session_state.show_big_view = True
    st.session_state.big_view_selection = ["grid[0,0]", "grid[0,1]"]
    for label in [
        "grid[0,0]",
        "grid[0,1]",
        "grid[0,2]",
        "grid[1,0]",
        "grid[1,1]",
        "grid[1,2]",
        "grid[2,0]",
        "grid[2,1]",
        "A",
        "B",
        "C",
        "D",
    ]:
        st.session_state.pop(f"big_view_{label}", None)
    for idx in range(TOTAL_QUESTIONS):
        st.session_state.pop(f"answer_{idx}", None)
        st.session_state.pop(f"merge_candidate_choice_{idx}", None)
        for label in ["A", "B", "C", "D"]:
            st.session_state.pop(f"merge_candidate_{idx}_{label}", None)


def generate_exam(username: str, mode: str, difficulty: str) -> None:
    """生成 PCRAR 考试试卷"""
    reset_exam_state()
    temp_dir_obj = tempfile.TemporaryDirectory()
    exam_dir = Path(temp_dir_obj.name)
    st.session_state.temp_dir_obj = temp_dir_obj
    st.session_state.exam_dir = str(exam_dir)
    seed = stable_seed(username, mode, difficulty)
    st.session_state.seed = seed
    
    # 解析规则
    rule_template = canonical_template(parse_mode(mode))
    
    config = PCRARConfig(
        n_points=POINTS_PER_CLOUD,
        num_options=4,
        rule_filter={rule_template},
        matrix_missing_one_per_row=(difficulty != "easy"),
    )
    generator = PCRARDatasetGenerator(config=config, seed=seed)
    try:
        generator.generate_dataset(exam_dir, TOTAL_QUESTIONS)
    except RuntimeError as exc:
        st.session_state.exam_meta = []
        st.session_state.exam_generated = False
        st.error(f"生成题目失败：{exc}")
        return
    
    meta_path = exam_dir / "meta.json"
    exam_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # 校验规则过滤是否生效
    expected_template = canonical_template(rule_template).value
    mismatches = [
        i for i, entry in enumerate(exam_meta)
        if normalize_template_value(entry.get("rule", {}).get("template", expected_template), rule_template) != expected_template
    ]
    if mismatches:
        st.session_state.exam_meta = []
        st.session_state.exam_generated = False
        st.error(
            f"生成的题目规则与所选模式不一致（期望 {expected_template}）。"
            f"不一致题目索引示例: {mismatches[:5]}"
        )
        return
    st.session_state.exam_meta = exam_meta
    st.session_state.exam_generated = True


def load_ply_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_ply_path(exam_root: Path, relative_path: str) -> Path:
    """解析 PLY 文件的相对路径
    
    Args:
        exam_root: 考试数据根目录
        relative_path: 相对路径，如 "sample_000000/in_0.ply"
        
    Returns:
        完整的 Path 对象
    """
    # 直接拼接相对路径
    return exam_root / relative_path


def build_result(
    username: str,
    mode: str,
    difficulty: str,
    meta: List[Dict],
    answers: Dict[int, str],
) -> Dict:
    details = []
    correct_count = 0
    wrong_reasons: List[str] = []
    wrong_type_counts = Counter()
    attribute_exposure_counts = Counter()
    attribute_wrong_counts = Counter()
    perturbation_exposure_counts = Counter()
    perturbation_wrong_counts = Counter()
    for idx, entry in enumerate(meta):
        user_option = answers.get(idx)
        # 新格式使用 gt_label
        gt_option = entry.get("gt_label", "")

        candidate_paths = entry.get("candidate_paths", [])
        labels = [chr(ord("A") + i) for i in range(len(candidate_paths))]
        candidate_notes = entry.get("notes", {}).get("candidate_notes", [])
        candidate_types = entry.get("distractor_types", [])
        cand_reasons = {}
        for i, label in enumerate(labels):
            if i < len(candidate_notes):
                cand_reasons[label] = candidate_notes[i]
            elif i == entry.get("gt_index", 0):
                cand_reasons[label] = "符合规则的正确答案"
            else:
                cand_reasons[label] = "干扰项"

        selected_idx = labels.index(user_option) if user_option in labels else None
        selected_type = (
            candidate_types[selected_idx]
            if selected_idx is not None and selected_idx < len(candidate_types)
            else None
        )
        question_attrs = extract_question_attributes(entry)
        perturbation_type = extract_perturbation_type(entry)
        for attr in question_attrs:
            attribute_exposure_counts[attr] += 1
        if perturbation_type:
            perturbation_exposure_counts[perturbation_type] += 1
        
        is_correct = user_option == gt_option
        if is_correct:
            correct_count += 1
        if user_option is None:
            wrong_reason = "未作答"
        else:
            wrong_reason = cand_reasons.get(user_option, "")
        if not is_correct:
            wrong_reasons.append(wrong_reason or "未知原因")
            if selected_type in ERROR_TYPE_KEYS:
                wrong_type_counts[selected_type] += 1
            for attr in question_attrs:
                attribute_wrong_counts[attr] += 1
            if perturbation_type:
                perturbation_wrong_counts[perturbation_type] += 1
        
        # 获取规则信息
        rule_info = entry.get("rule", {})
        rule_template = entry.get("rule_template", rule_info.get("template", "未知"))
        
        details.append(
            {
                "id": entry.get("id", f"q{idx + 1:02d}"),
                "task_type": entry.get("task_type", ""),
                "rule_template": rule_template,
                "focus": entry.get("focus", ""),
                "gt_option": gt_option,
                "user_option": user_option,
                "is_correct": is_correct,
                "selected_distractor_type": selected_type,
                "question_attributes": [ATTRIBUTE_LABELS.get(a, a) for a in question_attrs],
                "point_cloud_perturbation_type": PERTURBATION_LABELS.get(perturbation_type, perturbation_type),
            }
        )
    total = len(meta)
    accuracy = correct_count / total if total else 0.0
    reason_ratio = {}
    if wrong_reasons:
        counts = Counter(wrong_reasons)
        total_wrong = sum(counts.values())
        reason_ratio = {k: round(v / total_wrong, 4) for k, v in counts.items()}

    typed_wrong_total = sum(wrong_type_counts.get(k, 0) for k in ERROR_TYPE_KEYS)
    error_type_ratio = {
        k: round((wrong_type_counts.get(k, 0) / typed_wrong_total), 4) if typed_wrong_total else 0.0
        for k in ERROR_TYPE_KEYS
    }

    attr_error_rate_raw = {}
    for attr in ATTRIBUTE_ORDER:
        exposed = int(attribute_exposure_counts.get(attr, 0))
        if exposed <= 0:
            continue
        attr_error_rate_raw[attr] = float(attribute_wrong_counts.get(attr, 0)) / float(exposed)
    attr_rate_sum = sum(attr_error_rate_raw.values())
    error_attribute_rate = {
        ATTRIBUTE_LABELS[attr]: round(rate, 4)
        for attr, rate in attr_error_rate_raw.items()
    }
    error_attribute_ratio_normalized = {
        ATTRIBUTE_LABELS[attr]: round((rate / attr_rate_sum), 4) if attr_rate_sum else 0.0
        for attr, rate in attr_error_rate_raw.items()
    }
    attribute_exposure_ratio = {
        ATTRIBUTE_LABELS[attr]: round(float(attribute_exposure_counts[attr]) / float(total), 4) if total else 0.0
        for attr in ATTRIBUTE_ORDER
        if int(attribute_exposure_counts.get(attr, 0)) > 0
    }
    error_perturbation_ratio = {}
    perturbation_exposure_ratio = {}
    for perturb_type in PERTURBATION_TYPE_KEYS:
        label = PERTURBATION_LABELS.get(perturb_type, perturb_type)
        exposed = int(perturbation_exposure_counts.get(perturb_type, 0))
        wrong = int(perturbation_wrong_counts.get(perturb_type, 0))
        error_perturbation_ratio[label] = round(float(wrong) / float(exposed), 4) if exposed else 0.0
        perturbation_exposure_ratio[label] = round(float(exposed) / float(total), 4) if total else 0.0

    return {
        "username": username,
        "mode": mode,
        "difficulty": difficulty,
        "total": total,
        "score": correct_count,
        "accuracy": round(accuracy, 4),
        "error_reason_ratio": reason_ratio,
        "error_type_ratio": error_type_ratio,
        "error_attribute_rate": error_attribute_rate,
        "error_attribute_ratio_normalized": error_attribute_ratio_normalized,
        "attribute_exposure_ratio": attribute_exposure_ratio,
        "error_perturbation_ratio": error_perturbation_ratio,
        "perturbation_exposure_ratio": perturbation_exposure_ratio,
        "questions": details,
    }


def append_record(record: Dict) -> None:
    df = load_records()
    new_row = pd.DataFrame([{key: record.get(key) for key in RECORD_COLUMNS}], columns=RECORD_COLUMNS)
    updated = pd.concat([df, new_row], ignore_index=True)
    updated.to_csv(RECORDS_PATH, index=False)


def safe_slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return cleaned.strip("_") or "user"


def load_records() -> pd.DataFrame:
    if not RECORDS_PATH.exists():
        return pd.DataFrame(columns=RECORD_COLUMNS)
    try:
        df = pd.read_csv(RECORDS_PATH)
    except pd.errors.ParserError:
        df = pd.read_csv(RECORDS_PATH, engine="python", on_bad_lines="skip")
    for col in RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    normalized_df = df[RECORD_COLUMNS].copy()
    if list(df.columns) != RECORD_COLUMNS:
        normalized_df.to_csv(RECORDS_PATH, index=False)
    return normalized_df


def render_admin() -> None:
    render_page_banner(
        "PCRAR 管理后台",
        "查看考试记录、统计正确率，并管理导出数据。",
        "Admin Console",
        chips=["记录管理", "统计分析", "CSV 导出"],
    )
    if st.button("退出登录"):
        reset_exam_state()
        st.session_state.logged_in = False
        st.session_state.is_admin = False
        st.session_state.username = ""
        st.rerun()

    df = load_records()
    if df.empty:
        st.info("暂无考试记录。")
        return

    display_df = df.copy()
    display_df["mode"] = display_df["mode"].apply(mode_display_name)

    st.subheader("考试记录")
    st.dataframe(display_df, use_container_width=True)

    st.subheader("删除考试记录")
    df = df.reset_index(drop=True)
    display_df = display_df.reset_index(drop=True)
    label_to_index = {}
    option_labels = []
    for idx, row in display_df.iterrows():
        label = (
            f"{idx + 1}: {row.get('username', '')} | {row.get('mode', '')} | "
            f"{row.get('score', '')}/{row.get('total', '')} | {row.get('accuracy', '')}"
        )
        option_labels.append(label)
        label_to_index[label] = idx
    selected = st.multiselect("选择要删除的记录", option_labels)
    if st.button("删除选中记录"):
        if not selected:
            st.warning("请先选择要删除的记录。")
        else:
            drop_indices = [label_to_index[label] for label in selected]
            new_df = df.drop(index=drop_indices).reset_index(drop=True)
            new_df.to_csv(RECORDS_PATH, index=False)
            st.success(f"已删除 {len(drop_indices)} 条记录。")
            st.rerun()

    st.subheader("各用户正确率")
    user_acc = df.groupby("username")["accuracy"].mean().sort_values(ascending=False)
    st.bar_chart(user_acc)

    st.subheader("各规则平均正确率")
    mode_acc = display_df.groupby("mode")["accuracy"].mean()
    rule_df = pd.DataFrame(
        {"正确率": [mode_acc.get(name, 0.0) for name in ["递进", "循环", "增减", "守恒", "置换", "对称"]]},
        index=["递进", "循环", "增减", "守恒", "置换", "对称"],
    )
    st.bar_chart(rule_df)

    st.download_button(
        "下载 CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="exam_records.csv",
        mime="text/csv",
    )


def render_exam() -> None:
    answered = len(st.session_state.answers)
    render_page_banner(
        "PCRAR 3D 推理考试",
        "在同一视野内观察九宫格、比对候选点云并完成作答，减少上下滚动。",
        "Point Cloud Matrix Reasoning",
        chips=[
            f"用户：{st.session_state.username}",
            f"模式：{get_mode_description(st.session_state.mode)}",
            f"已作答：{answered}/{TOTAL_QUESTIONS}",
        ],
    )

    st.sidebar.markdown("## 考试设置")
    st.sidebar.markdown(
        f"""
        <div class="section-panel" style="background: rgba(255,255,255,0.10); border-color: rgba(255,255,255,0.12); box-shadow: none;">
          <div class="section-kicker" style="color:#dbeafe;">当前用户</div>
          <div class="section-title" style="color:#ffffff; margin-bottom:0.15rem;">{escape(st.session_state.username)}</div>
          <div class="section-text" style="color:#d8e7f6;">在此切换规则模式、生成试卷和跳转题号。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("选择考试模式")
    current_mode = st.session_state.mode
    mode_idx = MODE_IDS.index(current_mode) if current_mode in MODE_IDS else 0
    new_mode = st.sidebar.selectbox(
        "规则模式",
        MODE_IDS,
        index=mode_idx,
        format_func=get_mode_description,
    )
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        reset_exam_state()
        st.sidebar.info("已切换模式，请重新生成试卷。")

    st.session_state.difficulty = "easy"
    
    st.sidebar.caption(get_mode_description(st.session_state.mode))
    st.sidebar.caption(f"难度：{DIFFICULTY_LABELS.get(st.session_state.difficulty, st.session_state.difficulty)}")

    button_label = "生成试卷" if not st.session_state.exam_generated else "重新生成试卷"
    if st.sidebar.button(button_label):
        with st.spinner("生成题目中..."):
            generate_exam(
                st.session_state.username,
                st.session_state.mode,
                st.session_state.difficulty,
            )
        st.sidebar.success("试卷已生成。")

    if st.sidebar.button("退出登录"):
        reset_exam_state()
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.is_admin = False
        st.rerun()

    if not st.session_state.exam_generated:
        st.markdown(
            """
            <div class="section-panel">
              <div class="section-kicker">开始前</div>
              <div class="section-title">先生成试卷，再进入作答界面</div>
              <div class="section-text">
                当前系统固定为 3x3 九宫格补全题，缺失格位于右下角。你可以先在左侧选择规则模式，再生成整套题目。
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        intro_cols = st.columns(5)
        intro_items = [
            ("题型", "3x3 九宫格矩阵补全"),
            ("缺失位置", "固定第 9 格"),
            ("难度", "easy"),
            ("规则作用", "同一关系贯穿整张矩阵"),
            ("目标", "从 A-D 中选唯一正确答案"),
        ]
        for col, (label, value) in zip(intro_cols, intro_items):
            with col:
                st.markdown(
                    f"""
                    <div class="question-fact">
                      <div class="question-fact-label">{escape(label)}</div>
                      <div class="question-fact-value">{escape(value)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        rule_desc = {
            "递进规则 (Progression)": "属性沿固定步长变化：尺寸/姿态的递进",
            "Distribute-three规则 (Distribute-three)": "固定 3 档分布：密度 / 尺寸 / 形状 / 颜色（同一行/列三格各出现一次）",
            "增减规则 (Count)": "叶节点数量变化：2↔3",
            "守恒规则 (Conservation)": "尺寸守恒：固定3个几何体联动（+1/-1/0）",
            "置换规则 (Permutation)": "位置槽位循环置换",
            "对称规则 (Symmetry)": "对称变换：仅2个几何体；姿态/尺寸均为左+Δ右-Δ",
        }
        st.markdown("### 规则说明")
        rule_cols = st.columns(3)
        for i, (name, desc) in enumerate(rule_desc.items()):
            with rule_cols[i % 3]:
                st.markdown(
                    f"""
                    <div class="section-panel compact">
                      <div class="section-kicker">Rule</div>
                      <div class="section-title">{escape(name)}</div>
                      <div class="section-text">{escape(desc)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    st.sidebar.metric("已作答", f"{answered}/{TOTAL_QUESTIONS}")
    current_q = st.sidebar.number_input(
        "当前题号",
        min_value=1,
        max_value=TOTAL_QUESTIONS,
        value=st.session_state.question_index + 1,
        step=1,
    )
    st.session_state.question_index = int(current_q) - 1

    idx = st.session_state.question_index
    entry = st.session_state.exam_meta[idx]
    exam_root = Path(st.session_state.exam_dir)
    
    # 获取任务类型
    task_type = entry.get("task_type", "matrix_3x3")
    task_name = TASK_TYPE_NAMES.get(task_type, task_type)
    rule_template = entry.get("rule_template", entry.get("rule", {}).get("template", ""))
    k_h = entry.get("k_h")
    k_v = entry.get("k_v")

    st.markdown(
        f"""
        <div class="section-panel">
          <div class="section-kicker">当前题目</div>
          <div class="section-title">题目 {idx + 1}/{TOTAL_QUESTIONS}</div>
          <div class="section-text">保持当前视角进行比对，右侧候选区和上方第九格预览会同步联动。</div>
          <div class="question-facts">
            <div class="question-fact">
              <div class="question-fact-label">题型</div>
              <div class="question-fact-value">{escape(str(task_name))}</div>
            </div>
            <div class="question-fact">
              <div class="question-fact-label">规则</div>
              <div class="question-fact-value">{escape(str(rule_template))}</div>
            </div>
            <div class="question-fact">
              <div class="question-fact-label">难度</div>
              <div class="question-fact-value">{escape(str(DIFFICULTY_LABELS.get(st.session_state.difficulty, st.session_state.difficulty)))}</div>
            </div>
            <div class="question-fact">
              <div class="question-fact-label">k_h</div>
              <div class="question-fact-value">{escape(str(k_h))}</div>
            </div>
            <div class="question-fact">
              <div class="question-fact-label">k_v</div>
              <div class="question-fact-value">{escape(str(k_v))}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_cols = st.columns([1.2, 1.4, 5.4])
    with control_cols[0]:
        if st.button("重置当前题目视角"):
            st.session_state.viewer_reset_nonce += 1
            st.rerun()
    with control_cols[1]:
        st.download_button(
            "下载本题 meta.json",
            data=json.dumps(entry, ensure_ascii=False, indent=2),
            file_name=f"{entry.get('id','question')}_meta.json",
            mime="application/json",
        )
    with control_cols[2]:
        st.markdown(
            """
            <div class="section-panel" style="padding: 0.85rem 1rem; margin-bottom: 0;">
              <div class="section-kicker">操作提示</div>
              <div class="section-text">A/B/C/D 既是答案选项，也是第九格的预览开关；选“不显示”可只看原始题干。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    grid_paths = entry.get("grid_paths", [])
    candidate_paths = entry.get("candidate_paths", [])
    target_pos = tuple(entry.get("target_position", [2, 2]))
    raw_missing = entry.get("missing_positions") or entry.get("empty_grid_positions") or [list(target_pos)]
    missing_positions = set()
    for pos in raw_missing:
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            continue
        missing_positions.add((int(pos[0]), int(pos[1])))
    missing_positions.add((int(target_pos[0]), int(target_pos[1])))
    known_count = 9 - len(missing_positions)
    cand_labels = [chr(ord("A") + i) for i in range(len(candidate_paths))]
    if len(cand_labels) > 4:
        cand_labels = cand_labels[:4]

    st.markdown("### 九宫格作答区")
    reset_nonce = st.session_state.viewer_reset_nonce

    if not cand_labels:
        st.error("当前题目没有候选项。")
        return

    options = cand_labels[:4]
    answer_key = f"answer_{idx}"
    merge_options = ["不显示"] + options
    merge_choice_key = f"merge_candidate_choice_{idx}"
    current_answer = st.session_state.answers.get(idx)
    if current_answer not in options:
        current_answer = None
    current_merge_choice = st.session_state.get(merge_choice_key)
    if current_merge_choice not in merge_options:
        current_merge_choice = current_answer if current_answer in options else "不显示"
        st.session_state[merge_choice_key] = current_merge_choice
    selected_merge_candidate = st.radio(
        "第九格预览 / 直接作答",
        merge_options,
        index=merge_options.index(current_merge_choice),
        key=merge_choice_key,
        horizontal=True,
    )
    if selected_merge_candidate in options:
        st.session_state[answer_key] = selected_merge_candidate
        st.session_state.answers[idx] = selected_merge_candidate
        current_answer = selected_merge_candidate
    elif current_answer in options:
        st.session_state[answer_key] = current_answer
        st.session_state.answers[idx] = current_answer
    else:
        st.session_state.pop(answer_key, None)
        st.session_state.answers.pop(idx, None)
        current_answer = None
    selected_candidate_for_view = selected_merge_candidate if selected_merge_candidate in options else None

    # 构建九宫格合并视图：仅展示已知格，缺失格保持空白
    spacing = BIG_VIEW_GRID_SPACING
    row_depth = BIG_VIEW_ROW_DEPTH
    contents: List[str] = []
    labels: List[str] = []
    offsets: List[List[float]] = []

    for r in range(3):
        for c in range(3):
            if (r, c) in missing_positions:
                continue
            if r < len(grid_paths) and c < len(grid_paths[r]) and grid_paths[r][c]:
                path = grid_paths[r][c]
                contents.append(load_ply_text(resolve_ply_path(exam_root, path)))
                labels.append(f"grid[{r},{c}]")
                offsets.append(
                    [
                        float((c - 1) * spacing),
                        float((1 - r) * spacing),
                        float((1 - r) * row_depth),
                    ]
                )

    if selected_candidate_for_view is not None:
        cand_idx = ord(selected_candidate_for_view) - ord("A")
        if 0 <= cand_idx < len(candidate_paths):
            cand_path = candidate_paths[cand_idx]
            contents.append(load_ply_text(resolve_ply_path(exam_root, cand_path)))
            labels.append(f"cand[{selected_candidate_for_view}]")
            offsets.append(
                [
                    float((int(target_pos[1]) - 1) * spacing),
                    float((1 - int(target_pos[0])) * spacing),
                    float((1 - int(target_pos[0])) * row_depth),
                ]
            )
    workspace_cols = st.columns([3.6, 2.4], gap="large")
    with workspace_cols[0]:
        if selected_candidate_for_view is not None:
            st.caption(
                f"当前已把选项 {selected_candidate_for_view} 放到缺失格位置 {tuple(map(int, target_pos))}。"
            )
        else:
            st.caption("当前只显示题干九宫格；选择 A/B/C/D 会同时预览并记录答案。")
        pl_multi_component(
            contents,
            labels,
            offsets=offsets,
            height=BIG_PLY_HEIGHT,
            reset_nonce=reset_nonce,
        )
    with workspace_cols[1]:
        answer_display = current_answer if current_answer in options else "未作答"
        preview_display = selected_candidate_for_view or "不显示"
        st.markdown(
            f"""
            <div style="
                padding: 14px 16px;
                border-radius: 16px;
                border: 1px solid rgba(30, 41, 59, 0.10);
                background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
                margin-bottom: 12px;
            ">
              <div style="font-size: 12px; color: #475569; margin-bottom: 6px;">当前状态</div>
              <div style="font-size: 24px; font-weight: 700; color: #0f172a;">答案：{answer_display}</div>
              <div style="font-size: 13px; color: #334155; margin-top: 6px;">第九格预览：{preview_display}</div>
              <div style="font-size: 12px; color: #64748b; margin-top: 8px;">
                选择 A/B/C/D 会同时完成预览和作答；选“不显示”只隐藏第九格，不清空已有答案。
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### 候选点云")
        for row_start in range(0, 4, 2):
            candidate_row = st.columns(2, gap="small")
            for offset, col in enumerate(candidate_row):
                cand_idx = row_start + offset
                with col:
                    label = chr(ord("A") + cand_idx)
                    tag = []
                    if label == current_answer:
                        tag.append("当前答案")
                    if label == selected_candidate_for_view:
                        tag.append("正在预览")
                    suffix = f" · {' / '.join(tag)}" if tag else ""
                    st.caption(f"选项 {label}{suffix}")
                    if cand_idx < len(candidate_paths):
                        pl_component(
                            load_ply_text(resolve_ply_path(exam_root, candidate_paths[cand_idx])),
                            height=OPTION_PLY_HEIGHT,
                            reset_nonce=reset_nonce,
                        )
                    else:
                        st.info("空")

    nav_cols = st.columns(3)
    with nav_cols[0]:
        if st.button("上一题"):
            st.session_state.question_index = max(0, idx - 1)
            st.rerun()
    with nav_cols[1]:
        if st.button("下一题"):
            st.session_state.question_index = min(TOTAL_QUESTIONS - 1, idx + 1)
            st.rerun()
    with nav_cols[2]:
        finish = st.button("结束考试")

    if finish and not st.session_state.result_ready:
        result = build_result(
            st.session_state.username,
            st.session_state.mode,
            st.session_state.difficulty,
            st.session_state.exam_meta,
            st.session_state.answers,
        )
        st.session_state.result_json = json.dumps(result, ensure_ascii=False, indent=2)
        st.session_state.result_ready = True
        st.session_state.score = result["score"]
        result_path = Path(st.session_state.exam_dir) / "result.json"
        result_path.write_text(st.session_state.result_json, encoding="utf-8")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{safe_slug(st.session_state.username)}_"
            f"{safe_slug(st.session_state.mode)}_"
            f"{safe_slug(st.session_state.difficulty)}_"
            f"{timestamp}.json"
        )
        persistent_path = RESULTS_DIR / filename
        persistent_path.write_text(st.session_state.result_json, encoding="utf-8")
        if not st.session_state.result_saved:
            reason_payload = {
                "error_type_ratio": result.get("error_type_ratio", {}),
                "error_attribute_rate": result.get("error_attribute_rate", {}),
                "error_attribute_ratio_normalized": result.get("error_attribute_ratio_normalized", {}),
                "attribute_exposure_ratio": result.get("attribute_exposure_ratio", {}),
                "error_perturbation_ratio": result.get("error_perturbation_ratio", {}),
                "perturbation_exposure_ratio": result.get("perturbation_exposure_ratio", {}),
                "error_reason_ratio": result.get("error_reason_ratio", {}),
            }
            record = {
                "username": st.session_state.username,
                "mode": mode_display_name(st.session_state.mode),
                "score": result["score"],
                "total": result["total"],
                "accuracy": result["accuracy"],
                "reason": json.dumps(reason_payload, ensure_ascii=False),
            }
            append_record(record)
            st.session_state.result_saved = True

    if st.session_state.result_ready:
        st.success(
            f"考试结束，得分 {st.session_state.score}/{TOTAL_QUESTIONS}，"
            f"正确率 {st.session_state.score / TOTAL_QUESTIONS:.2%}"
        )
        st.download_button(
            "下载 result.json",
            data=st.session_state.result_json,
            file_name="result.json",
            mime="application/json",
        )


def render_login() -> None:
    render_page_banner(
        "PCRAR 3D 推理考试",
        "基于点云的 3x3 九宫格矩阵补全系统，聚焦观察、比对与规则推理。",
        "Point Cloud Matrix Reasoning",
        chips=["3D 点云", "九宫格推理", "单屏作答"],
    )

    shell_cols = st.columns([1.1, 1, 1.1])
    with shell_cols[1]:
        st.markdown(
            """
            <div class="section-panel">
              <div class="section-kicker">登录</div>
              <div class="section-title">输入姓名或管理员账号</div>
              <div class="section-text">普通用户直接输入姓名/ID 即可开始；管理员账号需要额外密码。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            username = st.text_input("姓名/ID")
            password = st.text_input("管理员密码（仅 admin）", type="password")
            submitted = st.form_submit_button("登录")
    if not submitted:
        return
    if not username.strip():
        st.error("请输入姓名/ID。")
        return
    if username.strip().lower() == "admin":
        if password != "123456":
            st.error("管理员密码错误。")
            return
        st.session_state.logged_in = True
        st.session_state.is_admin = True
        st.session_state.username = "admin"
    else:
        st.session_state.logged_in = True
        st.session_state.is_admin = False
        st.session_state.username = username.strip()
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="PCRAR 3D推理考试",
        layout="wide",
    )
    init_state()
    render_theme_styles()
    if not st.session_state.logged_in:
        render_login()
        return
    if st.session_state.is_admin:
        render_admin()
        return
    render_exam()


if __name__ == "__main__":
    main()
