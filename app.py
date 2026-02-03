from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from raven3d.pcrar_dataset import PCRARDatasetGenerator, PCRARConfig
from raven3d.pcrar_rules import RuleTemplate

RECORDS_PATH = Path("exam_records.csv")
RESULTS_DIR = Path("results")
TOTAL_QUESTIONS = 20
PLY_HEIGHT = 320
BIG_PLY_HEIGHT = 560
POINTS_PER_CLOUD = 8192

# 规则名称映射
RULE_NAMES = {
    RuleTemplate.PROGRESSION: "递进规则",
    RuleTemplate.CYCLE: "循环规则",
    RuleTemplate.TOGGLE: "切换规则",
    RuleTemplate.COUNT: "增减规则",
    RuleTemplate.CONSERVATION: "守恒规则",
    RuleTemplate.PERMUTATION: "置换规则",
    RuleTemplate.SYMMETRY: "对称规则",
}

# 任务类型
TASK_TYPES = ["relational", "analogical"]
TASK_TYPE_NAMES = {
    "relational": "关系推理",
    "analogical": "类比推理",
}

# 生成规则 ID 列表：任务类型-规则编号
def generate_mode_list() -> List[str]:
    """生成模式列表：关系推理-规则1, 关系推理-规则2, ... 类比推理-规则1, ..."""
    modes = []
    for task_type in TASK_TYPES:
        task_name = TASK_TYPE_NAMES[task_type]
        for idx, template in enumerate(RuleTemplate, 1):
            mode_id = f"{task_name}-规则{idx}"
            modes.append(mode_id)
    return modes

# 模式 ID 到 (task_type, rule_template) 的映射
def parse_mode(mode: str) -> tuple[str, RuleTemplate]:
    """解析模式 ID，返回 (task_type, rule_template)"""
    if mode.startswith("关系推理"):
        task_type = "relational"
        rule_num = int(mode.split("规则")[1])
    else:
        task_type = "analogical"
        rule_num = int(mode.split("规则")[1])
    template = list(RuleTemplate)[rule_num - 1]
    return task_type, template

def get_mode_description(mode: str) -> str:
    """获取模式描述"""
    task_type, template = parse_mode(mode)
    task_name = TASK_TYPE_NAMES[task_type]
    rule_name = RULE_NAMES[template]
    return f"{task_name} - {rule_name} ({template.value})"

MODE_IDS = generate_mode_list()
RECORD_COLUMNS = ["username", "mode", "score", "total", "accuracy", "reason", "result_path"]


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
        dist *= 1.15;

        const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
        if (dir.lengthSq() < 1e-12) dir.set(0, 0, 1);
        camera.position.copy(center).addScaledVector(dir, dist);
        camera.near = Math.max(dist / 10000, 0.0001);
        camera.far = Math.max(dist * 1000, camera.near + 1);
        camera.updateProjectionMatrix();

        controls.minDistance = dist * 0.85;
        controls.maxDistance = dist * 1.35;

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
    height: int = BIG_PLY_HEIGHT,
    reset_nonce: int = 0,
) -> None:
    import uuid

    container_id = f"pc_multi_{reset_nonce}_{uuid.uuid4().hex}"
    overlay_id = f"{container_id}_overlay"
    items = [{"label": label, "content": content} for label, content in zip(labels, ply_contents)]
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

        const sizeVal = Math.max(radius * 0.002, 0.001) * 3.0;
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
          points.name = item.label || "";
          group.add(points);

          geometry.computeBoundingBox();
          if (geometry.boundingBox) {{
            if (!hasBox) {{
              box.copy(geometry.boundingBox);
              hasBox = true;
            }} else {{
              box.union(geometry.boundingBox);
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


def stable_seed(username: str, mode: str) -> int:
    key = f"{username}|{mode}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def init_state() -> None:
    defaults = {
        "logged_in": False,
        "is_admin": False,
        "username": "",
        "mode": MODE_IDS[0] if MODE_IDS else "",
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
        "big_view_selection": ["输入A", "输入B"],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if MODE_IDS and st.session_state.get("mode") not in MODE_IDS:
        st.session_state["mode"] = MODE_IDS[0]


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
    st.session_state.big_view_selection = ["输入A", "输入B"]
    for label in ["输入A", "输入B", "输入C", "A", "B", "C", "D"]:
        st.session_state.pop(f"big_view_{label}", None)
    for idx in range(TOTAL_QUESTIONS):
        st.session_state.pop(f"answer_{idx}", None)


def generate_exam(username: str, mode: str) -> None:
    """生成 PCRAR 考试试卷"""
    reset_exam_state()
    temp_dir_obj = tempfile.TemporaryDirectory()
    exam_dir = Path(temp_dir_obj.name)
    st.session_state.temp_dir_obj = temp_dir_obj
    st.session_state.exam_dir = str(exam_dir)
    seed = stable_seed(username, mode)
    st.session_state.seed = seed
    
    # 解析模式
    task_type, rule_template = parse_mode(mode)
    
    # 设置任务类型比例
    task_mix = 1.0 if task_type == "relational" else 0.0
    
    config = PCRARConfig(
        n_points=POINTS_PER_CLOUD,
        task_mix=task_mix,
        rule_filter={rule_template},
    )
    generator = PCRARDatasetGenerator(config=config, seed=seed)
    generator.generate_dataset(exam_dir, TOTAL_QUESTIONS)
    
    meta_path = exam_dir / "meta.json"
    st.session_state.exam_meta = json.loads(meta_path.read_text(encoding="utf-8"))
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
    username: str, mode: str, meta: List[Dict], answers: Dict[int, str]
) -> Dict:
    details = []
    correct_count = 0
    wrong_reasons: List[str] = []
    for idx, entry in enumerate(meta):
        user_option = answers.get(idx)
        # 新格式使用 gt_label
        gt_option = entry.get("gt_label", "")
        
        # 干扰项原因
        distractors = entry.get("notes", {}).get("distractors", [])
        cand_reasons = {}
        labels = ["A", "B", "C", "D"]
        gt_idx = entry.get("gt_index", 0)
        d_idx = 0
        for i, label in enumerate(labels):
            if i == gt_idx:
                cand_reasons[label] = "符合规则的正确答案"
            else:
                cand_reasons[label] = distractors[d_idx] if d_idx < len(distractors) else "干扰项"
                d_idx += 1
        
        is_correct = user_option == gt_option
        if is_correct:
            correct_count += 1
        if user_option is None:
            wrong_reason = "未作答"
        else:
            wrong_reason = cand_reasons.get(user_option, "")
        if not is_correct:
            wrong_reasons.append(wrong_reason or "未知原因")
        
        # 获取规则信息
        rule_info = entry.get("rule", {})
        rule_template = rule_info.get("template", "未知")
        
        details.append(
            {
                "id": entry.get("id", f"q{idx + 1:02d}"),
                "task_type": entry.get("task_type", ""),
                "rule_template": rule_template,
                "focus": entry.get("focus", ""),
                "gt_option": gt_option,
                "user_option": user_option,
                "is_correct": is_correct,
            }
        )
    total = len(meta)
    accuracy = correct_count / total if total else 0.0
    reason_ratio = {}
    if wrong_reasons:
        counts = Counter(wrong_reasons)
        total_wrong = sum(counts.values())
        reason_ratio = {k: round(v / total_wrong, 4) for k, v in counts.items()}
    return {
        "username": username,
        "mode": mode,
        "total": total,
        "score": correct_count,
        "accuracy": round(accuracy, 4),
        "error_reason_ratio": reason_ratio,
        "questions": details,
    }


def append_record(record: Dict) -> None:
    file_exists = RECORDS_PATH.exists()
    with RECORDS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


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
    return df[RECORD_COLUMNS]


def render_admin() -> None:
    st.title("PCRAR 3D推理考试 管理后台")
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

    st.subheader("考试记录")
    st.dataframe(df, use_container_width=True)

    st.subheader("删除考试记录")
    df = df.reset_index(drop=True)
    label_to_index = {}
    option_labels = []
    for idx, row in df.iterrows():
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

    st.subheader("下载答题结果")
    df = df.reset_index(drop=True)
    downloadable = df[df["result_path"].notna() & df["result_path"].astype(str).str.len() > 0]
    if downloadable.empty:
        st.info("暂无可下载的 result.json。")
    else:
        label_to_path = {}
        options = []
        for idx, row in downloadable.iterrows():
            label = (
                f"{idx + 1}: {row.get('username', '')} | {row.get('mode', '')} | "
                f"{row.get('score', '')}/{row.get('total', '')}"
            )
            options.append(label)
            label_to_path[label] = Path(str(row.get("result_path", "")))
        selected_label = st.selectbox("选择记录", options)
        selected_path = label_to_path.get(selected_label)
        if selected_path and selected_path.exists():
            st.download_button(
                "下载 result.json",
                data=selected_path.read_bytes(),
                file_name=selected_path.name,
                mime="application/json",
            )
        else:
            st.warning("该记录的 result.json 文件不存在。")

    st.subheader("各规则平均正确率")
    mode_acc = df.groupby("mode")["accuracy"].mean()
    # 分组显示
    relational_modes = [m for m in MODE_IDS if m.startswith("关系推理")]
    analogical_modes = [m for m in MODE_IDS if m.startswith("类比推理")]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**关系推理**")
        rel_df = pd.DataFrame(
            {"正确率": [mode_acc.get(m, 0.0) for m in relational_modes]},
            index=[m.split("-")[1] for m in relational_modes],
        )
        st.bar_chart(rel_df)
    with col2:
        st.write("**类比推理**")
        ana_df = pd.DataFrame(
            {"正确率": [mode_acc.get(m, 0.0) for m in analogical_modes]},
            index=[m.split("-")[1] for m in analogical_modes],
        )
        st.bar_chart(ana_df)

    st.download_button(
        "下载 CSV",
        data=RECORDS_PATH.read_bytes(),
        file_name="exam_records.csv",
        mime="text/csv",
    )


def render_exam() -> None:
    st.title("PCRAR 3D推理考试")
    st.markdown(
        """
        <style>
        div[data-testid="stCheckbox"] label > div:first-child {
            transform: scale(1.35);
            transform-origin: left center;
        }
        div[data-testid="stCheckbox"] label {
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("考试设置")
    st.sidebar.write(f"用户：{st.session_state.username}")
    
    # 分组显示模式选择
    st.sidebar.subheader("选择考试模式")
    
    # 任务类型选择
    task_options = list(TASK_TYPE_NAMES.values())
    current_mode = st.session_state.mode
    current_task_name = current_mode.split("-")[0]
    task_idx = task_options.index(current_task_name) if current_task_name in task_options else 0
    
    selected_task = st.sidebar.radio("题目类型", task_options, index=task_idx)
    
    # 规则选择
    rule_labels = []
    for idx, template in enumerate(RuleTemplate, 1):
        rule_labels.append(f"规则{idx}: {RULE_NAMES[template]}")
    
    current_rule_num = int(current_mode.split("规则")[1]) if "规则" in current_mode else 1
    selected_rule_idx = st.sidebar.selectbox(
        "规则类型",
        range(len(rule_labels)),
        index=current_rule_num - 1,
        format_func=lambda i: rule_labels[i],
    )
    
    new_mode = f"{selected_task}-规则{selected_rule_idx + 1}"
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        reset_exam_state()
        st.sidebar.info("已切换模式，请重新生成试卷。")
    
    # 显示当前模式描述
    st.sidebar.caption(get_mode_description(st.session_state.mode))

    button_label = "生成试卷" if not st.session_state.exam_generated else "重新生成试卷"
    if st.sidebar.button(button_label):
        with st.spinner("生成题目中..."):
            generate_exam(st.session_state.username, st.session_state.mode)
        st.sidebar.success("试卷已生成。")

    if st.sidebar.button("退出登录"):
        reset_exam_state()
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.is_admin = False
        st.rerun()

    if not st.session_state.exam_generated:
        st.info("请先在侧边栏选择考试模式并生成试卷。")
        
        # 显示模式说明
        st.subheader("考试模式说明")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 关系推理 (Relational)")
            st.markdown("""
            - **输入**: 2个点云 (A, B)
            - **任务**: 推断 A→B 的变换规则 T，从候选中选出 D* = T(B)
            - **考察**: 认知属性/关系规律
            """)
        with col2:
            st.markdown("### 类比推理 (Analogical)")
            st.markdown("""
            - **输入**: 3个点云 (A, B, C)
            - **任务**: 已知 B=T(A)，从候选中选出 D* = T(C)
            - **考察**: 将规律迁移到新几何体
            """)
        
        st.subheader("规则说明")
        rule_desc = {
            "递进规则 (Progression)": "属性沿固定步长变化：尺寸/姿态/位置/密度的递进",
            "循环规则 (Cycle)": "形状离散循环：球体→立方体→圆柱→圆锥→...",
            "切换规则 (Toggle)": "布尔操作切换：Union ↔ Diff",
            "增减规则 (Count)": "叶节点数量变化：2↔3",
            "守恒规则 (Conservation)": "尺寸守恒：一增一减，总和不变",
            "置换规则 (Permutation)": "位置槽位循环置换",
            "对称规则 (Symmetry)": "对称变换：左+Δ，右-Δ",
        }
        for name, desc in rule_desc.items():
            st.markdown(f"- **{name}**: {desc}")
        
        return

    answered = len(st.session_state.answers)
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
    task_type = entry.get("task_type", "relational")
    task_name = TASK_TYPE_NAMES.get(task_type, task_type)
    rule_template = entry.get("rule", {}).get("template", "")
    
    st.subheader(f"题目 {idx + 1}/{TOTAL_QUESTIONS}")
    st.caption(f"题型: {task_name} | 规则: {rule_template}")
    st.download_button(
        "下载本题 meta.json",
        data=json.dumps(entry, ensure_ascii=False, indent=2),
        file_name=f"{entry.get('id','question')}_meta.json",
        mime="application/json",
    )
    
    control_cols = st.columns([1, 1, 6])
    with control_cols[0]:
        if st.button("重置当前题目视角"):
            st.session_state.viewer_reset_nonce += 1
            st.rerun()
    with control_cols[1]:
        show_big_view = st.checkbox("大视图", key="show_big_view")

    # 获取输入路径
    input_paths = entry.get("input_paths", [])
    n_inputs = len(input_paths)
    
    # 显示输入点云
    st.markdown("### 输入点云")
    reset_nonce = st.session_state.viewer_reset_nonce
    
    if n_inputs == 2:
        # Relational: 2个输入
        ref_cols = st.columns(2)
        with ref_cols[0]:
            st.caption("输入 A")
            pl_component(
                load_ply_text(resolve_ply_path(exam_root, input_paths[0])),
                reset_nonce=reset_nonce,
            )
        with ref_cols[1]:
            st.caption("输入 B")
            pl_component(
                load_ply_text(resolve_ply_path(exam_root, input_paths[1])),
                reset_nonce=reset_nonce,
            )
    else:
        # Analogical: 3个输入
        ref_cols = st.columns(3)
        input_labels = ["A", "B", "C"]
        for i, col in enumerate(ref_cols):
            if i < len(input_paths):
                with col:
                    st.caption(f"输入 {input_labels[i]}")
                    pl_component(
                        load_ply_text(resolve_ply_path(exam_root, input_paths[i])),
                        reset_nonce=reset_nonce,
                    )

    if show_big_view:
        st.markdown("### 合并点云视图")
        if n_inputs == 2:
            view_options = ["输入A", "输入B", "A", "B", "C", "D"]
        else:
            view_options = ["输入A", "输入B", "输入C", "A", "B", "C", "D"]
        
        selected = []
        cols = st.columns(len(view_options))
        for col, label in zip(cols, view_options):
            key = f"big_view_{label}"
            default = label in st.session_state.big_view_selection
            with col:
                checked = st.checkbox(label, key=key, value=default)
            if checked:
                selected.append(label)
        st.session_state.big_view_selection = selected
        
        # 构建路径映射
        candidate_paths = entry.get("candidate_paths", [])
        label_to_path = {}
        for i, path in enumerate(input_paths):
            label_to_path[f"输入{chr(65+i)}"] = path
        for i, path in enumerate(candidate_paths):
            label_to_path[chr(65+i)] = path
        
        if not selected:
            st.info("请选择要显示的点云。")
        else:
            contents = []
            for label in selected:
                path = label_to_path.get(label, "")
                if path:
                    contents.append(load_ply_text(resolve_ply_path(exam_root, path)))
            if contents:
                pl_multi_component(contents, selected, reset_nonce=reset_nonce)

    # 显示候选点云
    st.markdown("### 候选答案")
    candidate_paths = entry.get("candidate_paths", [])
    cand_cols = st.columns(4)
    cand_labels = ["A", "B", "C", "D"]
    
    for i, (col, label) in enumerate(zip(cand_cols, cand_labels)):
        if i < len(candidate_paths):
            with col:
                st.caption(f"选项 {label}")
                pl_component(
                    load_ply_text(resolve_ply_path(exam_root, candidate_paths[i])),
                    reset_nonce=reset_nonce,
                )

    options = ["未作答", "A", "B", "C", "D"]
    current_answer = st.session_state.answers.get(idx, "未作答")
    answer_key = f"answer_{idx}"
    if answer_key not in st.session_state:
        st.session_state[answer_key] = current_answer
    choice = st.radio(
        "选择答案",
        options,
        index=options.index(st.session_state[answer_key]),
        key=answer_key,
        horizontal=True,
    )
    if choice == "未作答":
        st.session_state.answers.pop(idx, None)
    else:
        st.session_state.answers[idx] = choice

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
        filename = f"{safe_slug(st.session_state.username)}_{safe_slug(st.session_state.mode)}_{timestamp}.json"
        persistent_path = RESULTS_DIR / filename
        persistent_path.write_text(st.session_state.result_json, encoding="utf-8")
        if not st.session_state.result_saved:
            record = {
                "username": st.session_state.username,
                "mode": st.session_state.mode,
                "score": result["score"],
                "total": result["total"],
                "accuracy": result["accuracy"],
                "reason": json.dumps(result["error_reason_ratio"], ensure_ascii=False),
                "result_path": str(persistent_path),
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
    st.title("PCRAR 3D推理考试 登录")
    st.markdown("""
    **PCRAR**: Point Cloud Relational and Analogical Reasoning
    
    基于 CSG 布尔几何体的 3D 点云推理考试系统
    """)
    
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
    if not st.session_state.logged_in:
        render_login()
        return
    if st.session_state.is_admin:
        render_admin()
        return
    render_exam()


if __name__ == "__main__":
    main()
