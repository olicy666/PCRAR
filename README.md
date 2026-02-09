# SPIRAL3D / PCRAR

**Structured Perception to Intelligent Reasoning And Logic in 3D**

本项目支持两种模式：
- **PCRAR 模式**（默认）：基于 CSG 布尔几何体的关系/类比推理数据生成器
- **Legacy 模式**：原有的多物体点云规则生成

## 快速开始

环境：Python 3.10+，依赖仅 `numpy`。

```bash
pip install -r requirements.txt

# PCRAR 模式（默认）
python main.py --output output_pcrar --num-samples 10 --seed 0

# Legacy 模式
python main.py --mode main --output output_legacy --num-samples 3 --seed 0
```

## PCRAR 模式（新）

### 核心概念

PCRAR 将每个点云定义为一个复合实体 E：

```
Attr(E) = (CSG, O)
- CSG：二叉布尔树，叶节点是 Primitive（Sphere/Box/Cylinder/Cone）
- O：全局摆放与观测配置
```

### 题型

1. **Relational（2→1）**：输入 A,B，推断变换 T，使得 B=T(A)，从候选中选 D* = T(B)
2. **Analogical（3→1）**：输入 A,B,C（B=T(A)），从候选中选 D* = T(C)

### 属性轴（7 个离散属性）

| 属性 | 描述 | 离散值 |
|------|------|--------|
| Shape | 基本几何体类型 | Sphere, Box, Cylinder, Cone |
| Boolean Ops | CSG 操作类型 | Union, Diff, Intersect |
| PartCount | 叶节点数量 | 2, 3 |
| Size | 尺寸档位 | S(0.667), M(1.0), L(1.5)（每升一档 ×1.5） |
| Pose | 离散旋转角度 | 0°, 60°, 120°, 180°, 240°, 300° |
| Position | 位置槽位 + delta 档 | slot: -1/0/+1, delta: Near/Mid/Far（单步位移比例 0.25/0.5/0.75，按当前尺寸缩放） |
| Density | 采样权重档位 | 均匀/偏左/偏右 等 |

### 规则库（7 条）

| 规则 | 描述 | 来源对齐 |
|------|------|----------|
| Progression | 属性沿固定步长递进 | R1-1, R1-2, R1-3, R1-4, R1-5 |
| Cycle | 形状离散循环 | R1-6, R3-10 |
| Copy | 尺寸/密度拷贝 | R3 |
| Count | 叶节点数量增减 2↔3 | R1-11, R3-4, R3-5, R4-3 |
| Conservation | 尺寸守恒（一增一减） | R2-2 |
| Permutation | 槽位循环置换 | R3-2, R3-7 |
| Symmetry | 对称变换（位置/姿态/尺寸/密度其一；左+Δ, 右-Δ） | R4-7 |

### 命令行参数

```bash
python main.py --mode pcrar \
    --output output_pcrar \
    --num-samples 10 \
    --points 8192 \
    --task-mix 0.5 \
    --leaf-count-min 2 \
    --leaf-count-max 3 \
    --pcrar-rules Progression,Cycle,Copy \
    --seed 0

# 生成带迷惑性2D视角的数据集（用于对比实验）
python main.py --mode pcrar \
    --output output_with_view \
    --num-samples 10 \
    --generate-confusing-view \
    --seed 0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | pcrar | 模式选择 |
| `--output` | output | 输出目录 |
| `--num-samples` | 3 | 样本数量 |
| `--points` | 8192 | 每个点云的点数 |
| `--task-mix` | 0.5 | Relational 任务比例 |
| `--leaf-count-min` | 2 | 最小叶节点数 |
| `--leaf-count-max` | 3 | 最大叶节点数 |
| `--pcrar-rules` | None | 规则过滤（逗号分隔） |
| `--seed` | None | 随机种子 |
| `--generate-confusing-view` | False | 生成迷惑性2D视角（用于3D vs 2D对比实验） |

### 输出格式

```
output_pcrar/
├── sample_000000/
│   ├── in_0.ply      # 输入 A
│   ├── in_1.ply      # 输入 B
│   ├── in_2.ply      # 输入 C（仅 Analogical）
│   ├── cand_0.ply    # 候选 0
│   ├── cand_1.ply    # 候选 1
│   ├── cand_2.ply    # 候选 2
│   ├── cand_3.ply    # 候选 3
│   └── meta.json     # 样本元数据
├── sample_000001/
│   └── ...
└── meta.json         # 汇总元数据
```

### meta.json Schema

```json
{
  "id": "sample_000000",
  "task_type": "relational" | "analogical",
  "input_paths": ["sample_000000/in_0.ply", ...],
  "candidate_paths": ["sample_000000/cand_0.ply", ...],
  "gt_index": 0,
  "gt_label": "A",
  "n_points": 8192,
  "rule": {
    "template": "Progression",
    "source_align": ["R1-1", ...],
    "params": {...}
  },
  "entities": {
    "inputs": [Attr(E)_json, ...],
    "candidates": [Attr(E)_json, ...]
  },
  "notes": {
    "distractors": ["错误原因1", ...]
  }
}
```

### CSG 实体 JSON 格式

```json
{
  "csg": {
    "type": "op",
    "op": "union",
    "left": {
      "type": "leaf",
      "id": 0,
      "prim": "sphere",
      "size": "M",
      "local_pose_deg": [0, 0, 0],
      "slot": -1,
      "delta_level": "Mid"
    },
    "right": {...}
  },
  "obs": {
    "global_pose_deg": [0, 0, 0],
    "global_translation": [0, 0, 0],
    "sampling_mode": "surface",
    "n_points": 8192,
    "part_sampling_weights": [0.5, 0.5],
    "density_preset_idx": 0
  },
  "expr": "Union(Sphere,Box)"
}
```

## Legacy 模式

### 模式与参数

- `--mode` 可选：
  - 主集合：`main`
  - 大类：`r1-only`, `r2-only`, `r3-only`, `r4-only`
  - 消融：`all-minus-r1`, `all-minus-r2`, `all-minus-r3`, `all-minus-r4`
- `--rules` 自定义规则列表（逗号分隔，如 `R1-1,R2-7,R3-2`，会覆盖 `--mode`）

```bash
python main.py --mode main --num-samples 10 --points 4096 --seed 0
python main.py --rules R1-1,R2-7,R3-2 --num-samples 5 --points 4096
```

### 输出格式

```
output/sample_000000/
    1.ply  # A
    2.ply  # B
    3.ply  # 候选 A
    4.ply  # 候选 B
    5.ply  # 候选 C
    6.ply  # 候选 D
    meta.json
output/meta.json  # 所有题目的 meta 列表
```

### 规则体系

- **R1 属性解耦推理**：单属性变化（R1-1 到 R1-11）
- **R2 几何交互推理**：属性间交互（R2-1 到 R2-16）
- **R3 结构组合推理**：子图模式（R3-1 到 R3-11）
- **R4 因果动力推理**：动态演化（R4-1 到 R4-10）

## 新旧差异对比

| 特性 | PCRAR（新） | Legacy（旧） |
|------|-------------|--------------|
| 点云定义 | 单个 CSG 布尔几何体 | 多个独立 primitive 拼接 |
| 题型 | Relational (2→1) / Analogical (3→1) | 固定 2→1 |
| 规则数 | 7 条 | 50+ 条 |
| 点数默认 | 8192 | 4096 |
| 属性离散化 | 严格离散 | 连续值 |
| 布尔操作 | 支持 Union/Diff/Intersect | 不支持 |

## 自测

```bash
# PCRAR 模式测试
python main.py --mode pcrar --num-samples 10 --task-mix 0.5 --output output_pcrar --seed 0

# 检查生成结果
ls output_pcrar/
cat output_pcrar/meta.json | head -50
```

## 迷惑性视角生成（新功能）🎯

### 功能说明

为了证明 **3D点云相比2D视角在空间推理任务中的必要性**，我们新增了"迷惑性视角"生成功能。该功能会为每个样本生成一个精心选择的2D视角渲染，该视角会：

- **遮挡关键属性**：使规则的关键变化在2D投影中不可见或模糊
- **制造视觉歧义**：让多个候选项在2D视角下难以区分
- **模拟信息损失**：展示2D投影导致的空间信息损失

### 快速使用

```bash
# 生成带迷惑性视角的数据集
python main.py --mode pcrar \
    --output output_comparison \
    --num-samples 100 \
    --generate-confusing-view \
    --seed 0

# 输出结构
output_comparison/sample_000000/
├── in_0.ply           # 3D点云
├── view_in_0.png      # 2D迷惑性视角
├── cand_*.ply         # 3D候选
├── view_cand_*.png    # 2D候选视角
└── meta.json          # 包含视角配置信息
```

### 迷惑性策略示例

| 规则类型 | 关键属性 | 迷惑性视角 | 迷惑原因 |
|---------|---------|-----------|---------|
| Progression(尺寸r) | 整体尺寸变化 | 正面远距离 | 深度压缩，大小变化不明显 |
| Progression(旋转R) | 绕轴旋转 | 沿旋转轴 | 旋转完全不可见 |
| Count | 部件数量 | 重叠视角 | 多部件投影重叠，数量难判 |
| Symmetry | 对称变化 | 非对称轴 | 对称性不可见 |

### 实验设计建议

**对比实验**：在同一数据集上分别测试：
1. **通用视觉模型**（GPT-4V, Claude等）使用2D迷惑性视角
2. **3D点云模型**使用完整点云

**预期结果**：2D视角准确率显著低于3D点云，从而证明3D数据的必要性。

详细文档：[CONFUSING_VIEW_GUIDE.md](CONFUSING_VIEW_GUIDE.md)

---

## 依赖

- Python 3.10+
- numpy
- (可选) Pillow - 用于保存2D视角图像
- (可选) streamlit, pandas - 用于 Web UI

```bash
pip install -r requirements.txt
pip install Pillow  # 如需生成2D视角
```

## 项目结构

```
raven3d/
├── __init__.py
├── csg.py              # CSG 数据结构（新）
├── pcrar_entity.py     # PCRAR 实体与采样（新）
├── pcrar_rules.py      # 7 条 PCRAR 规则（新）
├── pcrar_dataset.py    # PCRAR 数据集生成器（新）
├── dataset.py          # Legacy 数据集生成器
├── factory.py          # Registry 工厂
├── geometry.py         # 几何体定义
├── io.py               # 文件 I/O
├── registry.py         # 规则注册表
├── scene.py            # 场景定义
└── rules/              # Legacy 规则
    ├── __init__.py
    ├── base.py
    ├── simple.py
    ├── medium.py
    ├── complex.py
    ├── groups.py
    └── utils.py
```
# PCRAR
