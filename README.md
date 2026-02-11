# PCRAR

**Point Cloud Matrix Reasoning**

本项目默认生成 **3x3 九宫格矩阵推理题（RAVEN-like）**。旧的 relational/analogical 题型保留为 `legacy` 可选路径，不参与默认生成。

## 快速开始

环境：Python 3.10+，依赖仅 `numpy`。

```bash
pip install -r requirements.txt

# 默认：PCRAR matrix 题型
python main.py --mode pcrar --output output_matrix --num-samples 10 --seed 0

# 可选：PCRAR legacy（旧路径）
python main.py --mode pcrar-legacy --output output_legacy --num-samples 10 --seed 0
```

## Matrix 题型定义（默认）

- 单题固定一个规则实例 `T = (RuleTemplate + RuleParams)`。
- 3x3 按指数公式生成：
  - `E[r,c] = T^(r*k_v + c*k_h)(E[0,0])`
- 目标格固定挖空 `(2,2)`；默认每行再采样一个缺失位（总计 3 个缺失格）。
- 候选为多选一（默认 4 选 1，可配）。

默认步长：
- `k_h` 从 `{1,2}` 采样
- `k_v` 从 `{1,2}` 采样
- `K_max = 2*k_h + 2*k_v`

默认离散档位：
- `matrix_size_levels=7`
- `matrix_density_levels=5`
- `matrix_delta_levels=5`
- `matrix_slot_levels=[-1,0,1]`

## 命令行参数（PCRAR matrix）

```bash
python main.py --mode pcrar \
  --output output_matrix \
  --num-samples 100 \
  --num-options 4 \
  --k-h-choices 1,2 \
  --k-v-choices 1,2 \
  --matrix-size-levels 7 \
  --matrix-density-levels 5 \
  --matrix-delta-levels 5 \
  --matrix-slot-levels -1,0,1 \
  --matrix-missing-one-per-row \
  --pcrar-rules Progression,Cycle,Copy \
  --seed 0
```

默认会同时生成规则感知的 2D 渲染：
- `view_grid_r_c.png`
- `view_cand_i.png`
- `view_combined.png`（九宫格上下文 + 候选合成图）

可关闭：

```bash
python main.py --mode pcrar --no-generate-confusing-view --output output_matrix --num-samples 10

# 若要回退到仅挖空目标格 (2,2)
python main.py --mode pcrar --no-matrix-missing-one-per-row --output output_matrix --num-samples 10
```

## 输出格式（新标准）

```
output_matrix/
├── sample_000000/
│   ├── grid_0_0.ply
│   ├── grid_0_1.ply
│   ├── ...
│   ├── grid_2_1.ply
│   ├── cand_0.ply
│   ├── cand_1.ply
│   ├── cand_2.ply
│   ├── cand_3.ply
│   └── meta.json
└── meta.json
```

`meta.json`（每题）关键字段：

```json
{
  "task_type": "matrix_3x3",
  "grid_paths": [["...", null, "..."], [null, "...", "..."], ["...", "...", null]],
  "target_position": [2, 2],
  "missing_positions": [[0, 1], [1, 0], [2, 2]],
  "grid_observation_mask": [[true, false, true], [false, true, true], [true, true, false]],
  "rule_template": "Progression",
  "rule_params": {"template": "Progression", "axis": "r", "direction": 1},
  "k_h": 2,
  "k_v": 1,
  "K_max": 6,
  "matrix_level_config": {
    "size_levels": ["XS", "S", "SM", "M", "ML", "L", "XL"],
    "delta_levels": ["VeryNear", "Near", "Mid", "Far", "VeryFar"],
    "density_levels": [10240, 9216, 8192, 7168, 6144],
    "slot_levels": [-1, 0, 1]
  },
  "candidate_paths": ["...", "...", "...", "..."],
  "gt_index": 1,
  "distractor_types": ["irrelevant", "gt", "analogical_wrong_relation", "perceptual_plausible"]
}
```

## Legacy 路径

- `--mode pcrar-legacy`：启用旧 PCRAR relational/analogical 逻辑。
- 旧主项目模式（`main`, `r1-only` 等）仍可通过 `--mode` 使用。

## 自测

```bash
python main.py --mode pcrar --output quick_test --num-samples 10 --seed 0
python -m tests.test_matrix_smoke --dataset quick_test
```
