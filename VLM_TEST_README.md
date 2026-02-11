# VLM测试功能说明

## 新增功能

### 1. 组合图像生成

现在生成数据集时，会自动创建 `view_combined.png`，包含：
- **输入图像**：标注为 ref1, ref2 [, ref3]
- **候选图像**：标注为 A, B, C, D
- 所有图像排列在一张图上，方便VLM测试

### 2. 通用Prompt模板

文件：`vlm_prompt_template.txt`

支持关系推理（2→1）和类比推理（3→1）两种任务类型。

### 3. VLM测试脚本

文件：`test_vlm.py`

提供批量测试框架，支持：
- 自动加载数据集
- 调用VLM API
- 解析回答
- 计算准确率（总体、按规则、按迷惑性级别）

## 快速使用

### 生成数据集

```bash
python main.py --mode pcrar \
    --output vlm_test_data \
    --num-samples 20 \
    --generate-confusing-view \
    --seed 0
```

输出文件包含：
```
vlm_test_data/sample_000000/
├── view_combined.png    # ← 新增：组合图像
├── view_in_*.png        # 单独的输入视角
├── view_cand_*.png      # 单独的候选视角
└── meta.json
```

### 查看组合图像

```bash
# 打开第一个样本的组合图
xdg-open vlm_test_data/sample_000000/view_combined.png
```

### 测试VLM

**步骤1**：修改 `test_vlm.py` 中的 `example_vlm_function` 函数，接入你的VLM API：

```python
def example_vlm_function(image_path: str, prompt: str) -> str:
    # GPT-4V示例
    import openai
    import base64
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }}
            ]
        }]
    )
    return response.choices[0].message.content
```

**步骤2**：运行测试

```bash
python test_vlm.py vlm_test_data
```

输出示例：
```
Testing sample 1/20: sample_000000
  GT: A | Predicted: C | ✗
Testing sample 2/20: sample_000001
  GT: B | Predicted: B | ✓
...
============================================================
Overall Accuracy: 35.00% (7/20)

By Rule:
  Progression         : 30.00% (3/10)
  Count               : 40.00% (2/5)
  Symmetry            : 40.00% (2/5)

By Confusion Level:
  high      : 25.00% (3/12)
  medium    : 50.00% (4/8)
============================================================
```

## Prompt模板

### 关系推理（2→1）
```
观察 ref1 和 ref2，识别变换规则 T，使得 ref2 = T(ref1)。
然后从 A, B, C, D 中选择符合 T(ref2) 的答案。
```

### 类比推理（3→1）
```
观察 ref1, ref2, ref3，识别变换规则 T，使得 ref2 = T(ref1)。
然后从 A, B, C, D 中选择符合 T(ref3) 的答案。
```

完整prompt见 `vlm_prompt_template.txt`。

## 支持的VLM

测试脚本支持任何VLM，只需实现：
```python
def your_vlm_function(image_path: str, prompt: str) -> str:
    # 调用你的VLM API
    # 返回文本回答
    pass
```

常见VLM示例：
- **GPT-4V**: OpenAI API
- **Claude 3**: Anthropic API
- **Gemini**: Google API
- **Qwen-VL**: 通过API或本地推理

## 自定义Prompt

编辑 `vlm_prompt_template.txt`，修改：
- 任务说明
- 推理引导
- 回答格式

然后重新运行 `test_vlm.py` 即可使用新prompt。

