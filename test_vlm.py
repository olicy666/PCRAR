#!/usr/bin/env python3
"""VLM测试脚本 - 使用组合图像测试视觉语言模型"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_prompt_template() -> str:
    """加载prompt模板"""
    template_path = Path(__file__).parent / "vlm_prompt_template.txt"
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_prompt(sample: Dict[str, Any]) -> str:
    """为单个样本生成prompt"""
    template = load_prompt_template()
    
    task_type_map = {
        'relational': 'Relational Reasoning (2→1)',
        'analogical': 'Analogical Reasoning (3→1)',
        'matrix_3x3': 'Matrix Reasoning (3x3, missing bottom-right)'
    }
    
    task_type = task_type_map.get(sample.get('task_type', 'relational'), 'Unknown')
    
    return template.format(task_type=task_type)


def test_vlm_single_sample(
    sample_dir: Path,
    sample_meta: Dict[str, Any],
    vlm_function,  # 用户提供的VLM调用函数
) -> Dict[str, Any]:
    """测试单个样本
    
    Args:
        sample_dir: 样本目录
        sample_meta: 样本元数据
        vlm_function: VLM调用函数，签名为 vlm_function(image_path: str, prompt: str) -> str
        
    Returns:
        测试结果
    """
    # 获取组合图像路径
    confusing_view = sample_meta.get('confusing_view', {})
    combined_image = confusing_view.get('combined_image', 'view_combined.png')
    image_path = sample_dir / combined_image
    
    if not image_path.exists():
        return {
            'sample_id': sample_meta['id'],
            'error': 'Combined image not found',
            'gt_label': sample_meta.get('gt_label'),
        }
    
    # 生成prompt
    prompt = generate_prompt(sample_meta)
    
    # 调用VLM
    try:
        response = vlm_function(str(image_path), prompt)
        
        # 解析回答（简单提取A/B/C/D）
        predicted_label = None
        for char in response.upper():
            if char in ['A', 'B', 'C', 'D']:
                predicted_label = char
                break
        
        # 判断正确性
        gt_label = sample_meta.get('gt_label', 'A')
        correct = (predicted_label == gt_label)
        
        return {
            'sample_id': sample_meta['id'],
            'task_type': sample_meta.get('task_type'),
            'rule': sample_meta.get('rule', {}).get('template'),
            'confusion_level': confusing_view.get('view_config', {}).get('confusion_level'),
            'gt_label': gt_label,
            'predicted_label': predicted_label,
            'correct': correct,
            'response': response,
        }
    
    except Exception as e:
        return {
            'sample_id': sample_meta['id'],
            'error': str(e),
            'gt_label': sample_meta.get('gt_label'),
        }


def test_vlm_dataset(
    dataset_dir: Path,
    vlm_function,
    output_file: str = 'vlm_results.json',
) -> Dict[str, Any]:
    """测试整个数据集
    
    Args:
        dataset_dir: 数据集目录
        vlm_function: VLM调用函数
        output_file: 结果输出文件
        
    Returns:
        汇总结果
    """
    # 加载数据集元数据
    meta_path = dataset_dir / 'meta.json'
    with open(meta_path, 'r', encoding='utf-8') as f:
        dataset_meta = json.load(f)
    
    if not isinstance(dataset_meta, list):
        dataset_meta = [dataset_meta]
    
    # 测试每个样本
    results = []
    for i, sample_meta in enumerate(dataset_meta):
        print(f"Testing sample {i+1}/{len(dataset_meta)}: {sample_meta['id']}")
        
        sample_dir = dataset_dir / sample_meta['id']
        result = test_vlm_single_sample(sample_dir, sample_meta, vlm_function)
        results.append(result)
        
        print(f"  GT: {result.get('gt_label')} | Predicted: {result.get('predicted_label')} | "
              f"{'✓' if result.get('correct') else '✗'}")
    
    # 计算统计
    correct_count = sum(1 for r in results if r.get('correct'))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # 按规则类型统计
    by_rule = {}
    for r in results:
        rule = r.get('rule', 'unknown')
        if rule not in by_rule:
            by_rule[rule] = {'correct': 0, 'total': 0}
        by_rule[rule]['total'] += 1
        if r.get('correct'):
            by_rule[rule]['correct'] += 1
    
    for rule in by_rule:
        by_rule[rule]['accuracy'] = by_rule[rule]['correct'] / by_rule[rule]['total']
    
    # 按迷惑性级别统计
    by_confusion = {}
    for r in results:
        level = r.get('confusion_level', 'unknown')
        if level not in by_confusion:
            by_confusion[level] = {'correct': 0, 'total': 0}
        by_confusion[level]['total'] += 1
        if r.get('correct'):
            by_confusion[level]['correct'] += 1
    
    for level in by_confusion:
        by_confusion[level]['accuracy'] = by_confusion[level]['correct'] / by_confusion[level]['total']
    
    summary = {
        'dataset': str(dataset_dir),
        'total_samples': total_count,
        'correct': correct_count,
        'accuracy': accuracy,
        'by_rule': by_rule,
        'by_confusion_level': by_confusion,
        'detailed_results': results,
    }
    
    # 保存结果
    output_path = dataset_dir / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    print(f"\nBy Rule:")
    for rule, stats in sorted(by_rule.items()):
        print(f"  {rule:20s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print(f"\nBy Confusion Level:")
    for level, stats in sorted(by_confusion.items()):
        print(f"  {level:10s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")
    
    return summary


def example_vlm_function(image_path: str, prompt: str) -> str:
    """示例VLM函数（需要替换为实际的API调用）
    
    这里展示如何调用VLM API。你需要根据实际使用的模型替换此函数。
    """
    # 示例1：使用OpenAI GPT-4V
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-4-vision-preview",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt},
    #                 {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
    #             ]
    #         }
    #     ]
    # )
    # return response.choices[0].message.content
    
    # 示例2：使用Claude
    # import anthropic
    # client = anthropic.Anthropic()
    # with open(image_path, "rb") as f:
    #     image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    # response = client.messages.create(
    #     model="claude-3-opus-20240229",
    #     messages=[{
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
    #             {"type": "text", "text": prompt}
    #         ]
    #     }]
    # )
    # return response.content[0].text
    
    # 占位符：随机回答
    import random
    return f"Answer: {random.choice(['A', 'B', 'C', 'D'])}\nReasoning: Random guess for testing."


def main():
    """主函数 - 展示如何使用"""
    print("VLM测试工具")
    print("="*60)
    print("\n使用方法：")
    print("1. 修改 example_vlm_function 函数，接入你的VLM API")
    print("2. 运行: python test_vlm.py <dataset_dir>")
    print("\n示例：")
    print("  python test_vlm.py quick_test")
    print("\n支持的VLM：GPT-4V, Claude Sonnet, Gemini, 等")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n错误：请提供数据集目录")
        print("用法: python test_vlm.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    if not dataset_dir.exists():
        print(f"错误：目录不存在 {dataset_dir}")
        sys.exit(1)
    
    # 测试数据集
    print(f"\n开始测试数据集: {dataset_dir}")
    print("注意：当前使用随机回答作为示例，请修改 example_vlm_function 接入真实VLM API\n")
    
    test_vlm_dataset(dataset_dir, example_vlm_function)


if __name__ == "__main__":
    main()
