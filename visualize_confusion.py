#!/usr/bin/env python3
"""可视化迷惑性视角效果

该脚本读取生成的数据集，展示：
1. 2D迷惑性视角图像
2. 视角配置信息
3. 迷惑性原因说明
4. 统计分析
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def load_dataset_meta(output_dir: Path) -> List[Dict[str, Any]]:
    """加载数据集元数据"""
    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        print(f"错误：找不到 {meta_path}")
        return []
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data if isinstance(data, list) else [data]


def analyze_confusion_levels(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析迷惑性级别分布"""
    stats = {
        'total': len(samples),
        'with_confusing_view': 0,
        'confusion_levels': defaultdict(int),
        'confusion_by_rule': defaultdict(lambda: defaultdict(int)),
        'confusion_reasons': defaultdict(int),
    }
    
    for sample in samples:
        confusing_view = sample.get('confusing_view')
        if confusing_view:
            stats['with_confusing_view'] += 1
            
            view_config = confusing_view.get('view_config', {})
            level = view_config.get('confusion_level', 'unknown')
            reason = view_config.get('confusion_reason', 'unknown')
            
            stats['confusion_levels'][level] += 1
            stats['confusion_reasons'][reason] += 1
            
            # 按规则类型统计
            rule = sample.get('rule', {})
            template = rule.get('template', 'unknown')
            stats['confusion_by_rule'][template][level] += 1
    
    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """打印统计信息"""
    print("=" * 70)
    print("数据集统计")
    print("=" * 70)
    print(f"总样本数: {stats['total']}")
    print(f"含迷惑性视角: {stats['with_confusing_view']}")
    
    if stats['with_confusing_view'] == 0:
        print("\n警告：没有样本包含迷惑性视角！")
        print("请使用 --generate-confusing-view 参数生成数据集。")
        return
    
    print("\n" + "-" * 70)
    print("迷惑性级别分布")
    print("-" * 70)
    for level in ['high', 'medium', 'low', 'unknown']:
        count = stats['confusion_levels'].get(level, 0)
        if count > 0:
            pct = count / stats['with_confusing_view'] * 100
            print(f"  {level.upper():8s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n" + "-" * 70)
    print("各规则类型的迷惑性级别")
    print("-" * 70)
    for template, levels in sorted(stats['confusion_by_rule'].items()):
        print(f"\n  {template}:")
        for level in ['high', 'medium', 'low']:
            count = levels.get(level, 0)
            if count > 0:
                print(f"    {level}: {count}")
    
    print("\n" + "-" * 70)
    print("迷惑性原因（Top 5）")
    print("-" * 70)
    sorted_reasons = sorted(
        stats['confusion_reasons'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (reason, count) in enumerate(sorted_reasons[:5], 1):
        print(f"{i}. [{count}次] {reason}")


def show_sample_details(sample: Dict[str, Any], sample_dir: Path) -> None:
    """显示单个样本的详细信息"""
    print("\n" + "=" * 70)
    print(f"样本详情: {sample['id']}")
    print("=" * 70)
    
    print(f"任务类型: {sample.get('task_type', 'unknown')}")
    print(f"考点: {sample.get('focus', 'N/A')[:100]}...")
    print(f"正确答案: {sample.get('gt_label', 'N/A')} (索引 {sample.get('gt_index', -1)})")
    
    rule = sample.get('rule', {})
    print(f"\n规则: {rule.get('template', 'unknown')}")
    
    confusing_view = sample.get('confusing_view')
    if confusing_view:
        view_config = confusing_view.get('view_config', {})
        print(f"\n迷惑性视角配置:")
        print(f"  相机位置: {view_config.get('camera_position', 'N/A')}")
        print(f"  视场角 (FOV): {view_config.get('fov', 'N/A')}°")
        print(f"  迷惑性级别: {view_config.get('confusion_level', 'N/A').upper()}")
        print(f"  迷惑原因: {view_config.get('confusion_reason', 'N/A')}")
        
        rendered_paths = confusing_view.get('rendered_paths', [])
        print(f"\n已渲染视角图像 ({len(rendered_paths)} 张):")
        for path in rendered_paths:
            full_path = sample_dir / path
            exists = "✓" if full_path.exists() else "✗"
            print(f"    {exists} {path}")
    else:
        print("\n该样本未包含迷惑性视角。")


def interactive_browser(output_dir: Path, samples: List[Dict[str, Any]]) -> None:
    """交互式浏览器"""
    if not samples:
        return
    
    samples_with_view = [s for s in samples if 'confusing_view' in s]
    if not samples_with_view:
        print("\n没有样本包含迷惑性视角！")
        return
    
    print("\n" + "=" * 70)
    print("交互式浏览")
    print("=" * 70)
    print("命令: [数字] 查看样本详情 | [s] 显示统计 | [q] 退出")
    
    while True:
        try:
            cmd = input(f"\n输入命令 (样本 0-{len(samples_with_view)-1}): ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 's':
                stats = analyze_confusion_levels(samples)
                print_statistics(stats)
            elif cmd.isdigit():
                idx = int(cmd)
                if 0 <= idx < len(samples_with_view):
                    sample = samples_with_view[idx]
                    sample_dir = output_dir / sample['id']
                    show_sample_details(sample, sample_dir)
                else:
                    print(f"错误：索引超出范围 (0-{len(samples_with_view)-1})")
            else:
                print("未知命令，请重试。")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python visualize_confusion.py <output_dir>")
        print("\n示例:")
        print("  python visualize_confusion.py output_with_view")
        print("  python visualize_confusion.py test_output_progression_r")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"错误：目录不存在 {output_dir}")
        sys.exit(1)
    
    print(f"正在加载数据集: {output_dir}")
    samples = load_dataset_meta(output_dir)
    
    if not samples:
        print("错误：无法加载数据集元数据")
        sys.exit(1)
    
    print(f"成功加载 {len(samples)} 个样本")
    
    # 显示统计
    stats = analyze_confusion_levels(samples)
    print_statistics(stats)
    
    # 交互式浏览
    if stats['with_confusing_view'] > 0:
        try:
            interactive_browser(output_dir, samples)
        except KeyboardInterrupt:
            pass
    
    print("\n完成！")


if __name__ == "__main__":
    main()

