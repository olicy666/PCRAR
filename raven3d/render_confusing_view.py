"""生成具有迷惑性的2D视角图像模块

该模块根据题目规则类型，智能选择最具迷惑性的视角，
用于对比实验：证明3D点云相比2D视角在空间推理任务中的优势。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .pcrar_entity import PCRAREntity
from .pcrar_rules import RuleTemplate, RuleParams
from .csg import PrimType


class ConfusingViewGenerator:
    """迷惑性视角生成器
    
    根据题目规则类型，选择最具迷惑性的视角，使得：
    1. 关键属性变化在2D投影中不可见或模糊
    2. 多个物体在视觉上重叠或混淆
    3. 形状特征在投影后产生歧义
    """
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
    
    def select_confusing_viewpoint(
        self,
        rule_template: RuleTemplate,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        """根据规则类型选择迷惑性视角
        
        Args:
            rule_template: 规则模板
            params: 规则参数
            entities: 实体列表（输入 + 候选）
            
        Returns:
            视角配置字典：
            {
                'camera_position': (x, y, z),
                'camera_target': (x, y, z),
                'camera_up': (x, y, z),
                'fov': float,
                'confusion_reason': str,  # 迷惑性原因说明
                'confusion_level': str,   # 'high', 'medium', 'low'
            }
        """
        if rule_template == RuleTemplate.PROGRESSION:
            return self._confusing_view_progression(params, entities)
        elif rule_template == RuleTemplate.CYCLE:
            return self._confusing_view_cycle(params, entities)
        elif rule_template == RuleTemplate.COUNT:
            return self._confusing_view_count(params, entities)
        elif rule_template == RuleTemplate.CONSERVATION:
            return self._confusing_view_conservation(params, entities)
        elif rule_template == RuleTemplate.PERMUTATION:
            return self._confusing_view_permutation(params, entities)
        elif rule_template == RuleTemplate.SYMMETRY:
            return self._confusing_view_symmetry(params, entities)
        elif rule_template == RuleTemplate.COPY:
            return self._confusing_view_copy(params, entities)
        else:
            return self._default_confusing_view()
    
    def _confusing_view_progression(
        self,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        """Progression 规则的迷惑性视角"""
        axis = params.axis
        
        if axis == "r":  # 尺寸变化
            # 策略：正面视角 + 远距离，深度信息丢失，大小变化不明显
            return {
                'camera_position': (0.0, 0.0, 8.0),
                'camera_target': (0.0, 0.0, 0.0),
                'camera_up': (0.0, 1.0, 0.0),
                'fov': 30.0,  # 小视场角，正交投影效果
                'confusion_reason': '正面视角+远距离，深度压缩导致尺寸变化不明显',
                'confusion_level': 'high',
            }
        
        elif axis == "R":  # 旋转变化
            # 策略：沿旋转轴观察，旋转不可见
            rot_axis = params.rot_axis or 'z'
            if rot_axis.lower() == 'x':
                cam_pos = (8.0, 0.0, 0.0)
                confusion = '沿X轴观察，绕X轴旋转不可见'
            elif rot_axis.lower() == 'y':
                cam_pos = (0.0, 8.0, 0.0)
                confusion = '沿Y轴观察，绕Y轴旋转不可见'
            else:  # z
                cam_pos = (0.0, 0.0, 8.0)
                confusion = '沿Z轴观察，绕Z轴旋转不可见'
            
            return {
                'camera_position': cam_pos,
                'camera_target': (0.0, 0.0, 0.0),
                'camera_up': (0.0, 1.0, 0.0),
                'fov': 30.0,
                'confusion_reason': confusion,
                'confusion_level': 'high',
            }
        
        elif axis == "p":  # 位置变化
            # 策略：俯视角度，前后位移信息丢失
            return {
                'camera_position': (0.0, 8.0, 0.5),
                'camera_target': (0.0, 0.0, 0.0),
                'camera_up': (0.0, 0.0, -1.0),
                'fov': 40.0,
                'confusion_reason': '俯视角度，前后位移投影为小范围移动，不明显',
                'confusion_level': 'high',
            }
        
        elif axis == "d":  # 密度变化
            # 策略：远距离视角，点云密度差异不可见
            return {
                'camera_position': (5.0, 3.0, 5.0),
                'camera_target': (0.0, 0.0, 0.0),
                'camera_up': (0.0, 1.0, 0.0),
                'fov': 35.0,
                'confusion_reason': '2D图像无法体现点云密度差异',
                'confusion_level': 'high',
            }
        
        return self._default_confusing_view()
    
    def _confusing_view_cycle(
        self,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        """Cycle 规则的迷惑性视角（形状循环）"""
        # 策略：选择让形状差异最不明显的视角
        # 例如：让 Cylinder 侧面看起来像 Box
        return {
            'camera_position': (6.0, 2.0, 0.1),
            'camera_target': (0.0, 0.0, 0.0),
            'camera_up': (0.0, 1.0, 0.0),
            'fov': 40.0,
            'confusion_reason': '侧视角度：Cylinder看起来像Box，Cone看起来像三角形',
            'confusion_level': 'medium',
        }
    
    def _confusing_view_count(
        self,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        """Count 规则的迷惑性视角（部件数量变化）"""
        # 策略：让多个部件在投影中重叠，数量不清晰
        return {
            'camera_position': (0.1, 0.5, 8.0),
            'camera_target': (0.0, 0.0, 0.0),
            'camera_up': (0.0, 1.0, 0.0),
            'fov': 35.0,
            'confusion_reason': '正面视角使多个部件投影重叠，数量难以准确判断',
            'confusion_level': 'high',
        }
    
    def _confusing_view_conservation(
        self,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        """Conservation 规则的迷惑性视角（尺寸守恒）"""
        # 策略：深度压缩，一增一减的守恒关系不明显
        return {
            'camera_position': (0.0, 1.0, 8.0),
            'camera_target': (0.0, 0.0, 0.0),
            'camera_up': (0.0, 1.0, 0.0),
            'fov': 30.0,
            'confusion_reason': '正面视角，深度信息丢失，尺寸守恒关系不可见',
            'confusion_level': 'high',
        }
    
    def _confusing_view_permutation(
        self,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        """Permutation 规则的迷惑性视角（位置置换）"""
        # 策略：俯视或正视，位置循环变化不明显
        return {
            'camera_position': (0.2, 7.0, 1.0),
            'camera_target': (0.0, 0.0, 0.0),
            'camera_up': (0.0, 0.0, -1.0),
            'fov': 45.0,
            'confusion_reason': '俯视角度，位置置换在2D投影中不明显',
            'confusion_level': 'medium',
        }
    
    def _confusing_view_symmetry(
        self,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        """Symmetry 规则的迷惑性视角（对称变换）"""
        axis = params.axis
        
        if axis in ("p", "r"):  # 位置或尺寸对称
            # 策略：非对称轴视角，对称性不可见
            return {
                'camera_position': (7.0, 1.0, 1.0),
                'camera_target': (0.0, 0.0, 0.0),
                'camera_up': (0.0, 1.0, 0.0),
                'fov': 40.0,
                'confusion_reason': '侧视角度，左右对称变化在投影中不明显',
                'confusion_level': 'high',
            }
        elif axis == "R":  # 姿态对称
            return {
                'camera_position': (0.0, 0.5, 8.0),
                'camera_target': (0.0, 0.0, 0.0),
                'camera_up': (0.0, 1.0, 0.0),
                'fov': 35.0,
                'confusion_reason': '正面视角，姿态对称旋转不可见',
                'confusion_level': 'high',
            }
        else:  # 密度
            return {
                'camera_position': (5.0, 3.0, 5.0),
                'camera_target': (0.0, 0.0, 0.0),
                'camera_up': (0.0, 1.0, 0.0),
                'fov': 35.0,
                'confusion_reason': '2D图像无法体现密度对称变化',
                'confusion_level': 'high',
            }
    
    def _confusing_view_copy(
        self,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        """Copy 规则的迷惑性视角（循环拷贝）"""
        axis = params.axis
        
        if axis == "copy_density_cycle":
            # 密度循环：2D图像无法体现
            return {
                'camera_position': (5.0, 3.0, 5.0),
                'camera_target': (0.0, 0.0, 0.0),
                'camera_up': (0.0, 1.0, 0.0),
                'fov': 35.0,
                'confusion_reason': '2D图像无法体现点云密度的循环拷贝模式',
                'confusion_level': 'high',
            }
        else:
            # 尺寸或形状循环：选择让差异不明显的视角
            return {
                'camera_position': (0.0, 2.0, 7.0),
                'camera_target': (0.0, 0.0, 0.0),
                'camera_up': (0.0, 1.0, 0.0),
                'fov': 40.0,
                'confusion_reason': '正面视角，循环拷贝的空间排列模式不明显',
                'confusion_level': 'medium',
            }
    
    def _default_confusing_view(self) -> Dict[str, Any]:
        """默认迷惑性视角"""
        # 通用策略：选择一个部分遮挡的倾斜视角
        angle = self.rng.uniform(0, 2 * np.pi)
        elevation = self.rng.uniform(10, 30)  # 度
        
        r = 7.0
        x = r * np.cos(angle) * np.cos(np.radians(elevation))
        y = r * np.sin(np.radians(elevation))
        z = r * np.sin(angle) * np.cos(np.radians(elevation))
        
        return {
            'camera_position': (float(x), float(y), float(z)),
            'camera_target': (0.0, 0.0, 0.0),
            'camera_up': (0.0, 1.0, 0.0),
            'fov': 40.0,
            'confusion_reason': '倾斜视角，部分空间关系模糊',
            'confusion_level': 'medium',
        }
    
    def render_point_cloud_image(
        self,
        points: np.ndarray,
        view_config: Dict[str, Any],
        image_size: Tuple[int, int] = (512, 512),
        point_size: float = 2.0,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """渲染点云为2D图像（简单投影版本）
        
        Args:
            points: (N, 3) 点云数组
            view_config: 视角配置
            image_size: 图像尺寸 (width, height)
            point_size: 点的渲染大小
            background_color: 背景颜色 RGB (0-1)
            
        Returns:
            (height, width, 3) RGB 图像数组 (0-255)
        """
        # 相机坐标系变换
        cam_pos = np.array(view_config['camera_position'])
        cam_target = np.array(view_config['camera_target'])
        cam_up = np.array(view_config['camera_up'])
        
        # 计算相机坐标系
        forward = cam_target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        
        right = np.cross(forward, cam_up)
        right = right / (np.linalg.norm(right) + 1e-12)
        
        up = np.cross(right, forward)
        
        # 将点云变换到相机坐标系
        points_cam = points - cam_pos
        points_cam = np.stack([
            np.dot(points_cam, right),
            np.dot(points_cam, up),
            np.dot(points_cam, forward),
        ], axis=1)
        
        # 透视投影
        fov_rad = np.radians(view_config['fov'])
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        # 过滤掉相机后面的点
        mask = points_cam[:, 2] > 0.1
        points_cam = points_cam[mask]
        
        if len(points_cam) == 0:
            # 没有可见点，返回空白图像
            img = np.full((image_size[1], image_size[0], 3), 
                         np.array(background_color) * 255, dtype=np.uint8)
            return img
        
        # 投影到屏幕空间 [-1, 1]
        x_proj = points_cam[:, 0] * f / points_cam[:, 2]
        y_proj = points_cam[:, 1] * f / points_cam[:, 2]
        
        # 转换到像素坐标
        x_pixel = ((x_proj + 1.0) * 0.5 * image_size[0]).astype(int)
        y_pixel = ((1.0 - y_proj) * 0.5 * image_size[1]).astype(int)
        
        # 深度值（用于遮挡）
        depth = points_cam[:, 2]
        
        # 创建图像和深度缓冲
        img = np.full((image_size[1], image_size[0], 3), 
                     np.array(background_color) * 255, dtype=np.uint8)
        depth_buffer = np.full((image_size[1], image_size[0]), np.inf)
        
        # 按深度排序（远到近）
        sort_indices = np.argsort(-depth)
        
        # 渲染点
        point_color = np.array([0.2, 0.4, 0.8]) * 255  # 蓝色点
        radius = int(point_size)
        
        for idx in sort_indices:
            px, py = x_pixel[idx], y_pixel[idx]
            d = depth[idx]
            
            if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                # 绘制点（简单圆形）
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx * dx + dy * dy <= radius * radius:
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < image_size[0] and 0 <= ny < image_size[1]:
                                if d < depth_buffer[ny, nx]:
                                    img[ny, nx] = point_color
                                    depth_buffer[ny, nx] = d
        
        return img
    
    def save_rendered_image(
        self,
        img: np.ndarray,
        output_path: Path,
    ) -> None:
        """保存渲染图像
        
        Args:
            img: (H, W, 3) RGB 图像数组
            output_path: 输出文件路径
        """
        try:
            from PIL import Image
            Image.fromarray(img).save(output_path)
        except ImportError:
            # 如果没有 PIL，保存为 numpy 格式
            np.save(output_path.with_suffix('.npy'), img)


def generate_confusing_view_for_sample(
    rule_template: RuleTemplate,
    params: RuleParams,
    entities: List[PCRAREntity],
    input_point_clouds: List[np.ndarray],
    candidate_point_clouds: List[np.ndarray],
    output_dir: Path,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """为单个样本生成迷惑性视角渲染
    
    Args:
        rule_template: 规则模板
        params: 规则参数
        entities: 实体列表
        input_point_clouds: 输入点云列表
        candidate_point_clouds: 候选点云列表
        output_dir: 输出目录
        rng: 随机数生成器
        
    Returns:
        视角配置和渲染结果元数据
    """
    generator = ConfusingViewGenerator(rng)
    
    # 选择迷惑性视角
    view_config = generator.select_confusing_viewpoint(
        rule_template, params, entities
    )
    
    # 渲染所有点云
    rendered_paths = []
    
    # 渲染输入点云
    for i, points in enumerate(input_point_clouds):
        img = generator.render_point_cloud_image(points, view_config)
        img_path = output_dir / f"view_in_{i}.png"
        generator.save_rendered_image(img, img_path)
        rendered_paths.append(f"view_in_{i}.png")
    
    # 渲染候选点云
    for i, points in enumerate(candidate_point_clouds):
        img = generator.render_point_cloud_image(points, view_config)
        img_path = output_dir / f"view_cand_{i}.png"
        generator.save_rendered_image(img, img_path)
        rendered_paths.append(f"view_cand_{i}.png")
    
    return {
        'view_config': view_config,
        'rendered_paths': rendered_paths,
        'confusion_reason': view_config['confusion_reason'],
        'confusion_level': view_config['confusion_level'],
    }

