"""Confusing-view renderer for PCRAR.

Keeps the legacy camera-selection and projection rules, and extends output
layout to matrix-3x3 tasks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .pcrar_entity import PCRAREntity, color_rgb
from .pcrar_rules import RuleParams, RuleTemplate


class ConfusingViewGenerator:
    """Rule-aware confusing viewpoint generator."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def select_confusing_viewpoint(
        self,
        rule_template: RuleTemplate,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        del rule_template, params, entities
        return self._random_three_viewpoint()

    def _random_three_viewpoint(self) -> Dict[str, Any]:
        yaw_deg = float(self.rng.choice(np.array([-45.0, 0.0, 45.0], dtype=float)))
        yaw_rad = np.radians(yaw_deg)
        radius = 7.0
        cam_pos = (
            float(radius * np.sin(yaw_rad)),
            0.0,
            float(radius * np.cos(yaw_rad)),
        )
        label_map = {-45.0: "left", 0.0: "center", 45.0: "right"}
        label = label_map.get(yaw_deg, "center")
        return {
            "camera_position": cam_pos,
            "camera_target": (0.0, 0.0, 0.0),
            "camera_up": (0.0, 1.0, 0.0),
            "fov": 35.0,
            "view_label": label,
            "yaw_degrees": yaw_deg,
            "confusion_reason": f"random three-view selection: {label} ({yaw_deg:+.0f} deg)",
            "confusion_level": "medium",
        }

    def _confusing_view_progression(
        self,
        params: RuleParams,
        entities: List[PCRAREntity],
    ) -> Dict[str, Any]:
        del entities
        axis = params.axis
        if axis == "r":
            return {
                "camera_position": (0.0, 0.0, 8.0),
                "camera_target": (0.0, 0.0, 0.0),
                "camera_up": (0.0, 1.0, 0.0),
                "fov": 30.0,
                "confusion_reason": "正面视角+远距离，深度压缩导致尺寸变化不明显",
                "confusion_level": "high",
            }
        if axis == "R":
            rot_axis = params.rot_axis or "z"
            if rot_axis.lower() == "x":
                cam_pos = (8.0, 0.0, 0.0)
                reason = "沿X轴观察，绕X轴旋转不可见"
            elif rot_axis.lower() == "y":
                cam_pos = (0.0, 8.0, 0.0)
                reason = "沿Y轴观察，绕Y轴旋转不可见"
            else:
                cam_pos = (0.0, 0.0, 8.0)
                reason = "沿Z轴观察，绕Z轴旋转不可见"
            return {
                "camera_position": cam_pos,
                "camera_target": (0.0, 0.0, 0.0),
                "camera_up": (0.0, 1.0, 0.0),
                "fov": 30.0,
                "confusion_reason": reason,
                "confusion_level": "high",
            }
        if axis == "p":
            return {
                "camera_position": (0.0, 8.0, 0.5),
                "camera_target": (0.0, 0.0, 0.0),
                "camera_up": (0.0, 0.0, -1.0),
                "fov": 40.0,
                "confusion_reason": "俯视角度，前后位移投影为小范围移动，不明显",
                "confusion_level": "high",
            }
        if axis == "d":
            return {
                "camera_position": (5.0, 3.0, 5.0),
                "camera_target": (0.0, 0.0, 0.0),
                "camera_up": (0.0, 1.0, 0.0),
                "fov": 35.0,
                "confusion_reason": "2D图像无法体现点云密度差异",
                "confusion_level": "high",
            }
        return self._default_confusing_view()

    def _confusing_view_cycle(self, params: RuleParams, entities: List[PCRAREntity]) -> Dict[str, Any]:
        del params, entities
        return {
            "camera_position": (6.0, 2.0, 0.1),
            "camera_target": (0.0, 0.0, 0.0),
            "camera_up": (0.0, 1.0, 0.0),
            "fov": 40.0,
            "confusion_reason": "侧视角度：Cylinder看起来像Box，Cone看起来像三角形",
            "confusion_level": "medium",
        }

    def _confusing_view_count(self, params: RuleParams, entities: List[PCRAREntity]) -> Dict[str, Any]:
        del params, entities
        return {
            "camera_position": (0.1, 0.5, 8.0),
            "camera_target": (0.0, 0.0, 0.0),
            "camera_up": (0.0, 1.0, 0.0),
            "fov": 35.0,
            "confusion_reason": "正面视角使多个部件投影重叠，数量难以准确判断",
            "confusion_level": "high",
        }

    def _confusing_view_conservation(self, params: RuleParams, entities: List[PCRAREntity]) -> Dict[str, Any]:
        del params, entities
        return {
            "camera_position": (0.0, 1.0, 8.0),
            "camera_target": (0.0, 0.0, 0.0),
            "camera_up": (0.0, 1.0, 0.0),
            "fov": 30.0,
            "confusion_reason": "正面视角，深度信息丢失，尺寸守恒关系不可见",
            "confusion_level": "high",
        }

    def _confusing_view_permutation(self, params: RuleParams, entities: List[PCRAREntity]) -> Dict[str, Any]:
        del params, entities
        return {
            "camera_position": (0.2, 7.0, 1.0),
            "camera_target": (0.0, 0.0, 0.0),
            "camera_up": (0.0, 0.0, -1.0),
            "fov": 45.0,
            "confusion_reason": "俯视角度，位置置换在2D投影中不明显",
            "confusion_level": "medium",
        }

    def _confusing_view_symmetry(self, params: RuleParams, entities: List[PCRAREntity]) -> Dict[str, Any]:
        del entities
        axis = params.axis
        if axis in ("p", "r"):
            return {
                "camera_position": (7.0, 1.0, 1.0),
                "camera_target": (0.0, 0.0, 0.0),
                "camera_up": (0.0, 1.0, 0.0),
                "fov": 40.0,
                "confusion_reason": "侧视角度，左右对称变化在投影中不明显",
                "confusion_level": "high",
            }
        if axis == "R":
            return {
                "camera_position": (0.0, 0.5, 8.0),
                "camera_target": (0.0, 0.0, 0.0),
                "camera_up": (0.0, 1.0, 0.0),
                "fov": 35.0,
                "confusion_reason": "正面视角，姿态对称旋转不可见",
                "confusion_level": "high",
            }
        return {
            "camera_position": (5.0, 3.0, 5.0),
            "camera_target": (0.0, 0.0, 0.0),
            "camera_up": (0.0, 1.0, 0.0),
            "fov": 35.0,
            "confusion_reason": "2D图像无法体现密度对称变化",
            "confusion_level": "high",
        }

    def _confusing_view_copy(self, params: RuleParams, entities: List[PCRAREntity]) -> Dict[str, Any]:
        del entities
        if params.axis == "copy_density_cycle":
            return {
                "camera_position": (5.0, 3.0, 5.0),
                "camera_target": (0.0, 0.0, 0.0),
                "camera_up": (0.0, 1.0, 0.0),
                "fov": 35.0,
                "confusion_reason": "2D图像无法体现点云密度的循环拷贝模式",
                "confusion_level": "high",
            }
        return {
            "camera_position": (0.0, 2.0, 7.0),
            "camera_target": (0.0, 0.0, 0.0),
            "camera_up": (0.0, 1.0, 0.0),
            "fov": 40.0,
            "confusion_reason": "正面视角，循环拷贝的空间排列模式不明显",
            "confusion_level": "medium",
        }

    def _default_confusing_view(self) -> Dict[str, Any]:
        angle = self.rng.uniform(0, 2 * np.pi)
        elevation = self.rng.uniform(10, 30)
        r = 7.0
        x = r * np.cos(angle) * np.cos(np.radians(elevation))
        y = r * np.sin(np.radians(elevation))
        z = r * np.sin(angle) * np.cos(np.radians(elevation))
        return {
            "camera_position": (float(x), float(y), float(z)),
            "camera_target": (0.0, 0.0, 0.0),
            "camera_up": (0.0, 1.0, 0.0),
            "fov": 40.0,
            "confusion_reason": "倾斜视角，部分空间关系模糊",
            "confusion_level": "medium",
        }

    def render_point_cloud_image(
        self,
        points: np.ndarray,
        view_config: Dict[str, Any],
        image_size: Tuple[int, int] = (512, 512),
        point_size: float = 2.0,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        point_color: Optional[Tuple[int, int, int]] = None,
        proj_scale: float = 1.0,
    ) -> np.ndarray:
        points_arr = np.asarray(points)
        if points_arr.ndim != 2 or points_arr.shape[1] < 3:
            raise ValueError("points must have shape (N,3) or (N,>=6)")
        xyz = points_arr[:, :3]
        if points_arr.shape[1] >= 6:
            point_colors = np.clip(points_arr[:, 3:6], 0, 255).astype(np.uint8)
        elif point_color is not None:
            rgb = np.asarray(point_color, dtype=np.uint8).reshape(-1)
            if rgb.size != 3:
                raise ValueError("point_color must be a length-3 RGB tuple")
            point_colors = np.broadcast_to(rgb, (xyz.shape[0], 3)).astype(np.uint8)
        else:
            # Backward-compatible fallback color for uncolored point clouds.
            point_colors = np.broadcast_to(np.array([51, 102, 204], dtype=np.uint8), (xyz.shape[0], 3))

        cam_pos = np.array(view_config["camera_position"], dtype=float)
        cam_target = np.array(view_config["camera_target"], dtype=float)
        cam_up = np.array(view_config["camera_up"], dtype=float)

        forward = cam_target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        right = np.cross(forward, cam_up)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, forward)

        points_cam = xyz - cam_pos
        points_cam = np.stack(
            [
                np.dot(points_cam, right),
                np.dot(points_cam, up),
                np.dot(points_cam, forward),
            ],
            axis=1,
        )

        fov_rad = np.radians(float(view_config["fov"]))
        f = 1.0 / np.tan(fov_rad / 2.0)

        mask = points_cam[:, 2] > 0.1
        points_cam = points_cam[mask]
        point_colors = point_colors[mask]
        if len(points_cam) == 0:
            return np.full(
                (image_size[1], image_size[0], 3),
                np.array(background_color) * 255,
                dtype=np.uint8,
            )

        x_proj = points_cam[:, 0] * f / points_cam[:, 2]
        y_proj = points_cam[:, 1] * f / points_cam[:, 2]
        proj_scale = float(max(1e-6, proj_scale))
        x_proj *= proj_scale
        y_proj *= proj_scale

        x_pixel = ((x_proj + 1.0) * 0.5 * image_size[0]).astype(int)
        y_pixel = ((1.0 - y_proj) * 0.5 * image_size[1]).astype(int)
        depth = points_cam[:, 2]

        img = np.full(
            (image_size[1], image_size[0], 3),
            np.array(background_color) * 255,
            dtype=np.uint8,
        )
        depth_buffer = np.full((image_size[1], image_size[0]), np.inf)

        sort_indices = np.argsort(-depth)
        radius = int(point_size)

        for idx in sort_indices:
            px, py = x_pixel[idx], y_pixel[idx]
            d = depth[idx]
            if not (0 <= px < image_size[0] and 0 <= py < image_size[1]):
                continue
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < image_size[0] and 0 <= ny < image_size[1] and d < depth_buffer[ny, nx]:
                        img[ny, nx] = point_colors[idx]
                        depth_buffer[ny, nx] = d

        return img

    def _project_to_camera(
        self,
        points: np.ndarray,
        view_config: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        points_arr = np.asarray(points)
        xyz = points_arr[:, :3]

        cam_pos = np.array(view_config["camera_position"], dtype=float)
        cam_target = np.array(view_config["camera_target"], dtype=float)
        cam_up = np.array(view_config["camera_up"], dtype=float)

        forward = cam_target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        right = np.cross(forward, cam_up)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, forward)

        points_cam = xyz - cam_pos
        points_cam = np.stack(
            [
                np.dot(points_cam, right),
                np.dot(points_cam, up),
                np.dot(points_cam, forward),
            ],
            axis=1,
        )
        mask = points_cam[:, 2] > 0.1
        points_cam = points_cam[mask]
        if len(points_cam) == 0:
            return np.empty((0,), dtype=float), np.empty((0,), dtype=float)

        fov_rad = np.radians(float(view_config["fov"]))
        f = 1.0 / np.tan(fov_rad / 2.0)
        x_proj = points_cam[:, 0] * f / points_cam[:, 2]
        y_proj = points_cam[:, 1] * f / points_cam[:, 2]
        return x_proj, y_proj

    def compute_global_projection_scale(
        self,
        point_clouds: List[np.ndarray],
        view_config: Dict[str, Any],
        focus_quantile: float = 0.90,
        target_focus: float = 0.62,
        safety_quantile: float = 0.999,
        safety_margin: float = 0.90,
        min_scale: float = 0.45,
        max_scale: float = 2.50,
    ) -> float:
        projected_abs: List[np.ndarray] = []
        for points in point_clouds:
            if points is None:
                continue
            x_proj, y_proj = self._project_to_camera(points, view_config)
            if len(x_proj) == 0:
                continue
            # Use per-point projected extent to support robust quantile scaling.
            projected_abs.append(np.maximum(np.abs(x_proj), np.abs(y_proj)))

        if not projected_abs:
            return 1.0

        all_abs = np.concatenate(projected_abs, axis=0)
        if all_abs.size == 0:
            return 1.0

        fq = float(np.clip(focus_quantile, 0.5, 0.99))
        sq = float(np.clip(safety_quantile, fq, 0.9999))
        focus_val = float(np.quantile(all_abs, fq))
        safety_val = float(np.quantile(all_abs, sq))

        if focus_val <= 1e-6 or safety_val <= 1e-6:
            return 1.0

        scale = float(target_focus / focus_val)
        scale = float(np.clip(scale, min_scale, max_scale))

        # Hard safety cap: avoid clipping when a small tail extends to tile edges.
        if safety_val * scale > safety_margin:
            scale = float(safety_margin / safety_val)
        scale = float(np.clip(scale, min_scale, max_scale))
        return scale

    def save_rendered_image(self, img: np.ndarray, output_path: Path) -> None:
        try:
            from PIL import Image

            Image.fromarray(img).save(output_path)
        except ImportError:
            np.save(output_path.with_suffix(".npy"), img)


def _create_matrix_combined_image(
    grid_images: List[List[Optional[np.ndarray]]],
    candidate_images: List[np.ndarray],
    image_size: Tuple[int, int],
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    tile_w, tile_h = image_size
    pad = 16
    gap = 10
    cand_label_h = 36

    matrix_w = 3 * tile_w + 2 * gap
    matrix_h = 3 * tile_h + 2 * gap
    n_cand = max(1, len(candidate_images))
    cand_w = n_cand * tile_w + (n_cand - 1) * gap
    cand_h = tile_h

    canvas_w = max(matrix_w, cand_w) + 2 * pad
    canvas_h = pad + matrix_h + pad + cand_h + cand_label_h + pad

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    missing_boxes: List[Tuple[int, int, int, int]] = []

    mx = (canvas_w - matrix_w) // 2
    my = pad
    for r in range(3):
        for c in range(3):
            x0 = mx + c * (tile_w + gap)
            y0 = my + r * (tile_h + gap)
            img = grid_images[r][c]
            if img is None:
                blank = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
                blank[::8, :, :] = 220
                blank[:, ::8, :] = 220
                canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = blank
                missing_boxes.append((x0, y0, tile_w, tile_h))
            else:
                canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = img

    cy = my + matrix_h + pad
    cx = (canvas_w - cand_w) // 2
    candidate_boxes: List[Tuple[int, int, int, int, str]] = []
    for i, img in enumerate(candidate_images):
        x0 = cx + i * (tile_w + gap)
        canvas[cy : cy + tile_h, x0 : x0 + tile_w] = img
        label = chr(ord("A") + i)
        candidate_boxes.append((x0, cy, tile_w, tile_h, label))

    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    try:
        q_font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(48, int(tile_h * 0.45)))
    except Exception:
        q_font = ImageFont.load_default()
    try:
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        label_font = ImageFont.load_default()

    for x0, y0, w, h in missing_boxes:
        q_text = "?"
        q_bbox = draw.textbbox((0, 0), q_text, font=q_font)
        qw = q_bbox[2] - q_bbox[0]
        qh = q_bbox[3] - q_bbox[1]
        qx = x0 + (w - qw) // 2
        qy = y0 + (h - qh) // 2
        draw.text((qx, qy), q_text, fill=(70, 70, 70), font=q_font)

    for x0, y0, w, h, label in candidate_boxes:
        b = draw.textbbox((0, 0), label, font=label_font)
        tw = b[2] - b[0]
        tx = x0 + (w - tw) // 2
        ty = y0 + h + 6
        draw.text((tx, ty), label, fill=(20, 20, 20), font=label_font)

    return np.array(pil_img, dtype=np.uint8)


def generate_confusing_view_for_matrix_sample(
    rule_template: RuleTemplate,
    params: RuleParams,
    entities: List[PCRAREntity],
    grid_point_clouds: List[List[Optional[np.ndarray]]],
    candidate_point_clouds: List[np.ndarray],
    output_dir: Path,
    rng: Optional[np.random.Generator] = None,
    image_size: Tuple[int, int] = (512, 512),
) -> Dict[str, Any]:
    generator = ConfusingViewGenerator(rng)
    view_config = generator.select_confusing_viewpoint(rule_template, params, entities)
    all_point_clouds: List[np.ndarray] = []
    for row in grid_point_clouds:
        for points in row:
            if points is not None:
                all_point_clouds.append(points)
    all_point_clouds.extend(candidate_point_clouds)
    projection_scale = generator.compute_global_projection_scale(all_point_clouds, view_config)
    entity_iter = iter(entities)

    grid_view_paths: List[List[Optional[str]]] = [[None for _ in range(3)] for _ in range(3)]
    grid_images: List[List[Optional[np.ndarray]]] = [[None for _ in range(3)] for _ in range(3)]
    rendered_paths: List[str] = []

    for r in range(3):
        for c in range(3):
            points = grid_point_clouds[r][c]
            if points is None:
                continue
            entity = next(entity_iter, None)
            cloud_color = color_rgb(entity.obs.color_preset_idx) if entity is not None else None
            img = generator.render_point_cloud_image(
                points,
                view_config,
                image_size=image_size,
                point_color=cloud_color,
                proj_scale=projection_scale,
            )
            name = f"view_grid_{r}_{c}.png"
            generator.save_rendered_image(img, output_dir / name)
            grid_view_paths[r][c] = name
            grid_images[r][c] = img
            rendered_paths.append(name)

    candidate_view_paths: List[str] = []
    candidate_images: List[np.ndarray] = []
    for i, points in enumerate(candidate_point_clouds):
        entity = next(entity_iter, None)
        cloud_color = color_rgb(entity.obs.color_preset_idx) if entity is not None else None
        img = generator.render_point_cloud_image(
            points,
            view_config,
            image_size=image_size,
            point_color=cloud_color,
            proj_scale=projection_scale,
        )
        name = f"view_cand_{i}.png"
        generator.save_rendered_image(img, output_dir / name)
        candidate_view_paths.append(name)
        candidate_images.append(img)
        rendered_paths.append(name)

    combined = _create_matrix_combined_image(grid_images, candidate_images, image_size=image_size)
    combined_name = "view_combined.png"
    generator.save_rendered_image(combined, output_dir / combined_name)
    rendered_paths.append(combined_name)

    return {
        "view_config": view_config,
        "projection_scale": projection_scale,
        "layout": "matrix_3x3_plus_candidates",
        "grid_view_paths": grid_view_paths,
        "candidate_view_paths": candidate_view_paths,
        "rendered_paths": rendered_paths,
        "combined_image": combined_name,
        "confusion_reason": view_config["confusion_reason"],
        "confusion_level": view_config["confusion_level"],
    }
