import numpy as np
import json
from PIL import ImageDraw
from config import Config
from utils.image_utils import add_grid_to_image
from datetime import datetime

class ObjectLocator:
    """目标定位工具"""
    
    def __init__(self, mllm_processor):
        self.mllm = mllm_processor
        self.grid_rows = Config.GRID_ROWS
        self.grid_cols = Config.GRID_COLS
        
    def locate_object_with_points(self, tool_params, image, sam_processor):
        """定位目标"""
        print("定位目标...")
        
        # 提取点参数
        grid_points = tool_params.get("points")
        labels = tool_params.get("labels")
        
        if not grid_points or not labels or len(grid_points) != len(labels):
            print("未提供有效的点或标签")
            return {"success": False, "message": "Invalid points or labels"}
            
        print(f"收到网格点参数: {grid_points}, 标签: {labels}")
        
        # 将网格坐标转换为像素坐标
        points = []
        width, height = image.size
        
        # 重新计算线的位置
        row_positions = np.linspace(0, height, self.grid_rows + 1, dtype=int)
        col_positions = np.linspace(0, width, self.grid_cols + 1, dtype=int)
        
        valid_labels = []
        
        for i, p in enumerate(grid_points):
            try:
                # 输入是 [col_idx, row_idx] (x, y)
                # 列号 (x axis)
                col_idx = int(p[0])
                # 行号 (y axis)
                row_idx = int(p[1])
                
                # 检查边界
                if row_idx < 0 or row_idx > self.grid_rows or col_idx < 0 or col_idx > self.grid_cols:
                    print(f"警告: 点 {p} (Col: {col_idx}, Row: {row_idx}) 超出网格范围，已忽略")
                    continue
                
                # 计算像素坐标
                # 行号映射: row_idx -> y
                y = row_positions[row_idx]
                # 列号映射: col_idx -> x
                x = col_positions[col_idx]
                
                points.append([float(x), float(y)])
                valid_labels.append(labels[i])
                print(f"解析点 {i}: Grid(Col={col_idx}, Row={row_idx}) -> Pixel(x={x:.1f}, y={y:.1f})")
                
            except Exception as e:
                print(f"坐标转换失败 {p}: {e}")
        
        if not points:
            print("没有有效的像素点")
            return {"success": False, "message": "No valid points"}
            
        labels = valid_labels
        print(f"转换后的像素点({len(points)}个): {points}")
        
        # 可视化点
        try:
            points_img = self.visualize_points(
                image,
                points,
                labels
            )
            
            # 保存可视化结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            points_path = Config.TOOL_CALLS_LOG / f"object_locator_points_{timestamp}.jpg"
            points_img.save(points_path)
            print(f"点可视化已保存: {points_path}")
        except Exception as e:
            print(f"点可视化失败: {e}")
            
        # 尝试调用 sam 进行分割
        if sam_processor:
            print("调用 sam 进行分割...")
            result = sam_processor.segment_with_points(
                image,
                points=points,
                labels=labels,
                multimask_output=True
            )
            
            # 保存分割结果可视化
            if result.get("success") and result.get("best_result"):
                try:
                    best_mask = result["best_result"]["mask"]
                    # 应用MASK
                    vis_image = sam_processor.apply_mask_to_image(image, best_mask)
                    # 叠加点
                    vis_image = self.visualize_points(vis_image, points, labels)
                    
                    vis_path = Config.TOOL_CALLS_LOG / f"object_locator_result_{timestamp}.jpg"
                    vis_image.save(vis_path)
                    print(f"定位分割结果已保存: {vis_path}")
                except Exception as e:
                    print(f"保存分割结果可视化失败: {e}")
            
            return result

        # 返回点信息
        return {
            "success": True,
            "points": points,
            "labels": labels
        }

    def visualize_points(self, image, points, labels):
        """
        可视化点和标签
        
        Args:
            image: PIL Image
            points: 点列表
            labels: 标签列表
            
        Returns:
            PIL Image: 带点的图像
        """
        image_with_points = image.copy()
        draw = ImageDraw.Draw(image_with_points)
        
        for point, label in zip(points, labels):
            # 兼容点格式 [x, y] 或 (x, y)
            x, y = point
            
            # 前景点：绿色，背景点：红色
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            radius = 8
            
            # 画圆
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
                outline=(0, 0, 0)
            )
            
            # 添加标签文本
            label_text = "F" if label == 1 else "B"
            # 简单的文本位置调整
            draw.text((x - 3, y - 10), label_text, fill=(255, 255, 255)) # 白色文字
        
        return image_with_points