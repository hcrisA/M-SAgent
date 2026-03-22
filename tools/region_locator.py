"""
区域定位器 - 基于网格BBox + 多点采样的定位策略

核心思路：
1. MLLM输出网格级BBox（粗粒度区域描述）
2. 在BBox内均匀采样N个点
3. MLLM判断每个点是否在目标上（Yes/No判断）
4. 筛选正负点，输入SAM进行分割

优势：
- 降低MLLM负担：从"像素级定位"降为"区域级描述"
- 提高鲁棒性：多点投票机制，单点误判不影响整体
- 保留网格坐标系：完全兼容现有网格设计
"""

import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from config import Config


class RegionLocator:
    """区域定位器 - 基于网格BBox + 多点采样"""
    
    def __init__(self, mllm_processor):
        self.mllm = mllm_processor
        self.grid_rows = Config.GRID_ROWS
        self.grid_cols = Config.GRID_COLS
        self.sample_count = 20  # 默认采样点数
        self.batch_size = 5     # 每批验证的点数
        
    def locate_with_bbox_sampling(self, tool_params, image, sam_processor, grid_info=None, text_prompt=""):
        """
        使用BBox + 多点采样进行定位
        
        Args:
            tool_params: {"bbox": [col1, row1, col2, row2], "text_prompt": "..."}
            image: PIL Image
            sam_processor: SAM处理器
            grid_info: 网格信息（可选）
            text_prompt: 目标描述文本
            
        Returns:
            dict: 分割结果
        """
        print("=" * 60)
        print("区域定位器启动")
        print("=" * 60)
        
        # 提取BBox参数
        bbox = tool_params.get("bbox")
        if not bbox or len(bbox) != 4:
            print("错误: 无效的BBox参数")
            return {"success": False, "message": "Invalid bbox parameters"}
        
        # 提取文本提示
        target_text = tool_params.get("text_prompt", text_prompt)
        
        print(f"目标描述: {target_text}")
        print(f"网格BBox: 列[{bbox[0]}-{bbox[2]}], 行[{bbox[1]}-{bbox[3]}]")
        
        # Step 1: 将网格BBox转换为像素坐标
        pixel_bbox = self.grid_bbox_to_pixel(bbox, image.size)
        print(f"像素BBox: x[{pixel_bbox[0]}-{pixel_bbox[2]}], y[{pixel_bbox[1]}-{pixel_bbox[3]}]")
        
        # Step 2: 在BBox内均匀采样点
        sample_points = self.sample_points_in_bbox(pixel_bbox, n=self.sample_count)
        print(f"采样点数: {len(sample_points)}")
        
        # Step 3: 可视化采样点（带编号）
        marked_image = self.visualize_sampled_points(image, sample_points)
        
        # 保存可视化结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        marked_path = Config.TOOL_CALLS_LOG / f"region_locator_sampled_{timestamp}.jpg"
        marked_image.save(marked_path)
        print(f"采样点可视化已保存: {marked_path}")
        
        # Step 4: MLLM判断每个点是否在目标上
        positive_points, negative_points = self.verify_points_with_mllm(
            marked_image, sample_points, target_text
        )
        
        print(f"验证结果: 正点={len(positive_points)}个, 负点={len(negative_points)}个")
        
        # Step 5: 容错处理
        if not positive_points:
            print("警告: 未找到正点，使用BBox中心点作为默认前景点")
            center_x = (pixel_bbox[0] + pixel_bbox[2]) / 2
            center_y = (pixel_bbox[1] + pixel_bbox[3]) / 2
            positive_points = [[center_x, center_y]]
        
        # Step 6: 准备SAM输入
        all_points = positive_points + negative_points
        all_labels = [1] * len(positive_points) + [0] * len(negative_points)
        
        print(f"SAM输入: {len(all_points)}个点 ({len(positive_points)}正 + {len(negative_points)}负)")
        
        # Step 7: 调用SAM进行分割
        result = sam_processor.segment_with_points(
            image, 
            points=all_points, 
            labels=all_labels, 
            multimask_output=True
        )
        
        # Step 8: 保存分割结果可视化
        if result.get("success") and result.get("best_result"):
            try:
                best_mask = result["best_result"]["mask"]
                vis_image = sam_processor.apply_mask_to_image(image, best_mask)
                
                # 叠加采样点
                vis_image = self.visualize_points_on_result(
                    vis_image, positive_points, negative_points
                )
                
                vis_path = Config.TOOL_CALLS_LOG / f"region_locator_result_{timestamp}.jpg"
                vis_image.save(vis_path)
                print(f"分割结果已保存: {vis_path}")
            except Exception as e:
                print(f"保存分割结果可视化失败: {e}")
        
        # 添加元数据
        if result.get("success"):
            result["method"] = "region_locator_bbox_sampling"
            result["grid_bbox"] = bbox
            result["pixel_bbox"] = pixel_bbox
            result["positive_points"] = positive_points
            result["negative_points"] = negative_points
        
        print("=" * 60)
        return result
    
    def grid_bbox_to_pixel(self, grid_bbox, image_size):
        """
        将网格BBox转换为像素坐标
        
        Args:
            grid_bbox: [col1, row1, col2, row2] 网格坐标
            image_size: (width, height) 图像尺寸
            
        Returns:
            [x1, y1, x2, y2] 像素坐标
        """
        width, height = image_size
        
        # 计算网格线的像素位置
        row_positions = np.linspace(0, height, self.grid_rows + 1, dtype=int)
        col_positions = np.linspace(0, width, self.grid_cols + 1, dtype=int)
        
        col1, row1, col2, row2 = grid_bbox
        
        # 边界保护
        col1 = max(0, min(col1, self.grid_cols))
        col2 = max(0, min(col2, self.grid_cols))
        row1 = max(0, min(row1, self.grid_rows))
        row2 = max(0, min(row2, self.grid_rows))
        
        # 确保col1 <= col2, row1 <= row2
        if col1 > col2:
            col1, col2 = col2, col1
        if row1 > row2:
            row1, row2 = row2, row1
        
        x1 = int(col_positions[col1])
        y1 = int(row_positions[row1])
        x2 = int(col_positions[col2])
        y2 = int(row_positions[row2])
        
        return [x1, y1, x2, y2]
    
    def sample_points_in_bbox(self, pixel_bbox, n=20):
        """
        在BBox内均匀采样点
        
        Args:
            pixel_bbox: [x1, y1, x2, y2] 像素坐标
            n: 采样点数
            
        Returns:
            list of [x, y] 采样点列表
        """
        x1, y1, x2, y2 = pixel_bbox
        
        # 确保x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # 计算采样网格
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        if bbox_width <= 0 or bbox_height <= 0:
            return [[(x1 + x2) / 2, (y1 + y2) / 2]]
        
        # 根据长宽比计算行列数
        aspect_ratio = bbox_width / bbox_height
        cols = int(np.sqrt(n * aspect_ratio))
        rows = int(n / cols) if cols > 0 else 1
        
        cols = max(2, min(cols, n))
        rows = max(2, min(rows, n))
        
        # 均匀采样（排除边界）
        x_coords = np.linspace(x1, x2, cols + 2)[1:-1]
        y_coords = np.linspace(y1, y2, rows + 2)[1:-1]
        
        points = []
        for x in x_coords:
            for y in y_coords:
                points.append([float(x), float(y)])
        
        # 限制数量
        return points[:n]
    
    def visualize_sampled_points(self, image, points):
        """
        可视化采样点，每个点带编号
        
        Args:
            image: PIL Image
            points: 采样点列表
            
        Returns:
            PIL Image: 带编号的图像
        """
        marked_image = image.copy()
        draw = ImageDraw.Draw(marked_image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for i, point in enumerate(points):
            x, y = point
            
            # 绘制圆点
            radius = 8
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(255, 165, 0),  # 橙色
                outline=(0, 0, 0),
                width=2
            )
            
            # 添加编号
            text = str(i + 1)
            draw.text((x - 4, y - 4), text, fill=(255, 255, 255), font=font)
        
        return marked_image
    
    def visualize_points_on_result(self, image, positive_points, negative_points):
        """
        在分割结果上可视化正负点
        
        Args:
            image: PIL Image
            positive_points: 正点列表
            negative_points: 负点列表
            
        Returns:
            PIL Image
        """
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # 绘制正点（绿色）
        for point in positive_points:
            x, y = point
            radius = 6
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(0, 255, 0),  # 绿色
                outline=(0, 0, 0),
                width=2
            )
        
        # 绘制负点（红色）
        for point in negative_points:
            x, y = point
            radius = 6
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(255, 0, 0),  # 红色
                outline=(0, 0, 0),
                width=2
            )
        
        return result_image
    
    def verify_points_with_mllm(self, marked_image, points, text_prompt):
        """
        使用MLLM验证采样点是否在目标上
        
        Args:
            marked_image: 带编号的图像
            points: 采样点列表
            text_prompt: 目标描述
            
        Returns:
            (positive_points, negative_points)
        """
        print(f"\n开始MLLM多点验证...")
        
        positive_points = []
        negative_points = []
        
        # 分批处理
        for batch_start in range(0, len(points), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(points))
            batch_points = points[batch_start:batch_end]
            batch_indices = list(range(batch_start + 1, batch_end + 1))
            
            # 构建验证问题
            question = self._build_verification_question(
                batch_indices, text_prompt
            )
            
            # 调用MLLM
            try:
                answer = self._call_mllm_for_verification(
                    marked_image, question
                )
                
                # 解析答案
                batch_positive, batch_negative = self._parse_verification_answer(
                    answer, batch_points, batch_indices
                )
                
                positive_points.extend(batch_positive)
                negative_points.extend(batch_negative)
                
            except Exception as e:
                print(f"MLLM验证失败: {e}")
                # 失败时，默认所有点为正点（保守策略）
                positive_points.extend(batch_points)
        
        return positive_points, negative_points
    
    def _build_verification_question(self, indices, text_prompt):
        """构建验证问题"""
        indices_str = ", ".join([str(i) for i in indices])
        
        question = f"""请判断以下编号的点是否位于目标"{text_prompt}"上。

需要判断的点编号: {indices_str}

对于每个点，请回答"Yes"（在目标上）或"No"（不在目标上）。

请按以下JSON格式回答：
{{
    "1": "Yes/No",
    "2": "Yes/No",
    ...
}}
"""
        return question
    
    def _call_mllm_for_verification(self, image, question):
        """调用MLLM进行点验证"""
        # 这里需要根据实际的MLLM接口调用
        # 暂时返回模拟结果
        # 实际实现需要调用 self.mllm 的相应方法
        
        # 保存临时图像
        temp_path = Config.BASE_DIR / "temp_verify_image.jpg"
        image.save(temp_path)
        
        # 调用MLLM
        # TODO: 实现实际的MLLM调用
        # response = self.mllm.verify_points(temp_path, question)
        
        # 临时返回模拟结果
        return {"1": "Yes", "2": "Yes", "3": "No", "4": "No", "5": "Yes"}
    
    def _parse_verification_answer(self, answer, points, indices):
        """解析MLLM的验证答案"""
        positive = []
        negative = []
        
        for i, idx in enumerate(indices):
            idx_str = str(idx)
            if idx_str in answer:
                verdict = answer[idx_str].strip().lower()
                if verdict == "yes":
                    positive.append(points[i])
                else:
                    negative.append(points[i])
            else:
                # 未找到答案，默认为正点
                positive.append(points[i])
        
        return positive, negative


# 测试代码
if __name__ == "__main__":
    print("RegionLocator 模块加载成功")
    print("核心功能:")
    print("  1. grid_bbox_to_pixel: 网格BBox转像素坐标")
    print("  2. sample_points_in_bbox: 在BBox内均匀采样")
    print("  3. verify_points_with_mllm: MLLM验证点是否在目标上")
    print("  4. locate_with_bbox_sampling: 完整定位流程")
