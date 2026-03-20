import os
from pathlib import Path
import torch

class Config:
    # 基础路径
    BASE_DIR = Path("/root/autodl-tmp/mllm_sam_project")
    OUTPUT_DIR = BASE_DIR / "outputs"
    EXAMPLES_DIR = BASE_DIR / "examples"
    
    # 模型路径
    QWEN_MODEL_PATH = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"
    SAM3_MODEL_PATH = "/root/autodl-tmp/sam3"
    
    #文本提示词路径
    SYSTEM_PROMPT= BASE_DIR / "prompts" / "system_prompt_en.txt"
    SYSTEM_PROMPT_ITERATIVE_CHECKING= BASE_DIR / "prompts" / "system_prompt_iterative_checking_en.txt"
    
    # 工具调用中间记录保存路径
    TOOL_CALLS_LOG = BASE_DIR / "tool_calls_log"
    
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 工具配置
    MAX_TOOL_CALLS = 5           # 最大工具调用次数
    CONFIDENCE_THRESHOLD = 0.7   # 分割置信度阈值
    
    # 网格配置
    GRID_ROWS = 5               # 默认行数
    GRID_COLS = 5               # 默认列数
    MIN_CELL_SIZE = 50           # 最小单元格尺寸
    
    # 图像分辨率配置
    # 增大 MIN_PIXELS 会强制放大低分辨率图片，由 MLLM 看到更多细节
    # 减小 MAX_PIXELS 会强制缩小高分辨率图片，节省显存
    MLLM_MIN_PIXELS = 3000000  # ~300万像素 (默认)
    MLLM_MAX_PIXELS = 12845056       # ~1200万像素 (默认)

    # 定位配置
    MAX_POINTS = 5               # 最大定位点数
    MIN_POINTS = 2               # 最少定位点数
    
    # 概念生成配置
    MIN_CONCEPTS = 3             # 最少概念数
    MAX_CONCEPTS = 5             # 最多概念数
    
    @classmethod
    def setup_dirs(cls):
        """创建必要目录"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        cls.TOOL_CALLS_LOG.mkdir(parents=True, exist_ok=True)
        
        # 创建SAM3结果目录
        sam_results = Path("/root/autodl-tmp/sam3/results")
        sam_results.mkdir(parents=True, exist_ok=True)
        