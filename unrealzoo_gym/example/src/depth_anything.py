import cv2
import torch
import numpy as np
from typing import Optional, Literal
import logging
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
depth_anything_path = os.path.join(current_dir, '../solution/Depth_Anything_V2')
if depth_anything_path not in sys.path:
    sys.path.insert(0, depth_anything_path)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_ANYTHING_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_AVAILABLE = False
    logging.warning("Depth Anything V2 not installed. Depth estimation will be disabled.")


class DepthEstimator:
    """Depth Anything V2 模型封装"""
    
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    def __init__(
        self,
        encoder: Literal['vits', 'vitb', 'vitl', 'vitg'] = 'vitl',
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        初始化 Depth Estimator
        
        Args:
            encoder: 编码器类型，可选 'vits', 'vitb', 'vitl', 'vitg'
            checkpoint_path: 模型权重路径，默认为 'checkpoints/depth_anything_v2_{encoder}.pth'
            device: 设备类型，默认自动检测 (cuda > mps > cpu)
        """
        if not DEPTH_ANYTHING_AVAILABLE:
            raise ImportError(
                "Depth Anything V2 not installed. "
                "Please install it from: https://github.com/DepthAnything/Depth-Anything-V2"
            )
        
        self.encoder = encoder
        self.device = device or self._get_device()
        
        # 设置默认权重路径
        if checkpoint_path is None:
            checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
        self.checkpoint_path = checkpoint_path
        
        # 加载模型
        self.model = self._load_model()
        logging.info(f"Depth Estimator initialized with {encoder} on {self.device}")
    
    @staticmethod
    def _get_device() -> str:
        """自动检测最佳设备"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _load_model(self) -> DepthAnythingV2:
        """加载 Depth Anything V2 模型"""
        # 创建模型
        model_config = self.MODEL_CONFIGS[self.encoder]
        model = DepthAnythingV2(**model_config)
        
        # 加载权重
        try:
            state_dict = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint from {self.checkpoint_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please download it from: "
                f"https://github.com/DepthAnything/Depth-Anything-V2#pretrained-models"
            )
        
        # 移动到设备并设置为评估模式
        model = model.to(self.device).eval()
        return model
    
    def estimate_depth(
        self,
        image: np.ndarray,
        return_normalized: bool = False
    ) -> np.ndarray:
        """
        估计深度图
        
        Args:
            image: 输入图像 (RGB or BGR, HxWx3)
            return_normalized: 是否返回归一化的深度图 (0-255)
        
        Returns:
            depth_map: 深度图 (HxW)
        """
        # 如果是 RGB，转换为 BGR（OpenCV 格式）
        if image.shape[-1] == 3:
            # 假设输入是 RGB
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # 推理深度
        with torch.no_grad():
            depth = self.model.infer_image(image_bgr)  # HxW numpy array
        
        # 归一化到 0-255（可选）
        if return_normalized:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
        
        return depth
    
    def estimate_depth_from_path(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        从图像文件估计深度
        
        Args:
            image_path: 输入图像路径
            output_path: 输出深度图路径（可选）
        
        Returns:
            depth_map: 深度图
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # 估计深度
        depth = self.estimate_depth(image, return_normalized=True)
        
        # 保存（可选）
        if output_path:
            cv2.imwrite(output_path, depth)
            logging.info(f"Depth map saved to {output_path}")
        
        return depth


def create_depth_estimator(
    encoder: str = 'vitl',
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
) -> Optional[DepthEstimator]:
    """
    便捷函数：创建深度估计器
    
    Args:
        encoder: 编码器类型
        checkpoint_path: 模型权重路径
        device: 设备类型
    
    Returns:
        DepthEstimator 实例，如果不可用则返回 None
    """
    if not DEPTH_ANYTHING_AVAILABLE:
        logging.warning("Depth Anything V2 not available, returning None")
        return None
    
    try:
        estimator = DepthEstimator(encoder, checkpoint_path, device)
        return estimator
    except Exception as e:
        logging.error(f"Failed to create depth estimator: {e}")
        return None


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建深度估计器
    estimator = create_depth_estimator(encoder='vitl', checkpoint_path="/Volumes/KINGSTON/yujiarong/OpenEQA/unrealzoo_gym/example/solution/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth",
                                       device = 'mps')
    
    if estimator is not None:
        # 从文件估计深度
        depth = estimator.estimate_depth_from_path(
            image_path='/Volumes/KINGSTON/yujiarong/Depth-Anything-V2/img_test/test_img.png',
            output_path='depth_output.png'
        )
        print(f"Depth map shape: {depth.shape}")