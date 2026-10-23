"""颜色识别共用工具：物料块颜色特征（训练与在线节点共用，保证一致）。"""

import cv2
import numpy as np

# 规则颜色编号：红1 黄2 蓝3 绿4 黑5 浅蓝6
COLOR_NAMES = {1: '红', 2: '黄', 3: '蓝', 4: '绿', 5: '黑', 6: '浅蓝'}
# cv2.putText 不支持中文(显示 ??), 窗口文字用 ASCII 缩写
COLOR_ABBRS = {1: 'R', 2: 'Y', 3: 'B', 4: 'G', 5: 'K', 6: 'LB'}


def color_feature_vector(lab_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """掩膜区域的 Lab 颜色特征：各通道均值+标准差，6 维 float32。

    Lab 的 a/b 分量对亮度不敏感，是颜色判定的主特征；std 提供材质纹理信息。
    """
    pixels = lab_image[mask > 0].astype(np.float32)
    if pixels.shape[0] == 0:
        raise ValueError('掩膜为空，无法提取颜色特征')
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


def color_name(digit: int) -> str:
    return COLOR_NAMES.get(digit, str(digit))


def color_abbr(digit: int) -> str:
    return COLOR_ABBRS.get(digit, str(digit))
