"""数字识别共用工具：预处理、归一化、HOG 特征与 SVM 推理。

训练脚本 digit_train.py 与在线节点 number_detect_node.py 共用本模块，
保证训练和推理走完全相同的预处理链路。
"""

from typing import Optional, Tuple

import cv2
import numpy as np


DIGIT_SIZE = 64  # 归一化后的数字掩膜边长

# OpenCV 三类 C_SVC 的 1v1 决策函数顺序：(0,1), (0,2), (1,2)
_PAIRS_3 = ((0, 1), (0, 2), (1, 2))


def binarize_digit(gray: np.ndarray) -> np.ndarray:
    """灰度图 → 数字前景掩膜（数字为白 255，背景为黑 0），Otsu 自适应阈值。"""
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def normalize_mask(binary: np.ndarray) -> Optional[np.ndarray]:
    """把前景掩膜按外接框裁剪、等比缩放并居中到 DIGIT_SIZE x DIGIT_SIZE。

    无前景时返回 None。
    """
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    digit = binary[y:y + height, x:x + width]
    margin = 8
    scale = min(
        (DIGIT_SIZE - margin) / width,
        (DIGIT_SIZE - margin) / height,
    )
    resized = cv2.resize(
        digit,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((DIGIT_SIZE, DIGIT_SIZE), dtype=np.uint8)
    offset_x = (DIGIT_SIZE - resized.shape[1]) // 2
    offset_y = (DIGIT_SIZE - resized.shape[0]) // 2
    canvas[
        offset_y:offset_y + resized.shape[0],
        offset_x:offset_x + resized.shape[1],
    ] = resized
    return canvas


def create_hog() -> cv2.HOGDescriptor:
    """数字识别用的 HOG 描述器（经典手写数字配置）。"""
    return cv2.HOGDescriptor(
        _winSize=(DIGIT_SIZE, DIGIT_SIZE),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def hog_features(hog: cv2.HOGDescriptor, mask: np.ndarray) -> np.ndarray:
    """计算 64x64 掩膜的 HOG 特征向量（float32）。"""
    return hog.compute(mask).flatten().astype(np.float32)


def svm_class_scores(svm: cv2.ml_SVM, feature: np.ndarray) -> np.ndarray:
    """把 3 类 1v1 SVM 决策函数映射为各类别 [0,1] 置信度，索引 0..2 对应排序后的类别。"""
    gamma = svm.getGamma()
    support_vectors = svm.getSupportVectors().astype(np.float32)  # (sv数, 特征维)
    diff = support_vectors - feature.reshape(1, -1).astype(np.float32)
    kernel = np.exp(-gamma * np.sum(diff * diff, axis=1)).flatten()

    scores = np.zeros(3, dtype=np.float64)
    for index in range(3):
        rho, alpha, sv_index = svm.getDecisionFunction(index)
        alpha = np.asarray(alpha).flatten()
        sv_index = np.asarray(sv_index).flatten().astype(int)
        value = float(np.dot(alpha, kernel[sv_index])) - float(rho)
        probability = 1.0 / (1.0 + np.exp(-value))
        first, second = _PAIRS_3[index]
        # 决策值为正表示偏向序号更小的类别。
        scores[first] += probability
        scores[second] += 1.0 - probability
    return scores / 2.0
