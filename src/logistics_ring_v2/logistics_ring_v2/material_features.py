"""物料判色共用工具: 白背景基准与相对 Lab 特征(训练与在线共用, 保证一致)。

v2 相对化(相对旧包 color_features.py 的绝对 Lab 特征):
    旧特征 = 物料块自身 Lab 均值+方差(绝对量), 换光照/白平衡后整体漂移,
    实验室训练的模型到现场即失配。v2 把特征挂到每帧现算的白背景基准上:
    白地图+白盘占画面 70% 以上, 取 S<saturation_threshold 像素的 Lab 中位数
    为基准 (μL, μa, μb), 物料块特征全部改用相对量:
        d_a     = mean_a - μa          色度偏移(红绿轴)
        d_b     = mean_b - μb          色度偏移(黄蓝轴)
        l_ratio = mean_L / μL          亮度比(黑/深色/浅色的分界)
        std_a, std_b                  块内色度纹理
        d_ab    = mean(|a-μa|+|b-μb|)  整块色度强度(与分割判据同源)
    同一物料在实验室与现场的相对特征基本一致, 模型天生耐光照失配。
    黑色物料在线不走本特征(按"暗+灰"定义判黑), 训练集保留 color_5 仅为完整。

三方一致性约束(改任何一处必须同步):
    material2_sample_capture  采集瞬间用与在线一致的方式算 bg_ref 存 sidecar
    material2_color_train     训练时读 sidecar 的 bg_ref
    material2_detect_node     在线每帧现算 bg_ref
"""

from typing import Optional, Tuple

import numpy as np


# 规则颜色编号: 红1 黄2 蓝3 绿4 黑5 浅蓝6
COLOR_NAMES = {1: '红', 2: '黄', 3: '蓝', 4: '绿', 5: '黑', 6: '浅蓝'}
# cv2.putText 不支持中文(显示 ??), 窗口文字用 ASCII 缩写
COLOR_ABBRS = {1: 'R', 2: 'Y', 3: 'B', 4: 'G', 5: 'K', 6: 'LB'}

# 模型文件 features 键标记: 训练与在线共同校验, 防止误载旧绝对特征模型
FEATURE_TAG = 'lab_relative_6'

# 背景基准最少白像素数: 画面被大面积遮挡(手/夹爪伸入)时基准不可信
MIN_BACKGROUND_PIXELS = 1000


def estimate_background_ref(
    lab_f32: np.ndarray,
    sat: np.ndarray,
    saturation_threshold: int,
) -> Optional[Tuple[float, float, float]]:
    """白背景 Lab 基准 (μL, μa, μb): S<阈值的像素(白地图+白盘)取 Lab 中位数。

    输入须为同一帧的 float32 Lab 与 HSV 饱和度通道; 白像素不足时返回 None,
    由调用方决定沿用旧基准或跳帧。
    """
    bg = sat < saturation_threshold
    if int(bg.sum()) < MIN_BACKGROUND_PIXELS:
        return None
    return (
        float(np.median(lab_f32[:, :, 0][bg])),
        float(np.median(lab_f32[:, :, 1][bg])),
        float(np.median(lab_f32[:, :, 2][bg])),
    )


def color_feature_vector_relative(
    lab: np.ndarray,
    mask: np.ndarray,
    bg_ref: Tuple[float, float, float],
) -> np.ndarray:
    """相对白背景的物料块颜色特征, 6 维 float32。

    lab: float32 Lab 图(H/W/3, OpenCV 0~255 口径); mask: 物料块掩膜(非零即块内);
    bg_ref: 同帧白背景基准。维度: [d_a, d_b, l_ratio, std_a, std_b, d_ab]。
    """
    pixels = lab[mask > 0]
    if pixels.shape[0] == 0:
        raise ValueError('掩膜为空，无法提取颜色特征')
    mu_l, mu_a, mu_b = bg_ref
    mean_l = float(pixels[:, 0].mean())
    mean_a = float(pixels[:, 1].mean())
    mean_b = float(pixels[:, 2].mean())
    d_a = mean_a - mu_a
    d_b = mean_b - mu_b
    l_ratio = mean_l / max(mu_l, 1.0)
    std_a = float(pixels[:, 1].std())
    std_b = float(pixels[:, 2].std())
    d_ab = float(
        (np.abs(pixels[:, 1] - mu_a) + np.abs(pixels[:, 2] - mu_b)).mean()
    )
    return np.asarray(
        [d_a, d_b, l_ratio, std_a, std_b, d_ab], dtype=np.float32
    )


def color_name(digit: int) -> str:
    return COLOR_NAMES.get(digit, str(digit))


def color_abbr(digit: int) -> str:
    return COLOR_ABBRS.get(digit, str(digit))
