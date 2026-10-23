"""离线训练数字 1/2/3 分类器。

流程：实拍样本 → 预处理归一化 → 增强（旋转/缩放/光照/反光缺失/笔画断裂）
→ HOG → SVM(trainAuto) → 保存模型并打印评估结果。

用法：
    ros2 run logistics_vision_capture digit_trainer \
        --data-root ~/logistics_vision_data \
        --output ~/logistics_vision_data/digit_model/digits_svm.yml
"""

import argparse
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .digit_features import (
    DIGIT_SIZE,
    binarize_digit,
    create_hog,
    hog_features,
    normalize_mask,
)


def remove_edge_components(binary: np.ndarray, edge: int = 2) -> np.ndarray:
    """剔除贴到左右边界的窄条或小斑点（采集黑边、角点噪点）。

    裁剪很紧时数字本体也会贴边，因此只剔除宽度很小或面积很小的成分。
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    height, width = binary.shape
    result = np.zeros_like(binary)
    for label in range(1, count):
        x, _, box_width, _, area = stats[label]
        touches_edge = x <= edge or x + box_width >= width - edge
        is_debris = box_width <= 6 or area < 50
        if touches_edge and is_debris:
            continue
        result[labels == label] = 255
    return result


def preprocess_sample(gray: np.ndarray) -> Optional[np.ndarray]:
    """单张实拍灰度图 → 归一化掩膜；二值化后无前景返回 None。"""
    binary = remove_edge_components(binarize_digit(gray))
    return normalize_mask(binary)


def add_glare(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    """在灰度图上叠加高斯亮斑，模拟灯光反光；反光处二值化后自然吃掉笔画。"""
    height, width = gray.shape
    result = gray.astype(np.float32)
    for _ in range(rng.randint(2, 6)):
        center_x = rng.uniform(0, width)
        center_y = rng.uniform(0, height)
        radius = rng.uniform(2, 8)
        strength = rng.uniform(60, 130)
        spot = np.zeros((height, width), np.float32)
        cv2.circle(spot, (int(center_x), int(center_y)), int(radius) + 2, strength, -1)
        spot = cv2.GaussianBlur(spot, (0, 0), radius / 2.0)
        result += spot
    return np.clip(result, 0, 255).astype(np.uint8)


def add_stroke_breaks(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """在前景笔画上叠加黑色软边斑块，模拟数字局部断裂缺失。"""
    points = cv2.findNonZero(mask)
    if points is None:
        return mask
    height, width = mask.shape
    result = mask.copy()
    for _ in range(rng.randint(2, 6)):
        pixel = points[rng.randrange(len(points))][0]
        radius = rng.uniform(2, 6)
        spot = np.zeros((height, width), np.uint8)
        cv2.circle(spot, (int(pixel[0]), int(pixel[1])), int(radius), 255, -1)
        spot = cv2.GaussianBlur(spot, (0, 0), radius / 2.0)
        result[spot > 127] = 0
    return result


def warp_mask(mask: np.ndarray, angle_deg: float, scale: float, shear_deg: float) -> np.ndarray:
    """对掩膜做绕中心旋转 + 缩放 + 轻微剪切。"""
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, scale)
    shear = math.tan(math.radians(shear_deg))
    shear_matrix = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    matrix = shear_matrix @ np.vstack([matrix, [0.0, 0.0, 1.0]])
    return cv2.warpAffine(
        mask, matrix[:2], (width, height),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )


def augment_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """对归一化掩膜做几何与结构增强，返回一个新变体。"""
    result = warp_mask(
        mask,
        angle_deg=rng.uniform(-30.0, 30.0),
        scale=rng.uniform(0.8, 1.2),
        shear_deg=rng.uniform(-5.0, 5.0),
    )
    if rng.random() < 0.35:
        kernel_size = rng.choice([1, 2])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.erode(result, kernel, iterations=1)
    if rng.random() < 0.25:
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    if rng.random() < 0.5:
        result = add_stroke_breaks(result, rng)
    return result


def augment_sample(gray: np.ndarray, count: int, rng: random.Random) -> List[np.ndarray]:
    """一张实拍灰度图 → count 个增强掩膜变体（含光照与反光模拟）。"""
    variants: List[np.ndarray] = []
    for _ in range(count):
        augmented_gray = gray
        if rng.random() < 0.6:
            alpha = rng.uniform(0.75, 1.35)
            beta = rng.uniform(-25.0, 25.0)
            augmented_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        if rng.random() < 0.5:
            augmented_gray = add_glare(augmented_gray, rng)
        binary = binarize_digit(augmented_gray)
        mask = normalize_mask(binary)
        if mask is None:
            continue
        variants.append(augment_mask(mask, rng))
    return variants


def load_dataset(data_root: Path, subset: str) -> Dict[int, List[Path]]:
    """读取 digit_{1,2,3} 目录下的实拍样本路径，按类别分组。"""
    dataset: Dict[int, List[Path]] = {}
    for digit in (1, 2, 3):
        directory = data_root / subset / ('digit_%d' % digit)
        paths = sorted(
            path
            for path in directory.glob('*')
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        )
        if not paths:
            raise SystemExit('未找到样本目录或样本为空: %s' % directory)
        dataset[digit] = paths
    return dataset


def split_holdout(
    dataset: Dict[int, List[Path]], holdout_ratio: float, rng: random.Random
) -> Tuple[List[Tuple[int, Path]], List[Tuple[int, Path]]]:
    """分层划分训练/留出样本，返回 (train, test)，元素为 (digit, path)。"""
    train: List[Tuple[int, Path]] = []
    test: List[Tuple[int, Path]] = []
    for digit, paths in sorted(dataset.items()):
        shuffled = list(paths)
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * holdout_ratio)))
        test.extend((digit, path) for path in shuffled[:test_count])
        train.extend((digit, path) for path in shuffled[test_count:])
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description='训练数字 1/2/3 的 HOG+SVM 分类器')
    parser.add_argument('--data-root', default='~/logistics_vision_data')
    parser.add_argument('--subset', default='digit_dataset',
                        help='data_root 下存放样本的子目录名（digit_dataset / digit_train）')
    parser.add_argument('--output', default='')
    parser.add_argument('--augment-count', type=int, default=15)
    parser.add_argument('--holdout-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else data_root / 'digit_model' / 'digits_svm.yml'
    )
    rng = random.Random(args.seed)

    dataset = load_dataset(data_root, args.subset)
    train_items, test_items = split_holdout(dataset, args.holdout_ratio, rng)
    print('样本统计: 训练 %d 张, 留出测试 %d 张' % (len(train_items), len(test_items)))

    hog = create_hog()
    digit_by_index = (1, 2, 3)

    # 训练集：每张实拍图生成 augment-count 个增强变体，外加原始掩膜。
    features: List[np.ndarray] = []
    labels: List[int] = []
    for digit, path in train_items:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print('警告: 无法读取样本 %s，已跳过' % path.name)
            continue
        mask = preprocess_sample(gray)
        if mask is None:
            print('警告: 样本二值化后无前景 %s，已跳过' % path.name)
            continue
        variants = augment_sample(gray, args.augment_count, rng)
        variants.append(mask)
        for variant in variants:
            features.append(hog_features(hog, variant))
            labels.append(digit)
    samples = np.array(features, dtype=np.float32)
    label_array = np.array(labels, dtype=np.int32)
    print('训练样本: %d 个 (含增强变体)' % len(samples))

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setTermCriteria(
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)
    )
    svm.trainAuto(samples, cv2.ml.ROW_SAMPLE, label_array)

    # 自检：置信度重构 argmax 必须与 predict 一致，否则说明 1v1 符号约定有误。
    from .digit_features import svm_class_scores
    indices = range(0, len(samples), max(1, len(samples) // 100))
    agree = sum(
        int(np.argmax(svm_class_scores(svm, samples[i])) ==
            int(svm.predict(samples[i].reshape(1, -1))[1][0][0]) - 1)
        for i in indices
    )
    checked = len(indices)
    print('置信度自检: argmax(分数) 与 predict 一致率 %d/%d' % (agree, checked))
    if agree == 0:
        raise SystemExit('置信度重构与 predict 完全不一致，请检查 svm_class_scores 符号约定。')

    # 留出集评估：干净图 + 带反光/断裂的困难变体。
    def evaluate(items, hard: bool) -> Tuple[int, Dict[int, List[bool]]]:
        total = 0
        correct: Dict[int, List[bool]] = {digit: [] for digit in digit_by_index}
        for digit, path in items:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            masks = (
                augment_sample(gray, 10, rng)
                if hard
                else [preprocess_sample(gray)]
            )
            for mask in masks:
                if mask is None:
                    continue
                prediction = int(svm.predict(
                    hog_features(hog, mask).reshape(1, -1)
                )[1][0][0])
                total += 1
                correct[digit].append(prediction == digit)
        return total, correct

    for hard in (False, True):
        total, correct = evaluate(test_items, hard)
        if total == 0:
            print('%s评估: 无有效样本' % ('困难变体' if hard else '干净'))
            continue
        accuracy = sum(sum(values) for values in correct.values()) / total
        tag = '困难变体(反光/断裂)' if hard else '干净'
        print('%s评估: %.1f%% (%d/%d)' % (tag, accuracy * 100.0, sum(
            sum(values) for values in correct.values()
        ), total))
        for digit in digit_by_index:
            values = correct[digit]
            if values:
                print('  数字%d: %.1f%% (%d/%d)' % (
                    digit,
                    100.0 * sum(values) / len(values),
                    sum(values),
                    len(values),
                ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(output_path))
    print('模型已保存: %s' % output_path)


if __name__ == '__main__':
    main()
