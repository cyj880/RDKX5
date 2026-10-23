"""离线训练物料颜色分类器（Lab 特征 + kNN）。

流程：读 color_dataset/color_{1..6} 样本 → 亮度抖动增强 → Lab 特征
→ kNN 训练 → 留出集评估并打印混淆对 → 保存模型。

用法：
    ros2 run logistics_vision_capture color_trainer
"""

import argparse
from collections import Counter
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .color_features import color_feature_vector, color_name


def augment_bgr(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """亮度/对比度抖动，模拟不同光照。"""
    alpha = rng.uniform(0.7, 1.3)
    beta = rng.uniform(-25.0, 25.0)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def load_dataset(data_root: Path) -> Dict[int, List[Path]]:
    """读 color_dataset/color_{1..6} 目录样本路径，空类别跳过并告警。"""
    dataset: Dict[int, List[Path]] = {}
    for color in range(1, 7):
        directory = data_root / 'color_dataset' / ('color_%d' % color)
        paths = sorted(
            path for path in directory.glob('*')
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        )
        if paths:
            dataset[color] = paths
        else:
            print('警告: 颜色%d(%s) 无样本，跳过该类 (%s)' % (color, color_name(color), directory))
    if not dataset:
        raise SystemExit('没有任何颜色样本: %s' % data_root)
    return dataset


def split_holdout(
    dataset: Dict[int, List[Path]], holdout_ratio: float, rng: random.Random
) -> Tuple[List[Tuple[int, Path]], List[Tuple[int, Path]]]:
    train: List[Tuple[int, Path]] = []
    test: List[Tuple[int, Path]] = []
    for color, paths in sorted(dataset.items()):
        shuffled = list(paths)
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * holdout_ratio)))
        test.extend((color, path) for path in shuffled[:test_count])
        train.extend((color, path) for path in shuffled[test_count:])
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description='训练物料颜色 kNN 分类器')
    parser.add_argument('--data-root', default='~/logistics_vision_data')
    parser.add_argument('--output', default='')
    parser.add_argument('--augment-count', type=int, default=8)
    parser.add_argument('--holdout-ratio', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    output_path = (
        Path(args.output).expanduser() if args.output
        else data_root / 'color_model' / 'colors_knn.yml'
    )
    rng = random.Random(args.seed)

    dataset = load_dataset(data_root)
    train_items, test_items = split_holdout(dataset, args.holdout_ratio, rng)
    print('样本统计: 训练 %d 张, 留出测试 %d 张, 类别 %s' % (
        len(train_items), len(test_items),
        ','.join('%d(%s)' % (c, color_name(c)) for c in sorted(dataset)),
    ))

    def features_of(bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        mask = np.full(lab.shape[:2], 255, dtype=np.uint8)
        return color_feature_vector(lab, mask)

    def build_set(items, augment: bool) -> Tuple[np.ndarray, np.ndarray]:
        feats: List[np.ndarray] = []
        labels: List[int] = []
        for color, path in items:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print('警告: 无法读取 %s，跳过' % path.name)
                continue
            variants = [bgr]
            if augment:
                variants += [
                    augment_bgr(bgr, rng) for _ in range(args.augment_count)
                ]
            for variant in variants:
                feats.append(features_of(variant))
                labels.append(color)
        return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int32)

    train_samples, train_labels = build_set(train_items, augment=True)
    print('训练特征: %d 条 (含亮度增强)' % len(train_samples))

    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(args.k)
    knn.train(train_samples, cv2.ml.ROW_SAMPLE, train_labels)

    # 留出集评估（干净 + 抖动），统计混淆对。
    def evaluate(items, augment: bool) -> Tuple[int, int, Counter]:
        total = correct = 0
        confusion: Counter = Counter()
        for color, path in items:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            variants = [augment_bgr(bgr, rng) for _ in range(5)] if augment else [bgr]
            for variant in variants:
                sample = features_of(variant).reshape(1, -1)
                _, results, _, _ = knn.findNearest(sample, args.k)
                prediction = int(results[0][0])
                total += 1
                if prediction == color:
                    correct += 1
                else:
                    confusion[(color, prediction)] += 1
        return total, correct, confusion

    for tag, augment in (('干净', False), ('抖动(模拟换光照)', True)):
        total, correct, confusion = evaluate(test_items, augment)
        if total:
            print('%s评估: %.1f%% (%d/%d)' % (tag, 100.0 * correct / total, correct, total))
            for (expected, predicted), count in confusion.most_common(5):
                print('  混淆: %d(%s) → %d(%s) ×%d' % (
                    expected, color_name(expected), predicted, color_name(predicted), count,
                ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    storage.write('samples', train_samples)
    storage.write('labels', train_labels.reshape(-1, 1))
    storage.write('k', float(args.k))
    storage.write('features', 'lab_mean_std_6')
    storage.write(
        'classes', ','.join(str(color) for color in sorted(dataset))
    )
    storage.release()
    print('模型已保存: %s' % output_path)


if __name__ == '__main__':
    main()
