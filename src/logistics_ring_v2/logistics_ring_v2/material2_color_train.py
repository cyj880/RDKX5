"""Material2 颜色模型训练: 相对特征 kNN, 读 crop+sidecar 数据集。

与旧包 color_train.py 的差异:
    输入  新数据集 color_dataset_v2: 每样本 = 裁剪 PNG + sidecar JSON
          (bg_ref 由 material2_sample_capture 采集瞬间现算存档)
    特征  相对白背景 Lab 特征(共享 material_features, 与在线节点一致)
    增广  亮度抖动只作用于 L 通道, 且 sidecar 的 μL 同步做相同变换——
          相对特征下 l_ratio 的抖动才与真实光照变化一致; a/b 不抖
          (抖 a/b 需同步抖基准, 相对特征下 d_a/d_b 不变, 无增益)
    输出  colors_knn_v2.yml, features 标记 lab_relative_6

kNN 的"训练"就是存样本(惰性), 无迭代过程, 重训是秒级——现场急救包。

用法:
    python3 -m logistics_ring_v2.material2_color_train
    python3 -m logistics_ring_v2.material2_color_train \
        --data-root ~/logistics_vision_data/color_dataset_v2 \
        --output ~/logistics_vision_data/color_model/colors_knn_v2.yml
"""

import argparse
import json
from pathlib import Path
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .material_features import (
    color_feature_vector_relative,
    color_name,
    FEATURE_TAG,
)


COLOR_IDS = (1, 2, 3, 4, 5, 6)
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def load_dataset(data_root: Path) -> List[dict]:
    """读全部样本: 每项 {color_id, lab(float32 全图), bg_ref, path}。"""
    samples: List[dict] = []
    for color_id in COLOR_IDS:
        color_dir = data_root / ('color_%d' % color_id)
        if not color_dir.is_dir():
            print('警告: 缺少类别目录 %s' % color_dir)
            continue
        for png_path in sorted(color_dir.iterdir()):
            if png_path.suffix.lower() not in IMAGE_EXTS:
                continue
            json_path = png_path.with_suffix('.json')
            if not json_path.is_file():
                print('警告: %s 缺少 sidecar JSON, 跳过(旧版数据集不兼容相对特征)。'
                      % png_path.name)
                continue
            image = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            if image is None:
                print('警告: 图片读取失败, 跳过: %s' % png_path)
                continue
            sidecar = json.loads(json_path.read_text(encoding='utf-8'))
            bg_ref = tuple(float(v) for v in sidecar['bg_ref'])
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            samples.append({
                'color_id': int(sidecar.get('color_id', color_id)),
                'lab': lab,
                'bg_ref': bg_ref,
                'path': png_path,
            })
    return samples


def jitter_brightness(
    lab: np.ndarray, bg_ref: Tuple[float, float, float],
    alpha: float, beta: float,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """亮度抖动: L' = α·L + β, a/b 不变; μL 同步变换保持口径一致。"""
    jittered = lab.copy()
    jittered[:, :, 0] = np.clip(lab[:, :, 0] * alpha + beta, 0, 255)
    mu_l = float(np.clip(bg_ref[0] * alpha + beta, 1.0, 255.0))
    return jittered, (mu_l, bg_ref[1], bg_ref[2])


def _features_of(sample: dict) -> np.ndarray:
    mask = np.full(sample['lab'].shape[:2], 255, dtype=np.uint8)
    return color_feature_vector_relative(sample['lab'], mask, sample['bg_ref'])


def _split_holdout(samples: List[dict], holdout: float, seed: int
                   ) -> Tuple[List[dict], List[dict]]:
    """按类别分层、按"源图"划分留出集(增广副本跟随源图, 防泄漏)。"""
    rng = random.Random(seed)
    train: List[dict] = []
    test: List[dict] = []
    for color_id in COLOR_IDS:
        group = [s for s in samples if s['color_id'] == color_id]
        if not group:
            continue
        rng.shuffle(group)
        cut = max(1, int(round(len(group) * (1.0 - holdout))))
        if len(group) - cut < 1:
            cut = len(group) - 1  # 至少留 1 张做验证
        train.extend(group[:cut])
        test.extend(group[cut:])
    return train, test


def train(args: argparse.Namespace) -> Optional[int]:
    data_root = Path(args.data_root).expanduser()
    output = Path(args.output).expanduser()
    samples = load_dataset(data_root)
    if not samples:
        print('错误: 数据集为空 (%s)。先用 material2_sample_capture 采集。' % data_root)
        return 1

    per_class = {cid: sum(1 for s in samples if s['color_id'] == cid)
                 for cid in COLOR_IDS}
    print('样本统计: %s' % '  '.join(
        '%s(%d):%d' % (color_name(cid), cid, n) for cid, n in per_class.items()))
    if any(n == 0 for n in per_class.values()):
        print('错误: 存在零样本类别(黑5 若无样本可不采, 在线按定义判黑);'
              '彩色类别缺失将无法判色。')
        missing = [cid for cid, n in per_class.items() if n == 0]
        if any(cid != 5 for cid in missing):
            return 1

    train_samples, test_samples = _split_holdout(samples, args.holdout, args.seed)
    rng = random.Random(args.seed + 1)
    train_vectors: List[np.ndarray] = []
    train_labels: List[int] = []
    for sample in train_samples:
        train_vectors.append(_features_of(sample))
        train_labels.append(sample['color_id'])
        for _ in range(args.augment):
            alpha = rng.uniform(0.75, 1.25)
            beta = rng.uniform(-20.0, 20.0)
            lab_j, bg_j = jitter_brightness(sample['lab'], sample['bg_ref'], alpha, beta)
            mask = np.full(lab_j.shape[:2], 255, dtype=np.uint8)
            train_vectors.append(color_feature_vector_relative(lab_j, mask, bg_j))
            train_labels.append(sample['color_id'])

    train_matrix = np.asarray(train_vectors, dtype=np.float32)
    label_matrix = np.asarray(train_labels, dtype=np.int32).reshape(-1, 1)
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(args.k)
    knn.train(train_matrix, cv2.ml.ROW_SAMPLE, label_matrix)

    # 留出集评估(干净特征, 不增广)
    correct = 0
    confusion: dict = {}
    for sample in test_samples:
        feature = _features_of(sample).reshape(1, -1)
        _, results, _, _ = knn.findNearest(feature, args.k)
        predicted = int(results[0][0])
        actual = sample['color_id']
        key = (actual, predicted)
        confusion[key] = confusion.get(key, 0) + 1
        correct += int(predicted == actual)
    print('训练集: %d 样本(含 %d 倍增广), 留出集: %d 张, 准确率 %d/%d = %.1f%%'
          % (len(train_vectors), args.augment + 1, len(test_samples),
             correct, len(test_samples),
             100.0 * correct / max(1, len(test_samples))))
    wrong = {key: n for key, n in confusion.items() if key[0] != key[1]}
    if wrong:
        print('混淆对(实际→误判为: 次数):')
        for (actual, predicted), n in sorted(wrong.items()):
            print('  %s(%d) → %s(%d): %d'
                  % (color_name(actual), actual, color_name(predicted), predicted, n))
    else:
        print('留出集无混淆。')

    output.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(output), cv2.FILE_STORAGE_WRITE)
    storage.write('samples', train_matrix)
    storage.write('labels', label_matrix)
    storage.write('k', float(args.k))
    storage.write('features', FEATURE_TAG)
    storage.write('classes', ','.join(str(c) for c in COLOR_IDS))
    storage.release()
    print('模型已保存: %s (features=%s, k=%d)' % (output, FEATURE_TAG, args.k))
    print('重启 material2_detect 后生效; 现场判色失配时重采+重跑本脚本即可。')
    return 0


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data-root', default='~/logistics_vision_data/color_dataset_v2',
                        help='样本数据集根目录(color_{1..6}, PNG+sidecar JSON)')
    parser.add_argument('--output', default='~/logistics_vision_data/color_model/colors_knn_v2.yml',
                        help='输出模型路径')
    parser.add_argument('--k', type=int, default=3, help='kNN 的 k')
    parser.add_argument('--augment', type=int, default=4, help='每张训练图亮度抖动副本数')
    parser.add_argument('--holdout', type=float, default=0.2, help='留出集比例')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    raise SystemExit(train(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
