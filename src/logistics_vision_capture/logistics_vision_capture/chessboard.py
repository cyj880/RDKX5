"""Shared chessboard detection helpers."""

from typing import Optional, Tuple

import cv2
import numpy as np


def find_chessboard_corners(
    gray_image: np.ndarray,
    pattern_size: Tuple[int, int],
) -> Tuple[bool, Optional[np.ndarray]]:
    """Find all inner chessboard corners and refine them when supported."""
    if hasattr(cv2, 'findChessboardCornersSB'):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray_image, pattern_size, flags)
        if found:
            return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_image, pattern_size, flags)
    if not found:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(
        gray_image,
        corners,
        (11, 11),
        (-1, -1),
        criteria,
    )
    return True, refined
