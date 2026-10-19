"""Persistent-hot mask: the burning sample, the burner and the flame.

A pixel is persistent when it is at or above the hot threshold in at least
`persist_frac` of the frames in a window centred on the frame being analysed.
Burning fuel stays hot at the same pixels for seconds; a flying firebrand
crosses any one pixel in a frame or two. The window is centred, so a pixel that
is about to become part of the flame is already masked when it first lights.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import cv2
import numpy as np


class PersistenceWindow:
    def __init__(self, shape: Tuple[int, int], window: int, frac: float, dilate_px: int):
        self.window = int(window)
        self.half = self.window // 2
        self.frac = float(frac)
        self.counts = np.zeros(shape, dtype=np.uint16)
        self.hot: Deque[Tuple[int, np.ndarray]] = deque()
        k = 2 * int(dilate_px) + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if dilate_px > 0 else None

    def push(self, idx: int, hot: np.ndarray) -> None:
        self.hot.append((idx, hot))
        self.counts += hot

    def mask_for(self, centre: int) -> np.ndarray:
        """Persistent mask for `centre`, using frames centre-half .. centre+half already pushed."""
        while self.hot and self.hot[0][0] < centre - self.half:
            _, old = self.hot.popleft()
            self.counts -= old
        n = len(self.hot)
        need = max(1, int(np.ceil(self.frac * n)))
        mask = (self.counts >= need).astype(np.uint8)
        if self.kernel is not None:
            mask = cv2.dilate(mask, self.kernel)
        return mask.astype(bool)
