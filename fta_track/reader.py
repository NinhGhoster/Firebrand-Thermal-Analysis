"""Level 1.1 NetCDF reader: chunk-aligned sequential iteration over float32 degC frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Tuple

import numpy as np


@dataclass
class Recording:
    path: str
    n_frames: int
    height: int
    width: int
    fps: float
    distance_m: float
    emissivity: float
    threshold_c: float
    attrs: dict = field(default_factory=dict)
    _ds: object = None
    _var: object = None
    _chunk: int = 10

    def iter_frames(self, start: int = 0, stop: int | None = None) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_index, frame) in order, reading whole on-disk chunks at a time.

        One frame read decompresses its entire chunk (10 frames in this release), so
        reading chunk-aligned slabs is about ten times cheaper than frame by frame.
        """
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        c = self._chunk
        i = (start // c) * c
        while i < stop:
            j = min(i + c, stop)
            block = np.asarray(self._var[i:j], dtype=np.float32)
            for k in range(block.shape[0]):
                idx = i + k
                if idx >= start:
                    yield idx, block[k]
            i = j

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None


def _attr(ds, name, default):
    return ds.getncattr(name) if name in ds.ncattrs() else default


def open_recording(path: str) -> Recording:
    import netCDF4 as nc

    ds = nc.Dataset(path, "r")
    var = ds.variables["temperature"]
    var.set_auto_maskandscale(False)  # plain float32 degC; no scale/offset in this release
    n, h, w = var.shape
    chunking = var.chunking()
    chunk = int(chunking[0]) if isinstance(chunking, (list, tuple)) else 10
    attrs = {k: _attr(ds, k, None) for k in ds.ncattrs()}
    for k, v in list(attrs.items()):
        if isinstance(v, np.generic):
            attrs[k] = v.item()
    return Recording(
        path=path,
        n_frames=int(n),
        height=int(h),
        width=int(w),
        fps=float(_attr(ds, "frame_rate", 30.0)),
        distance_m=float(_attr(ds, "distance", 2.0)),
        emissivity=float(_attr(ds, "emissivity_original", float("nan"))),
        threshold_c=float(_attr(ds, "threshold_degC", float("nan"))),
        attrs=attrs,
        _ds=ds,
        _var=var,
        _chunk=max(1, chunk),
    )
