#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
raw_sequence_to_mp4.py

Convert a sequence of uint16 RAW/BIN grayscale images into an MP4 video.

Default:
  input : sample/grayscale_test_video_768x768/frame_0000.bin ...
  output: grayscale_test_video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import cv2


def load_raw_u16(path: Path, width: int, height: int, little_endian: bool = True) -> np.ndarray:
    dt = np.dtype("<u2") if little_endian else np.dtype(">u2")
    data = np.fromfile(str(path), dtype=dt)
    if data.size != width * height:
        raise ValueError(f"Size mismatch: {path.name}")
    return data.reshape((height, width))


def normalize_u16_to_u8(img_u16: np.ndarray) -> np.ndarray:
    """
    Convert uint16 image to uint8 using min-max normalization.
    """
    img = img_u16.astype(np.float32)
    minv, maxv = img.min(), img.max()
    if maxv > minv:
        img = (img - minv) / (maxv - minv)
    else:
        img[:] = 0
    return (img * 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert RAW/BIN sequence to MP4")

    parser.add_argument("--in_dir", type=str,
                        default=r"sample/grayscale_test_video_768x768",
                        help="Input folder")
    parser.add_argument("--out_mp4", type=str,
                        default="grayscale_test_video.mp4",
                        help="Output mp4 filename")

    parser.add_argument("--ext", type=str, default=".bin", help="Input extension (.raw or .bin)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)

    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=149)
    parser.add_argument("--pattern", type=str, default="frame_{:04d}")

    parser.add_argument("--fourcc", type=str, default="mp4v",
                        help="FourCC (mp4v, avc1, H264 etc)")
    parser.add_argument("--big_endian", action="store_true")

    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_mp4 = Path(args.out_mp4)

    little_endian = not args.big_endian

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(
        str(out_mp4),
        fourcc,
        args.fps,
        (args.width, args.height),
        isColor=False
    )

    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    for i in range(args.start, args.end + 1):
        name = args.pattern.format(i) + args.ext
        path = in_dir / name

        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue

        img_u16 = load_raw_u16(path, args.width, args.height, little_endian)
        img_u8 = normalize_u16_to_u8(img_u16)

        writer.write(img_u8)

        if (i - args.start) % 10 == 0:
            print(f"[OK] added frame {i}")

    writer.release()
    print(f"Done: {out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
