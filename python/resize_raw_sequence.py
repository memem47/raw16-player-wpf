#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resize_raw_sequence.py

- Reads a sequence of raw grayscale frames (uint16, little-endian by default)
- Resizes each frame to a target size
- Saves as raw binary (same dtype) with configurable extension (default: .bin)

Default behavior matches:
  in : sample\grayscale_test_video\
       frame_0000.raw ... frame_0149.raw (512x512, uint16)
  out: sample\grayscale_test_video_768x768\
       frame_0000.bin ... frame_0149.bin (768x768, uint16)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as e:
    raise SystemExit(
        "OpenCV (cv2) が必要です。以下でインストールしてください:\n"
        "  pip install opencv-python\n"
    ) from e


def load_raw_u16(path: Path, width: int, height: int, little_endian: bool = True) -> np.ndarray:
    """Read uint16 raw and reshape to (H, W)."""
    dt = np.dtype("<u2") if little_endian else np.dtype(">u2")
    expected_elems = width * height

    data = np.fromfile(str(path), dtype=dt)
    if data.size != expected_elems:
        raise ValueError(f"Size mismatch: {path.name} has {data.size} uint16 elems, expected {expected_elems}")

    return data.reshape((height, width))


def save_raw_u16(path: Path, img: np.ndarray, little_endian: bool = True) -> None:
    """Write uint16 array as raw binary."""
    if img.dtype != np.uint16:
        img = img.astype(np.uint16, copy=False)

    if little_endian:
        img.astype(np.dtype("<u2"), copy=False).tofile(str(path))
    else:
        img.astype(np.dtype(">u2"), copy=False).tofile(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Resize a sequence of uint16 RAW frames and save as .bin (or other ext).")

    parser.add_argument(
        "--in_dir",
        type=str,
        default=r"sample\grayscale_test_video",
        help="Input folder containing frame_*.raw",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"sample\grayscale_test_video_768x768",
        help="Output folder",
    )
    parser.add_argument("--in_ext", type=str, default=".raw", help="Input extension (default: .raw)")
    parser.add_argument("--out_ext", type=str, default=".bin", help="Output extension (default: .bin)")

    parser.add_argument("--src_w", type=int, default=512, help="Source width")
    parser.add_argument("--src_h", type=int, default=512, help="Source height")
    parser.add_argument("--dst_w", type=int, default=768, help="Destination width")
    parser.add_argument("--dst_h", type=int, default=768, help="Destination height")

    parser.add_argument("--start", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--end", type=int, default=149, help="End frame index inclusive (default: 149)")
    parser.add_argument("--pattern", type=str, default="frame_{:04d}", help="Basename pattern without extension")

    parser.add_argument(
        "--interp",
        type=str,
        default="linear",
        choices=["nearest", "linear", "cubic", "area", "lanczos"],
        help="Resize interpolation (default: linear)",
    )
    parser.add_argument(
        "--big_endian",
        action="store_true",
        help="If set, treat input/output as big-endian uint16 (default: little-endian)",
    )

    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interp = interp_map[args.interp]

    little_endian = not args.big_endian

    for i in range(args.start, args.end + 1):
        base = args.pattern.format(i)
        in_path = in_dir / f"{base}{args.in_ext}"
        out_path = out_dir / f"{base}{args.out_ext}"

        if not in_path.exists():
            print(f"[WARN] missing: {in_path} (skip)")
            continue

        img = load_raw_u16(in_path, args.src_w, args.src_h, little_endian=little_endian)

        # OpenCV resize expects (W,H)
        resized = cv2.resize(img, (args.dst_w, args.dst_h), interpolation=interp)

        # cv2.resize keeps dtype for uint16 in normal cases, but ensure:
        if resized.dtype != np.uint16:
            resized = resized.astype(np.uint16)

        save_raw_u16(out_path, resized, little_endian=little_endian)

        if (i - args.start) % 10 == 0:
            print(f"[OK] {in_path.name} -> {out_path.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
