from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Example:
# python .\mp4_to_raw16_dump.py --video .\grayscale_test_video.mp4 --out_dir .\raw16_frames --preview_dir .\previews


def to_gray_u8(frame: np.ndarray) -> np.ndarray:
    """Ensure grayscale uint8."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.uint8)


def u8_to_u16_full_range(img_u8: np.ndarray) -> np.ndarray:
    """Map 0..255 -> 0..65535 via *257 (255*257=65535)."""
    return (img_u8.astype(np.uint16) * 257).astype(np.uint16)


def save_preview_png(img_u16: np.ndarray, out_png: Path) -> None:
    """
    Save preview PNG for quick check.
    We downscale to 8-bit for PNG viewing: u16 -> u8 by /257.
    """
    img_u8 = (img_u16 // 257).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_png), img_u8)
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input mp4 path")
    parser.add_argument("--out_dir", type=str, default="raw16_frames", help="Output folder")
    parser.add_argument("--prefix", type=str, default="frame_", help="RAW filename prefix")
    parser.add_argument("--digits", type=int, default=4, help="Zero padding digits")
    parser.add_argument("--preview_dir", type=str, default="previews", help="Preview PNG output folder")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    preview_dir = Path(args.preview_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nframe_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_paths: list[Path] = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray8 = to_gray_u8(frame)
        img16 = u8_to_u16_full_range(gray8)

        raw_path = out_dir / f"{args.prefix}{idx:0{args.digits}d}.raw"
        img16.tofile(str(raw_path))
        raw_paths.append(raw_path)

        idx += 1

    cap.release()

    if idx == 0:
        raise RuntimeError("No frames were read from the video.")

    # Save metadata (recommended)
    meta = {
        "source_video": str(video_path.name),
        "width": width,
        "height": height,
        "dtype": "uint16",
        "endianness": "little",
        "scaling": "v16 = v8 * 257 (uint8->uint16 full range)",
        "fps": fps,
        "frame_count": idx,
        "frame_count_hint_from_container": nframe_hint,
        "raw_folder": str(out_dir),
        "filename_pattern": f"{args.prefix}{{index:0{args.digits}d}}.raw",
        "bytes_per_frame": width * height * 2,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Pick first/middle/last and save preview PNGs
    first_i = 0
    mid_i = idx // 2
    last_i = idx - 1

    def load_raw16(path: Path) -> np.ndarray:
        arr = np.fromfile(str(path), dtype=np.uint16)
        if arr.size != width * height:
            raise ValueError(f"Size mismatch: {path} has {arr.size} pixels, expected {width*height}")
        return arr.reshape((height, width))

    picks = [first_i, mid_i, last_i]
    for pi in picks:
        rp = raw_paths[pi]
        img16 = load_raw16(rp)
        png_path = preview_dir / f"{rp.stem}.png"
        save_preview_png(img16, png_path)

    print("=== Done ===")
    print(f"Input video : {video_path}")
    print(f"RAW out dir : {out_dir} (frames={idx})")
    print(f"Preview PNG : {preview_dir}")
    print("Saved previews:")
    print(f"  {raw_paths[first_i].name} -> { (preview_dir / (raw_paths[first_i].stem + '.png')).name }")
    print(f"  {raw_paths[mid_i].name}   -> { (preview_dir / (raw_paths[mid_i].stem + '.png')).name }")
    print(f"  {raw_paths[last_i].name}  -> { (preview_dir / (raw_paths[last_i].stem + '.png')).name }")


if __name__ == "__main__":
    main()
