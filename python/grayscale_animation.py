import numpy as np
import imageio.v2 as imageio
import math
import os

# =========================
# Parameters
# =========================
W, H = 512, 512
FPS = 30
SECONDS = 5
N_FRAMES = FPS * SECONDS
OUT_PATH = "grayscale_test_512x512.mp4"

CIRCLE_RADIUS = 40
RECT_WIDTH = 60
RECT_HEIGHT = 80
NOISE_SIGMA = 0.02  # Gaussian noise strength (0-1 scale)

# =========================
# Video writer
# =========================
writer = imageio.get_writer(
    OUT_PATH,
    fps=FPS,
    codec="libx264",
    pixelformat="yuv420p",
    quality=8
)

# =========================
# Base gradient (horizontal)
# =========================
x = np.linspace(0, 1, W, dtype=np.float32)
gradient = np.tile(x, (H, 1))  # shape (H, W)

yy, xx = np.mgrid[0:H, 0:W]

# =========================
# Frame loop
# =========================
for t in range(N_FRAMES):
    frame = gradient.copy()

    # ---- Moving circle (left -> right) ----
    cx = int((W + 2 * CIRCLE_RADIUS) * t / (N_FRAMES - 1) - CIRCLE_RADIUS)
    cy = H // 2
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= CIRCLE_RADIUS ** 2
    frame[circle_mask] = 1.0

    # ---- Moving rectangle (right -> left) ----
    rx = int(W - (W + RECT_WIDTH) * t / (N_FRAMES - 1))
    ry = H // 3
    x0 = max(rx, 0)
    x1 = min(rx + RECT_WIDTH, W)
    y0 = max(ry, 0)
    y1 = min(ry + RECT_HEIGHT, H)
    frame[y0:y1, x0:x1] = 0.0

    # ---- Gaussian noise ----
    rng = np.random.default_rng(seed=t)
    noise = rng.normal(0.0, NOISE_SIGMA, size=(H, W)).astype(np.float32)
    frame = np.clip(frame + noise, 0.0, 1.0)

    # ---- Convert to uint8 grayscale ----
    gray8 = (frame * 255).astype(np.uint8)

    # ---- MP4 requires 3 channels (R=G=B) ----
    rgb = np.stack([gray8, gray8, gray8], axis=2)

    writer.append_data(rgb)

writer.close()

print(f"Saved: {OUT_PATH}")
print(f"Resolution: {W}x{H}, FPS: {FPS}, Frames: {N_FRAMES}")
