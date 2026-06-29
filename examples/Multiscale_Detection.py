"""
Multi-scale checkerboard detection test.

Three detection runs per image:
  1. ORIGINAL   – detect on the image as-is.
  2. DOWNSAMPLED – halve repeatedly until the image is < MAX_W × MAX_H,
                   detect on that, scale corners back up to original space.
  3. UPSAMPLED  – double the original (2×, bilinear), detect on that,
                  scale corners back down to original space.

All three corner sets are drawn on the original image in one window so
spatial quality / drift is immediately visible.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.PyCBD.pipelines import CBDPipeline

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR = r'.\images'
IMAGE_FILES = {
    'broken':   os.path.join(IMAGE_DIR, 'broken.jpg'),
    'charuco':   os.path.join(IMAGE_DIR, 'charuco.png'),
    'flare': os.path.join(IMAGE_DIR, 'flare.jpg'),
    'thermal': os.path.join(IMAGE_DIR, 'thermal.tiff'),
    'warped': os.path.join(IMAGE_DIR, 'warped.jpg'),
}

# Downsample until the image width is below this threshold.
MAX_W = 2000

# Checkerboard inner-corner dimensions (cols, rows).
# Set to None to let the detector figure it out automatically.
BOARD_SIZE = None  # e.g. (9, 14)

# Visual style per variant  (color, marker, label)
STYLE = {
    'original':    ('lime',       'o', 'Original'),
    'downsampled': ('red',        's', 'Downsampled → projected up'),
    'upsampled':   ('deepskyblue','D', 'Upsampled → projected down'),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_downsample_factor(w: int, h: int) -> int:
    """
    Return the integer scale factor (power of 2) needed to bring
    (w, h) below (MAX_W, MAX_H) by repeated halving.
    """
    factor = 1
    cw, ch = w, h
    while cw >= MAX_W:
        cw //= 2
        ch //= 2
        factor *= 2
    return factor


def run_detection(pipeline: CBDPipeline, image: np.ndarray):
    """
    Run detect_checkerboard and return (board_uv, board_xy, error_str).
    board_uv / board_xy are None on failure.
    """
    try:
        if BOARD_SIZE is not None:
            _, board_uv, board_xy = pipeline.detect_checkerboard(image, BOARD_SIZE)
        else:
            _, board_uv, board_xy = pipeline.detect_checkerboard(image)
        return board_uv, board_xy, None
    except Exception as exc:
        return None, None, str(exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pipeline = CBDPipeline(expand=True, predict=True)

    for cam, filepath in IMAGE_FILES.items():
        print(f'\n{"="*64}')
        print(f'  Camera : {cam}   →   {filepath}')
        print('='*64)

        # ── Load ──────────────────────────────────────────────────────────────
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f'  ERROR: could not load "{filepath}" — skipping.')
            continue

        orig_h, orig_w = image.shape[:2]
        print(f'  Original size : {orig_w} × {orig_h}')

        # ── Build the three test images ───────────────────────────────────────

        # 1) Original — untouched
        img_orig = image.copy()

        # 2) Downsampled — halve until below MAX_W × MAX_H
        ds_factor = compute_downsample_factor(orig_w, orig_h)
        ds_w, ds_h = orig_w // ds_factor, orig_h // ds_factor
        img_down = cv2.resize(image, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR_EXACT)
        print(f'  Downsampled   : {ds_w} × {ds_h}  (÷{ds_factor})')

        # 3) Upsampled — double the original
        us_w, us_h = orig_w * 2, orig_h * 2
        img_up = cv2.resize(image, (us_w, us_h), interpolation=cv2.INTER_LINEAR_EXACT)
        print(f'  Upsampled     : {us_w} × {us_h}  (×2)')

        # ── Run detection on each ─────────────────────────────────────────────
        uv_orig, xy_orig, err_orig = run_detection(pipeline, img_orig)
        uv_down, xy_down, err_down = run_detection(pipeline, img_down)
        uv_up,   xy_up,   err_up   = run_detection(pipeline, img_up)

        print(uv_orig.dtype)

        # ── Project all corners back to original coordinate space ─────────────
        #   Downsampled corners were detected on a (÷ds_factor) image → scale up
        if uv_down is not None:
            uv_down_proj = uv_down * ds_factor
        else:
            uv_down_proj = None

        #   Upsampled corners were detected on a (×2) image → scale down
        if uv_up is not None:
            uv_up_proj = uv_up / 2.0
        else:
            uv_up_proj = None

        # ── Print summary ─────────────────────────────────────────────────────
        detections = {
            'original':    (uv_orig,      xy_orig, err_orig),
            'downsampled': (uv_down_proj, xy_down, err_down),
            'upsampled':   (uv_up_proj,   xy_up,   err_up),
        }

        print(f'\n  Detection results:')
        ref_n = None
        for key, (uv, xy, err) in detections.items():
            style_label = STYLE[key][2]
            if err:
                print(f'    {style_label:40s}  ✗ FAILED: {err}')
            elif uv is None or len(uv) == 0:
                print(f'    {style_label:40s}  ⚠ no corners found')
            else:
                n = len(uv)
                match = '' if ref_n is None else (
                    '  ✓ matches' if n == ref_n else f'  ✗ MISMATCH (ref={ref_n})')
                print(f'    {style_label:40s}  ✓ {n} corners{match}')
                if ref_n is None:
                    ref_n = n

        # ── Figure 1: each detection drawn on its own native-res image ──────────
        native = [
            ('original',    img_orig, uv_orig, xy_orig, err_orig),
            ('downsampled', img_down, uv_down, xy_down, err_down),
            ('upsampled',   img_up,   uv_up,   xy_up,   err_up),
        ]

        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
        fig1.suptitle(f'Native-resolution detections  —  {cam}',
                      fontsize=11, fontweight='bold')

        for ax1, (key, img, uv, xy, err) in zip(axes1, native):
            color, marker, label = STYLE[key]
            h_n, w_n = img.shape[:2]
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
            ax1.imshow(disp, cmap='gray' if img.ndim == 2 else None)

            if err is not None:
                ax1.set_title(f'{label}\n{w_n}×{h_n}\n✗ FAILED: {err}',
                              color='red', fontsize=8)
            elif uv is None or len(uv) == 0:
                ax1.set_title(f'{label}\n{w_n}×{h_n}\n⚠ no corners',
                              color='orange', fontsize=8)
            else:
                ax1.plot(uv[:, 0], uv[:, 1],
                         linestyle='-', marker=marker,
                         color=color, markeredgecolor='black',
                         markersize=4, linewidth=0.8, alpha=0.9)
                ax1.annotate(
                    f'({int(xy[0, 0])},{int(xy[0, 1])})',
                    xy=(uv[0, 0], uv[0, 1]),
                    xytext=(4, -10), textcoords='offset points',
                    color=color, fontsize=6, fontweight='bold',
                )
                ax1.set_title(f'{label}\n{w_n}×{h_n}\n✓ {len(uv)} corners',
                              color='green', fontsize=8)
            ax1.axis('off')

        plt.tight_layout()
        plt.show()

        # ── Figure 2: all detections projected onto the original image ────────
        display = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                   if image.ndim == 3 else image)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(display, cmap='gray' if image.ndim == 2 else None)

        legend_handles = []
        for key, (uv, xy, err) in detections.items():
            color, marker, label = STYLE[key]

            if err is not None:
                legend_handles.append(
                    mpatches.Patch(color=color, label=f'{label}  [FAILED]'))
                continue

            if uv is None or len(uv) == 0:
                legend_handles.append(
                    mpatches.Patch(color=color, label=f'{label}  [no corners]'))
                continue

            # Draw the detection path
            ax.plot(uv[:, 0], uv[:, 1],
                    linestyle='-', marker=marker,
                    color=color, markeredgecolor='black',
                    markersize=5, linewidth=1.0,
                    alpha=0.85, zorder=3)

            # Label the first corner with board coordinates
            ax.annotate(
                f'({int(xy[0, 0])},{int(xy[0, 1])})',
                xy=(uv[0, 0], uv[0, 1]),
                xytext=(4, -12), textcoords='offset points',
                color=color, fontsize=6, fontweight='bold',
            )

            n = len(uv)
            handle = mpatches.Patch(color=color, label=f'{label}  [{n} corners]')
            legend_handles.append(handle)

        ax.legend(handles=legend_handles, loc='upper right',
                  fontsize=8, framealpha=0.85)
        ax.set_title(
            f'All detections projected onto original  —  {cam}  ({orig_w}×{orig_h})',
            fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    print('\nDone.')


if __name__ == '__main__':
    main()