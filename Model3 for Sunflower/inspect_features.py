"""
inspect_features.py
====================
Công cụ kiểm tra số liệu thực tế của các feature đã trích xuất trong
sunflower_classifier_grouped.py.

Cách dùng:
  1. Đặt file này CÙNG thư mục với sunflower_classifier_grouped.py
     (để import được các hàm extract_* và CLASS_NAMES, DATASET_DIR).
  2. Chạy:
       python inspect_features.py                 -> tự lấy 1 ảnh mẫu/class
       python inspect_features.py duong/dan/anh.jpg -> xem chi tiết 1 ảnh cụ thể

Script làm 2 việc:
  A. In chi tiết các feature "có tên/dễ đọc" (disk ratio, center vs border,
     edge density, saturation stats, brightness) cho 1 ảnh mẫu mỗi class.
  B. Quét TOÀN BỘ dataset, xuất ra:
       outputs/feature_table_named.csv   -> các feature dễ đọc, có tên cột, theo từng ảnh
       outputs/feature_summary_by_class.csv -> mean/std của từng feature theo từng class
       outputs/feature_boxplots.png      -> boxplot so sánh giữa 3 class
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sunflower_classifier_grouped import (
    preprocess_image,
    extract_disk_color_ratio,
    extract_center_vs_border,
    extract_petal_edge_density,
    extract_saturation_stats,
    extract_brightness_features,
    CLASS_NAMES,
    DATASET_DIR,
    IMAGE_SIZE,
    OUTPUT_DIR,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tên cột cho từng nhóm feature dễ đọc (đúng thứ tự return của từng hàm)
NAMES_DISK   = ["disk_yellow_ratio", "disk_brown_ratio", "disk_green_ratio",
                "disk_dark_ratio", "disk_dark_green_ratio"]
NAMES_CENTER = ["center_brown", "center_yellow", "center_green",
                "border_yellow", "ratio_brown_center_vs_yellow_border"]
NAMES_EDGE   = ["edge_tight_mean", "edge_loose_mean", "edge_tight_std", "log_laplacian_var"]
NAMES_SAT    = ["sat_mean", "sat_std", "sat_p25", "sat_p75",
                "val_mean", "val_std", "val_p25", "val_p75", "sat_x_val_mean"]
NAMES_BRIGHT = ["gray_mean", "gray_std", "gray_p10", "gray_p90",
                "v_mean", "v_std",
                "vhist_b0", "vhist_b1", "vhist_b2", "vhist_b3",
                "vhist_b4", "vhist_b5", "vhist_b6", "vhist_b7"]

ALL_NAMES = NAMES_DISK + NAMES_CENTER + NAMES_EDGE + NAMES_SAT + NAMES_BRIGHT


def extract_named_features(image_path):
    """Trích các feature 'dễ đọc' (không gồm color_hist/HOG/LBP/GLCM) cho 1 ảnh."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, IMAGE_SIZE)
    _, gray, hsv = preprocess_image(image)

    vals = np.concatenate([
        extract_disk_color_ratio(hsv),
        extract_center_vs_border(hsv),
        extract_petal_edge_density(gray),
        extract_saturation_stats(hsv),
        extract_brightness_features(gray, hsv),
    ])
    return vals


def print_one_image(image_path):
    print(f"\n📸 Ảnh: {image_path}")
    vals = extract_named_features(image_path)
    if vals is None:
        print("   ❌ Không đọc được ảnh.")
        return
    for name, v in zip(ALL_NAMES, vals):
        print(f"   {name:35s} = {v:.4f}")


def show_sample_per_class():
    print("=" * 70)
    print("A. XEM CHI TIẾT 1 ẢNH MẪU MỖI CLASS")
    print("=" * 70)
    for label in CLASS_NAMES:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if not files:
            continue
        sample_path = os.path.join(folder, files[0])
        print(f"\n=== Class: {label} ===")
        print_one_image(sample_path)


def build_named_table():
    print("\n" + "=" * 70)
    print("B. QUÉT TOÀN BỘ DATASET -> BẢNG FEATURE CÓ TÊN CỘT")
    print("=" * 70)
    rows = []
    for label in CLASS_NAMES:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        for fname in files:
            fpath = os.path.join(folder, fname)
            vals = extract_named_features(fpath)
            if vals is None:
                continue
            row = {"filepath": fpath, "class": label}
            row.update(dict(zip(ALL_NAMES, vals)))
            rows.append(row)

    if not rows:
        print("❌ Không có ảnh nào được xử lý. Kiểm tra lại DATASET_DIR.")
        return None

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, "feature_table_named.csv")
    df.to_csv(out_csv, index=False)
    print(f"✅ Đã lưu bảng feature từng ảnh: {out_csv}  ({len(df)} ảnh)")

    summary = df.groupby("class")[ALL_NAMES].agg(["mean", "std"])
    out_summary = os.path.join(OUTPUT_DIR, "feature_summary_by_class.csv")
    summary.to_csv(out_summary)
    print(f"✅ Đã lưu mean/std theo từng class: {out_summary}")

    return df


def plot_boxplots(df, max_cols_per_fig=10):
    print("\n" + "=" * 70)
    print("C. VẼ BOXPLOT SO SÁNH GIỮA 3 CLASS")
    print("=" * 70)
    cols = ALL_NAMES
    n_fig = (len(cols) + max_cols_per_fig - 1) // max_cols_per_fig
    for fig_idx in range(n_fig):
        chunk = cols[fig_idx * max_cols_per_fig:(fig_idx + 1) * max_cols_per_fig]
        n = len(chunk)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, chunk):
            sns.boxplot(data=df, x="class", y=col, ax=ax, palette="YlOrBr")
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30, labelsize=7)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"feature_boxplots_{fig_idx+1}.png")
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"✅ Đã lưu: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Xem chi tiết 1 ảnh cụ thể do user truyền vào
        print_one_image(sys.argv[1])
    else:
        show_sample_per_class()
        df = build_named_table()
        if df is not None:
            plot_boxplots(df)
