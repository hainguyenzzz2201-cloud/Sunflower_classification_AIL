"""
compute_groups.py
====================
Gom các ảnh trùng / gần trùng nhau thành các "group" (dùng Union-Find trên
perceptual hash). Output: groups.csv với cột (filepath, label, group_id).

File này CHẠY TRƯỚC sunflower_classifier.py (bản đã patch group-aware).
Mục đích: đảm bảo train_test_split không bao giờ chia 2 ảnh trùng/gần trùng
vào cả train và test cùng lúc -> loại bỏ leakage.

Cách dùng:
    cd "D:\\SU26\\AIL303m\\Model3 for Sunflower"
    python compute_groups.py
"""

import os
import imagehash
from PIL import Image
import pandas as pd

CLASS_NAMES = ["Stage1_Young_Bud", "Stage2_Early_Bloom", "Stage3_Full_Bloom"]
DATASET_DIR = "."
NEAR_DUP_THRESHOLD = 8  # cùng ngưỡng đã dùng ở check_duplicates.py
OUTPUT_CSV = "groups.csv"


class UnionFind:
    """Cấu trúc Union-Find để gom các ảnh liên thông (A trùng B, B trùng C => A,B,C cùng group)."""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def main():
    print("=" * 70)
    print("BƯỚC 1: TÍNH HASH CHO TOÀN BỘ ẢNH")
    print("=" * 70)
    records = []  # (label, filepath, hash)
    for label in CLASS_NAMES:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            print(f"⚠️  Không tìm thấy thư mục: {folder}")
            continue
        image_files = [f for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        for fname in image_files:
            fpath = os.path.join(folder, fname)
            try:
                with Image.open(fpath) as img:
                    h = imagehash.phash(img, hash_size=16)
                records.append((label, fpath, h))
            except Exception as e:
                print(f"❌ Lỗi đọc {fpath}: {e}")

    n = len(records)
    print(f"✅ Tổng số ảnh: {n}")

    print("\n" + "=" * 70)
    print(f"BƯỚC 2: GOM NHÓM ẢNH TRÙNG/GẦN TRÙNG (threshold={NEAR_DUP_THRESHOLD})")
    print("=" * 70)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            # Chỉ gom nhóm nếu CÙNG class (đã xác nhận không có cặp khác class nào)
            if records[i][0] != records[j][0]:
                continue
            dist = records[i][2] - records[j][2]
            if dist <= NEAR_DUP_THRESHOLD:
                uf.union(i, j)

    # Gán group_id liên tục (0, 1, 2, ...)
    root_to_group = {}
    rows = []
    for i, (label, fpath, h) in enumerate(records):
        root = uf.find(i)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        group_id = root_to_group[root]
        rows.append({"filepath": fpath, "label": label, "group_id": group_id})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    n_groups = df["group_id"].nunique()
    print(f"✅ Tổng số ảnh: {n}")
    print(f"✅ Tổng số GROUP (sau khi gom trùng/gần trùng): {n_groups}")
    print(f"✅ Số ảnh bị gom chung group với ảnh khác: {n - n_groups} ảnh "
          f"(tức là có ít nhất 1 ảnh khác giống nó)")
    print(f"💾 Đã lưu: {OUTPUT_CSV}")
    print("\n👉 Bước tiếp theo: chạy sunflower_classifier.py (bản đã patch group-aware split)")


if __name__ == "__main__":
    main()
