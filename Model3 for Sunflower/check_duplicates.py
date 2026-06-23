"""
check_duplicates.py
====================
Kiểm tra ảnh trùng hoặc gần trùng (near-duplicate) trong dataset hoa hướng dương.
Mục đích: phát hiện nguyên nhân khiến Test Accuracy = 1.0 có thể do leakage
(ảnh giống nhau bị chia vào cả train và test sau train_test_split ngẫu nhiên).

Cách dùng:
    cd "D:\\SU26\\AIL303m\\Model3 for Sunflower"
    python check_duplicates.py

Yêu cầu: pip install imagehash pillow
"""

import os
import imagehash
from PIL import Image
from collections import defaultdict

# =====================================================================
CLASS_NAMES = ["Stage1_Young_Bud", "Stage2_Early_Bloom", "Stage3_Full_Bloom"]
DATASET_DIR = "."

# Ngưỡng Hamming distance cho perceptual hash (phash):
#   0      = giống tuyệt đối (trùng pixel-level sau resize hash)
#   1-5    = rất giống nhau (gần như chắc chắn là biến thể / leakage)
#   6-10   = giống một phần (đáng nghi, nên xem lại bằng mắt)
NEAR_DUP_THRESHOLD = 8


def compute_hashes(dataset_dir, class_names):
    """Tính phash cho toàn bộ ảnh, trả về list (label, filepath, hash)."""
    records = []
    for label in class_names:
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            print(f"⚠️  Không tìm thấy thư mục: {folder}")
            continue
        image_files = [f for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        for fname in image_files:
            fpath = os.path.join(folder, fname)
            try:
                with Image.open(fpath) as img:
                    h = imagehash.phash(img, hash_size=16)  # hash_size lớn hơn = nhạy hơn
                records.append((label, fpath, h))
            except Exception as e:
                print(f"❌ Lỗi đọc {fpath}: {e}")
    return records


def find_exact_duplicates(records):
    """Tìm ảnh có hash giống tuyệt đối (distance = 0)."""
    groups = defaultdict(list)
    for label, fpath, h in records:
        groups[str(h)].append((label, fpath))
    return {h: items for h, items in groups.items() if len(items) > 1}


def find_near_duplicates(records, threshold):
    """Tìm các cặp ảnh có khoảng cách Hamming <= threshold (O(n^2), ổn với vài trăm-vài nghìn ảnh)."""
    pairs = []
    n = len(records)
    for i in range(n):
        label_i, path_i, hash_i = records[i]
        for j in range(i + 1, n):
            label_j, path_j, hash_j = records[j]
            dist = hash_i - hash_j  # Hamming distance
            if dist <= threshold:
                pairs.append((dist, label_i, path_i, label_j, path_j))
    pairs.sort(key=lambda x: x[0])
    return pairs


def main():
    print("=" * 70)
    print("BƯỚC 1: TÍNH PERCEPTUAL HASH CHO TOÀN BỘ ẢNH")
    print("=" * 70)
    records = compute_hashes(DATASET_DIR, CLASS_NAMES)
    print(f"✅ Tổng số ảnh đã hash: {len(records)}")

    if len(records) == 0:
        print("❌ Không có ảnh nào. Kiểm tra lại đường dẫn.")
        return

    print("\n" + "=" * 70)
    print("BƯỚC 2: TÌM ẢNH TRÙNG TUYỆT ĐỐI (hash giống 100%)")
    print("=" * 70)
    exact_dups = find_exact_duplicates(records)
    if exact_dups:
        print(f"🔴 Phát hiện {len(exact_dups)} nhóm ảnh TRÙNG TUYỆT ĐỐI:")
        for h, items in exact_dups.items():
            print(f"\n  Hash {h}:")
            for label, fpath in items:
                print(f"    - [{label}] {fpath}")
    else:
        print("✅ Không có ảnh trùng tuyệt đối.")

    print("\n" + "=" * 70)
    print(f"BƯỚC 3: TÌM ẢNH GẦN GIỐNG NHAU (Hamming distance <= {NEAR_DUP_THRESHOLD})")
    print("=" * 70)
    near_dups = find_near_duplicates(records, NEAR_DUP_THRESHOLD)

    if near_dups:
        print(f"🟠 Phát hiện {len(near_dups)} cặp ảnh GẦN GIỐNG NHAU:\n")
        same_class_count = 0
        diff_class_count = 0
        for dist, label_i, path_i, label_j, path_j in near_dups:
            tag = "⚠️ KHÁC CLASS" if label_i != label_j else "(cùng class)"
            if label_i != label_j:
                diff_class_count += 1
            else:
                same_class_count += 1
            print(f"  Distance={dist:2d} {tag}")
            print(f"    [{label_i}] {path_i}")
            print(f"    [{label_j}] {path_j}")
            print()

        print("-" * 70)
        print(f"📊 Tổng kết: {same_class_count} cặp cùng class, "
              f"{diff_class_count} cặp khác class (nguy hiểm hơn vì gây nhiễu nhãn)")
        print("\n💡 Ý nghĩa:")
        print("   - Cặp ảnh gần giống nhau CÙNG class: nếu 1 ảnh rơi vào train,")
        print("     1 ảnh rơi vào test -> đây chính là LEAKAGE, làm Test Accuracy ảo cao.")
        print("   - Cặp ảnh gần giống nhau KHÁC class: có thể do gán nhãn nhầm,")
        print("     hoặc ảnh ở giai đoạn chuyển tiếp khó phân biệt -> nên xem lại bằng mắt.")
    else:
        print(f"✅ Không có ảnh nào gần giống nhau (threshold={NEAR_DUP_THRESHOLD}).")
        print("   => Test Accuracy cao (vd: Random Forest = 1.0) nhiều khả năng là THẬT,")
        print("   do đặc trưng disk-color-ratio phân biệt rất rõ giữa 3 giai đoạn,")
        print("   không phải do leakage giữa train/test.")

    print("\n" + "=" * 70)
    print("HOÀN TẤT")
    print("=" * 70)


if __name__ == "__main__":
    main()
