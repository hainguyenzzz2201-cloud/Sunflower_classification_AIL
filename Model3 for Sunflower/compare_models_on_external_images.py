"""
compare_models_on_external_images.py
======================================
Train lại đủ 7 model (không chỉ best model) trên CÙNG train/test split
group-aware đã dùng ở sunflower_classifier_grouped.py, sau đó cho TẤT CẢ
7 model dự đoán trên các ảnh ngoài dataset (internet) để xem model nào
tổng quát hoá tốt nhất trong thực tế.

YÊU CẦU: đặt file này CÙNG THƯ MỤC với sunflower_classifier_grouped.py
và đã chạy compute_groups.py trước đó (cần file groups.csv).

Cách dùng:
    cd "D:\\SU26\\AIL303m\\Model3 for Sunflower"
    python compare_models_on_external_images.py
"""

import os
import numpy as np
import pandas as pd

# Import lại toàn bộ hàm từ file train chính (không chạy main() vì có if __name__ guard)
from sunflower_classifier_grouped import (
    build_dataset, extract_features_from_image, load_groups, get_models,
    DATASET_DIR, CLASS_NAMES, RANDOM_STATE
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =====================================================================
# CONFIG: điền đường dẫn ảnh ngoài dataset + nhãn thật (nếu biết) để tính accuracy
# Để trống "" ở true_label nếu không biết chắc nhãn thật (script vẫn in dự đoán, chỉ không tính đúng/sai)
# =====================================================================
EXTERNAL_IMAGES = [
    {"path": r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_bud.jpg",    "true_label": "Stage1_Young_Bud"},
    {"path": r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_early.jpg",  "true_label": "Stage2_Early_Bloom"},
    {"path": r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_early3.jpg", "true_label": "Stage2_Early_Bloom"},
    {"path": r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_full.jpg",   "true_label": "Stage3_Full_Bloom"},
    {"path": r"D:\SU26\AIL303m\Model3 for Sunflower\test_image.jpg",        "true_label": ""},  # chưa rõ nhãn thật
    # test_image_early2.jpg lấy từ dataset gốc -> KHÔNG đưa vào đây vì không phải bài test tổng quát hoá thật
]


def main():
    print("=" * 80)
    print("BƯỚC 1: TÁI TẠO LẠI TRAIN/TEST SPLIT GIỐNG sunflower_classifier_grouped.py")
    print("=" * 80)
    X, y, paths = build_dataset(DATASET_DIR, CLASS_NAMES)
    print(f"✅ Tổng số ảnh dataset: {len(X)} | Số chiều feature: {X.shape[1]}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    groups = load_groups(paths)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train, y_train = X[train_idx], y_encoded[train_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"✅ Train set: {len(X_train)} ảnh")

    # ===================================================================
    # BƯỚC 2: TRAIN LẠI TẤT CẢ 7 MODEL TRÊN TRAIN SET
    # ===================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 2: TRAIN LẠI TOÀN BỘ 7 MODEL")
    print("=" * 80)
    models = get_models()
    trained_models = {}
    for name, model in models.items():
        print(f"  Đang train: {name} ...")
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    print(f"✅ Đã train xong {len(trained_models)} model.")

    # ===================================================================
    # BƯỚC 3: TRÍCH XUẤT FEATURE CHO ẢNH NGOÀI DATASET
    # ===================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 3: DỰ ĐOÁN TRÊN ẢNH NGOÀI DATASET (test tổng quát hoá thật)")
    print("=" * 80)

    rows = []
    for item in EXTERNAL_IMAGES:
        img_path = item["path"]
        true_label = item["true_label"]

        if not os.path.exists(img_path):
            print(f"⚠️  Không tìm thấy file: {img_path} -> bỏ qua")
            continue

        feat = extract_features_from_image(img_path)
        if feat is None:
            print(f"⚠️  Không đọc được ảnh: {img_path} -> bỏ qua")
            continue

        feat_scaled = scaler.transform(feat.reshape(1, -1))

        print(f"\n🌻 Ảnh: {os.path.basename(img_path)}"
              + (f"  (nhãn thật: {true_label})" if true_label else "  (chưa rõ nhãn thật)"))

        row = {"image": os.path.basename(img_path), "true_label": true_label}
        for name, model in trained_models.items():
            pred_encoded = model.predict(feat_scaled)[0]
            pred_label = le.inverse_transform([pred_encoded])[0]
            is_correct = (pred_label == true_label) if true_label else None
            row[name] = pred_label

            mark = ""
            if true_label:
                mark = " ✅" if is_correct else " ❌"
            print(f"   {name:22s} -> {pred_label}{mark}")

        rows.append(row)

    # ===================================================================
    # BƯỚC 4: TỔNG HỢP BẢNG SO SÁNH + TÍNH ACCURACY TỪNG MODEL
    # ===================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 4: BẢNG TỔNG HỢP DỰ ĐOÁN CỦA TỪNG MODEL")
    print("=" * 80)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    labeled_rows = df[df["true_label"] != ""]
    if len(labeled_rows) > 0:
        print("\n" + "=" * 80)
        print(f"BƯỚC 5: ĐỘ CHÍNH XÁC TRÊN {len(labeled_rows)} ẢNH NGOÀI DATASET CÓ NHÃN THẬT")
        print("=" * 80)
        acc_results = []
        for name in trained_models.keys():
            correct = (labeled_rows[name] == labeled_rows["true_label"]).sum()
            total = len(labeled_rows)
            acc_results.append({"Model": name, "Đúng": correct, "Tổng": total,
                                 "Accuracy trên ảnh ngoài": correct / total})
        acc_df = pd.DataFrame(acc_results).sort_values("Accuracy trên ảnh ngoài", ascending=False)
        print(acc_df.to_string(index=False))
        print("\n💡 So sánh với Test Accuracy trên dataset nội bộ (model_comparison.csv) để")
        print("   thấy model nào có khoảng cách nhỏ nhất -> tổng quát hoá tốt nhất trong thực tế.")

    # Lưu kết quả ra CSV để đối chiếu
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/external_test_predictions.csv", index=False)
    print(f"\n💾 Đã lưu bảng dự đoán chi tiết vào: outputs/external_test_predictions.csv")


if __name__ == "__main__":
    main()
