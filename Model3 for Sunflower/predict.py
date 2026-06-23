"""
predict.py
==========
Dự đoán giai đoạn hoa hướng dương cho 1 ảnh, dùng model đã train
(outputs/best_model.pkl từ sunflower_classifier_grouped.py).

⚠️  LƯU Ý (v2 Pipeline):
    best_model.pkl bây giờ là một sklearn Pipeline gồm:
        StandardScaler → PCA → Classifier
    Bạn KHÔNG cần load scaler.pkl riêng nữa.
    Chỉ cần gọi model.predict(raw_feature) là đủ.

Chỉ cần sửa đường dẫn ảnh ở phần __main__ dưới cùng rồi chạy:
    python predict.py
"""

import joblib
from sunflower_classifier_grouped import extract_features_from_image

OUTPUT_DIR = "outputs"


def predict(image_path):
    # Pipeline đã tích hợp StandardScaler + PCA bên trong
    # → không cần load scaler.pkl hay transform thủ công
    model = joblib.load(f"{OUTPUT_DIR}/best_model.pkl")
    le    = joblib.load(f"{OUTPUT_DIR}/label_encoder.pkl")

    feat = extract_features_from_image(image_path)
    if feat is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return

    # Pipeline tự scale + PCA, truyền raw feature trực tiếp
    pred_encoded = model.predict([feat])[0]
    pred_label   = le.inverse_transform([pred_encoded])[0]

    print("========================================")
    print(f"🌻 Ảnh: {image_path}")
    print(f"➡️  Giai đoạn dự đoán: {pred_label}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([feat])[0]
        print("\nXác suất từng class:")
        for cls, p in zip(le.classes_, proba):
            print(f"   {cls:25s}: {p*100:.2f}%")
    return pred_label


if __name__ == "__main__":
    # ===== SỬA ĐƯỜNG DẪN ẢNH Ở ĐÂY =====
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_bud.jpg")
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_early.jpg")
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_early2.jpg")
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_early3.jpg")
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_full.jpg")
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image.jpg")
