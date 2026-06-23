"""
predict.py
==========
Dự đoán giai đoạn hoa hướng dương cho 1 ảnh, dùng model đã train
(outputs/best_model.pkl từ sunflower_classifier.py).

Chỉ cần sửa đường dẫn ảnh ở phần __main__ dưới cùng rồi chạy:
    python predict.py
(không cần gõ thêm gì ở terminal)
"""

import joblib
from sunflower_classifier import extract_features_from_image

OUTPUT_DIR = "outputs"


def predict(image_path):
    model = joblib.load(f"{OUTPUT_DIR}/best_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    le = joblib.load(f"{OUTPUT_DIR}/label_encoder.pkl")

    feat = extract_features_from_image(image_path)
    if feat is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return

    feat_scaled = scaler.transform([feat])
    pred_encoded = model.predict(feat_scaled)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    print("========================================")
    print(f"🌻 Ảnh: {image_path}")
    print(f"➡️  Giai đoạn dự đoán: {pred_label}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat_scaled)[0]
        print("\nXác suất từng class:")
        for cls, p in zip(le.classes_, proba):
            print(f"   {cls:25s}: {p*100:.2f}%")
    return pred_label


if __name__ == "__main__":
    # ===== SỬA ĐƯỜNG DẪN ẢNH Ở ĐÂY =====
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_bud.jpg")
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_early.jpg")
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_early2.jpg")
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image_full.jpg")
    predict(r"D:\SU26\AIL303m\Model3 for Sunflower\test_image.jpg")
