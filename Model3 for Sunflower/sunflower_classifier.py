#Quân
"""
Sunflower Growth Stage Classifier - Traditional Machine Learning Pipeline
===========================================================================
Phân loại 3 giai đoạn: Young Bud, Early Bloom, Full Bloom
Chỉ dùng Machine Learning truyền thống (KHÔNG dùng Deep Learning)

Pipeline:
1. Load ảnh từ các thư mục (mỗi thư mục = 1 class)
2. Trích xuất đặc trưng: HOG, Color Histogram (HSV), LBP, GLCM,
   tỉ lệ vùng màu đĩa hoa (disk-region color ratio)
3. Train nhiều mô hình: SVM, Random Forest, KNN, Logistic Regression,
   Gradient Boosting, XGBoost, Naive Bayes
4. Đánh giá & so sánh bằng Cross-Validation + Test set
5. Lưu model tốt nhất ra .pkl
"""

import os
import cv2
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# CONFIG - chỉnh các đường dẫn này theo dataset của bạn
# =====================================================================
DATASET_DIR = "."               # 3 thư mục class nằm ngay trong thư mục hiện tại
CLASS_NAMES = ["Stage1_Young_Bud", "Stage2_Early_Bloom", "Stage3_Full_Bloom"]
IMAGE_SIZE = (256, 256)          # resize ảnh về kích thước cố định
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# 1. FEATURE EXTRACTION
# =====================================================================
def extract_color_histogram(image_hsv, bins=(8, 8, 8)):
    """Histogram màu trong không gian HSV — phân biệt sắc độ vàng/xanh/nâu."""
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, bins,
                         [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_hog_features(gray):
    """HOG — đặc trưng hình dạng/cấu trúc cạnh (hình dạng cánh hoa, nụ hoa)."""
    features = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        transform_sqrt=True, feature_vector=True
    )
    return features


def extract_lbp_features(gray, P=24, R=3):
    """Local Binary Pattern — đặc trưng kết cấu (texture) bề mặt cánh hoa."""
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_glcm_features(gray):
    """GLCM — đặc trưng kết cấu thống kê bậc 2 (độ tương phản, độ đồng nhất...)."""
    gray_resized = cv2.resize(gray, (128, 128))
    glcm = graycomatrix(gray_resized, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for p in props:
        feats.extend(graycoprops(glcm, p).flatten())
    return np.array(feats)


def extract_disk_color_ratio(image_hsv):
    """
    Tỉ lệ vùng màu đĩa hoa (nâu/xanh ở giữa) so với vùng cánh hoa (vàng).
    Đặc trưng QUAN TRỌNG để phân biệt Early Bloom vs Full Bloom:
    - Young Bud: ít vàng, chủ yếu xanh lá (đài hoa che kín)
    - Early Bloom: vàng bắt đầu xuất hiện, đĩa hoa còn nhỏ/xanh
    - Full Bloom: vàng chiếm đa số, đĩa hoa nâu/đen lộ rõ và to
    """
    h, s, v = cv2.split(image_hsv)

    # Mask màu vàng (cánh hoa)
    yellow_mask = ((h >= 15) & (h <= 35) & (s > 60) & (v > 60))
    # Mask màu nâu/đen (đĩa hoa nở - hạt)
    brown_mask = ((h >= 5) & (h <= 25) & (s > 30) & (v < 120))
    # Mask màu xanh (lá/đài hoa, nụ chưa nở)
    green_mask = ((h >= 35) & (h <= 85) & (s > 40))

    total = h.size
    yellow_ratio = np.sum(yellow_mask) / total
    brown_ratio = np.sum(brown_mask) / total
    green_ratio = np.sum(green_mask) / total

    return np.array([yellow_ratio, brown_ratio, green_ratio])


def extract_features_from_image(image_path):
    """Tổng hợp toàn bộ feature vector cho 1 ảnh."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_hist = extract_color_histogram(hsv)
    hog_feat = extract_hog_features(gray)
    lbp_feat = extract_lbp_features(gray)
    glcm_feat = extract_glcm_features(gray)
    disk_ratio = extract_disk_color_ratio(hsv)

    return np.concatenate([color_hist, hog_feat, lbp_feat, glcm_feat, disk_ratio])


def build_dataset(dataset_dir, class_names):
    """Quét thư mục, trích xuất feature cho toàn bộ ảnh."""
    X, y, paths = [], [], []
    for label in class_names:
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            print(f"⚠️  Không tìm thấy thư mục: {folder}")
            continue
        image_files = [f for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        print(f"📂 {label}: {len(image_files)} ảnh")
        for fname in image_files:
            fpath = os.path.join(folder, fname)
            feat = extract_features_from_image(fpath)
            if feat is not None:
                X.append(feat)
                y.append(label)
                paths.append(fpath)
    return np.array(X), np.array(y), paths


# =====================================================================
# 2. MODEL DEFINITIONS - các model để so sánh
# =====================================================================
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="mlogloss", use_label_encoder=False
        )
    return models


# Lưới hyperparameter để tinh chỉnh model tốt nhất sau khi chọn ra
PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF)": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.001]},
    "SVM (Linear)": {"C": [0.01, 0.1, 1, 10]},
    "KNN": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
    "Random Forest": {"n_estimators": [200, 300, 500], "max_depth": [None, 10, 20, 30]},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]},
    "Naive Bayes": {},
    "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5, 7]},
}


# =====================================================================
# 3. TRAIN & COMPARE
# =====================================================================
def main():
    print("=" * 70)
    print("BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG TỪ ẢNH")
    print("=" * 70)
    X, y, paths = build_dataset(DATASET_DIR, CLASS_NAMES)
    print(f"\n✅ Tổng số ảnh: {len(X)} | Số chiều feature: {X.shape[1] if len(X) else 0}")

    if len(X) == 0:
        print("❌ Không có dữ liệu. Kiểm tra lại DATASET_DIR và cấu trúc thư mục.")
        return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 70)
    print("BƯỚC 2: TRAIN & SO SÁNH NHIỀU MÔ HÌNH (5-fold Cross-Validation)")
    print("=" * 70)

    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")

        results.append({
            "Model": name,
            "CV Accuracy (mean)": cv_scores.mean(),
            "CV Accuracy (std)": cv_scores.std(),
            "Test Accuracy": test_acc,
            "Test F1 (macro)": test_f1,
        })
        print(f"  {name:22s} | CV Acc: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
              f"| Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    results_df = pd.DataFrame(results).sort_values("Test F1 (macro)", ascending=False).reset_index(drop=True)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    print("\n" + "=" * 70)
    print("BẢNG XẾP HẠNG MODEL (theo Test F1-macro)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # ===================================================================
    # BƯỚC 3: TINH CHỈNH HYPERPARAMETER CHO MODEL TỐT NHẤT
    # ===================================================================
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n🏆 Model tốt nhất (trước tuning): {best_model_name}")

    print("\n" + "=" * 70)
    print(f"BƯỚC 3: GRIDSEARCHCV TINH CHỈNH '{best_model_name}'")
    print("=" * 70)

    base_model = get_models()[best_model_name]
    param_grid = PARAM_GRIDS.get(best_model_name, {})

    if param_grid:
        grid = GridSearchCV(base_model, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        print(f"✅ Best params: {grid.best_params_}")
        print(f"✅ Best CV F1-macro: {grid.best_score_:.4f}")
    else:
        best_model = base_model
        best_model.fit(X_train_scaled, y_train)

    y_pred_final = best_model.predict(X_test_scaled)
    final_acc = accuracy_score(y_test, y_pred_final)
    final_f1 = f1_score(y_test, y_pred_final, average="macro")

    print(f"\n📊 KẾT QUẢ CUỐI CÙNG trên Test set:")
    print(f"   Accuracy: {final_acc:.4f}")
    print(f"   F1-macro: {final_f1:.4f}")
    print("\n" + classification_report(y_test, y_pred_final, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {best_model_name} (tuned)")
    plt.ylabel("Thực tế")
    plt.xlabel("Dự đoán")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_best_model.png"), dpi=150)
    plt.close()

    # Biểu đồ so sánh các model
    plt.figure(figsize=(9, 5))
    plot_df = results_df.sort_values("Test F1 (macro)")
    plt.barh(plot_df["Model"], plot_df["Test F1 (macro)"], color="#d4a017")
    plt.xlabel("Test F1 (macro)")
    plt.title("So sánh các mô hình Machine Learning")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_chart.png"), dpi=150)
    plt.close()

    # ===================================================================
    # LƯU MODEL + SCALER + LABEL ENCODER
    # ===================================================================
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    print(f"\n💾 Đã lưu model tốt nhất ({best_model_name}) vào: {OUTPUT_DIR}/best_model.pkl")
    print(f"💾 Đã lưu scaler vào: {OUTPUT_DIR}/scaler.pkl")
    print(f"💾 Đã lưu label encoder vào: {OUTPUT_DIR}/label_encoder.pkl")
    print(f"📈 Biểu đồ và bảng so sánh nằm trong thư mục: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
