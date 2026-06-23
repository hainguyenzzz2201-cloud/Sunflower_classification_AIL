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

Fix Overfitting (v2):
- Dùng sklearn Pipeline: StandardScaler → PCA(100) → Model
- PCA giảm chiều HOG từ 8100 → 100 (chống curse of dimensionality)
- RF hyperparameter thắt chặt hơn: max_depth nhỏ hơn, max_samples < 1.0
- Thêm Train Accuracy để phát hiện overfitting
- Thêm Learning Curve để visualize mức độ overfitting
"""

import os
import cv2
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
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

GROUPS_CSV = "groups.csv"   # tạo bằng compute_groups.py - chạy script đó TRƯỚC file này


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


def preprocess_image(image):
    """
    Tiền xử lý chuẩn hoá trước khi trích xuất feature:
    - CLAHE trên kênh V (HSV) để cân bằng độ sáng, giúp feature màu
      nhất quán hơn giữa ảnh tối (Young Bud macro) và ảnh sáng.
    - Gaussian blur nhẹ để giảm noise pixel.
    Trả về: (image_bgr, gray, hsv) sau tiền xử lý.
    """
    # Gaussian blur nhẹ — giảm noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # CLAHE trên kênh V của HSV
    hsv_raw = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_raw[:, :, 2] = clahe.apply(hsv_raw[:, :, 2])
    # Convert lại sang BGR rồi sang gray để thống nhất
    image_eq = cv2.cvtColor(hsv_raw, cv2.COLOR_HSV2BGR)
    gray_eq  = cv2.cvtColor(image_eq, cv2.COLOR_BGR2GRAY)
    hsv_eq   = cv2.cvtColor(image_eq, cv2.COLOR_BGR2HSV)
    return image_eq, gray_eq, hsv_eq


def extract_disk_color_ratio(image_hsv):
    """
    Tỉ lệ vùng màu đĩa hoa (nâu/xanh ở giữa) so với vùng cánh hoa (vàng).
    Đặc trưng QUAN TRỌNG để phân biệt Early Bloom vs Full Bloom:
    - Young Bud: ít vàng, chủ yếu xanh lá (che kín) + xanh đen tối (macro)
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
    # Mask tối (v < 80): đặc trưng của Young Bud chụp macro trong điều kiện bóng tối
    dark_mask = (v < 80)
    # Mask xanh lá tối: Young Bud macro (đen-xanh)
    dark_green_mask = ((h >= 35) & (h <= 85) & (v < 100))

    total = h.size
    yellow_ratio    = np.sum(yellow_mask)     / total
    brown_ratio     = np.sum(brown_mask)      / total
    green_ratio     = np.sum(green_mask)      / total
    dark_ratio      = np.sum(dark_mask)       / total
    dark_green_ratio= np.sum(dark_green_mask) / total

    return np.array([yellow_ratio, brown_ratio, green_ratio,
                     dark_ratio, dark_green_ratio])


def extract_brightness_features(gray, image_hsv):
    """
    Thống kê độ sáng toàn ảnh và phân phối histogram.
    - Young Bud macro: mean thấp (tối), std thấp (màu đồng nhất xanh-đen)
    - Early Bloom   : mean trung bình, màu xanh sáng
    - Full Bloom    : mean cao (vàng tươi), std cao (vàng + nâu tương phản)
    """
    v_channel = image_hsv[:, :, 2].astype(np.float32)
    gray_f    = gray.astype(np.float32)

    # Histogram 8 bin của kênh V — phân phối độ sáng
    hist_v, _ = np.histogram(v_channel, bins=8, range=(0, 256), density=True)

    return np.array([
        gray_f.mean(),                    # độ sáng trung bình
        gray_f.std(),                     # độ tương phản
        np.percentile(gray_f, 10),        # vùng tối
        np.percentile(gray_f, 90),        # vùng sáng
        v_channel.mean(),
        v_channel.std(),
        *hist_v,                          # 8 bin histogram độ sáng
    ])



# ── FEATURE MỚI 1 ────────────────────────────────────────────────────────────
def extract_center_vs_border(image_hsv):
    """
    So sánh màu vùng trung tâm (đĩa hoa) vs vùng ngoài rìa (cánh hoa).
    Đây là đặc trưng quan trọng nhất để phân biệt 3 giai đoạn:
    - Young Bud  : trung tâm xanh, rìa xanh/ít vàng
    - Early Bloom: trung tâm vàng-xanh lẫn lộn, rìa vàng rõ hơn
    - Full Bloom : trung tâm NÂU/ĐEN rõ rệt, rìa vàng tươi
    """
    h_img, w_img = image_hsv.shape[:2]
    cx, cy = w_img // 2, h_img // 2
    r = min(h_img, w_img) // 5   # bán kính ~20% chiều nhỏ hơn

    # Vùng trung tâm
    center = image_hsv[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
    hc, sc, vc = center[:,:,0], center[:,:,1], center[:,:,2]

    # Vùng ngoài rìa (vành khăn)
    mask_border = np.ones((h_img, w_img), dtype=bool)
    mask_border[max(0, cy-r):cy+r, max(0, cx-r):cx+r] = False
    h_full = image_hsv[:,:,0]
    s_full = image_hsv[:,:,1]
    v_full = image_hsv[:,:,2]

    brown_center  = ((hc >= 5)  & (hc <= 25) & (sc > 30) & (vc < 130)).mean()
    yellow_center = ((hc >= 15) & (hc <= 35) & (sc > 50) & (vc > 60)).mean()
    green_center  = ((hc >= 35) & (hc <= 85) & (sc > 40)).mean()
    yellow_border = (
        (h_full[mask_border] >= 15) & (h_full[mask_border] <= 35) &
        (s_full[mask_border] > 50)  & (v_full[mask_border] > 60)
    ).mean()
    ratio_brown_yellow = brown_center / (yellow_border + 1e-6)

    return np.array([brown_center, yellow_center, green_center,
                     yellow_border, ratio_brown_yellow])


# ── FEATURE MỚI 2 ────────────────────────────────────────────────────────────
def extract_petal_edge_density(gray):
    """
    Mật độ cạnh cánh hoa qua Canny edge detection.
    - Young Bud  : ít cạnh (hoa chưa mở, bề mặt trơn)
    - Early Bloom: cạnh trung bình (cánh hoa mới xoè ra)
    - Full Bloom : cạnh nhiều (cánh hoa xoè tối đa, đĩa hoa có hạt rõ)
    """
    edges_tight = cv2.Canny(gray, 80, 160)
    edges_loose = cv2.Canny(gray, 30, 100)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return np.array([
        edges_tight.mean(),
        edges_loose.mean(),
        edges_tight.std(),
        np.log1p(lap_var),
    ])


# ── FEATURE MỚI 3 ────────────────────────────────────────────────────────────
def extract_saturation_stats(image_hsv):
    """
    Thống kê kênh Saturation và Value toàn ảnh.
    - Young Bud  : S thấp (màu nhạt/xanh lá), V trung bình
    - Early Bloom: S trung bình, V cao (vàng tươi nhưng chưa đều)
    - Full Bloom : S cao ở cánh hoa vàng, V thấp ở đĩa nâu → phương sai lớn
    """
    s = image_hsv[:,:,1].astype(np.float32) / 255.0
    v = image_hsv[:,:,2].astype(np.float32) / 255.0
    return np.array([
        s.mean(), s.std(), np.percentile(s, 25), np.percentile(s, 75),
        v.mean(), v.std(), np.percentile(v, 25), np.percentile(v, 75),
        (s * v).mean(),   # S×V cao → màu bão hoà và sáng = cánh hoa vàng tươi
    ])


def extract_features_from_image(image_path):
    """Tổng hợp toàn bộ feature vector cho 1 ảnh (bao gồm các feature mới)."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, IMAGE_SIZE)

    # ── Tiền xử lý (CLAHE + blur) để chuẩn hoá độ sáng ────────────────────
    image_eq, gray, hsv = preprocess_image(image)

    color_hist   = extract_color_histogram(hsv)          # 512 dim
    hog_feat     = extract_hog_features(gray)            # 8100 dim
    lbp_feat     = extract_lbp_features(gray)            # 26 dim
    glcm_feat    = extract_glcm_features(gray)           # 48 dim
    disk_ratio   = extract_disk_color_ratio(hsv)         # 5 dim  (củ 3 + 2 mới)
    center_feat  = extract_center_vs_border(hsv)         # 5 dim
    edge_feat    = extract_petal_edge_density(gray)      # 4 dim
    sat_feat     = extract_saturation_stats(hsv)         # 9 dim
    bright_feat  = extract_brightness_features(gray, hsv) # 14 dim (mới)

    return np.concatenate([color_hist, hog_feat, lbp_feat, glcm_feat,
                           disk_ratio, center_feat, edge_feat,
                           sat_feat, bright_feat])


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
def load_groups(paths, groups_csv=GROUPS_CSV):
    """
    Đọc groups.csv (tạo bởi compute_groups.py) và map mỗi ảnh trong `paths`
    sang group_id của nó. Các ảnh trùng/gần trùng nhau sẽ có cùng group_id,
    đảm bảo GroupShuffleSplit/GroupKFold không chia chúng vào cả train và test.
    """
    if not os.path.exists(groups_csv):
        raise FileNotFoundError(
            f"❌ Không tìm thấy {groups_csv}. Hãy chạy `python compute_groups.py` trước."
        )
    df = pd.read_csv(groups_csv)
    # Chuẩn hoá đường dẫn để so khớp chắc chắn (Windows backslash vs forward slash)
    df["filepath_norm"] = df["filepath"].apply(lambda p: os.path.normpath(p))
    path_to_group = dict(zip(df["filepath_norm"], df["group_id"]))

    groups = []
    missing = 0
    for p in paths:
        key = os.path.normpath(p)
        if key in path_to_group:
            groups.append(path_to_group[key])
        else:
            missing += 1
            groups.append(-1)  # ảnh không tìm thấy trong groups.csv -> group riêng (hiếm khi xảy ra)
    if missing:
        print(f"⚠️  {missing} ảnh không khớp được với groups.csv (kiểm tra lại đường dẫn).")
    return np.array(groups)


def make_pipeline(model, n_pca=100):
    """
    Tạo Pipeline: StandardScaler → PCA(n_components) → Model.
    PCA giảm chiều HOG từ 8100 về n_pca, giúp chống curse of dimensionality
    và giảm overfitting đáng kể trên dataset nhỏ.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("clf",    model),
    ])


def get_models():
    """
    Trả về dict các Pipeline (scaler → PCA → model).
    RF: max_depth=8, min_samples_leaf=5, max_samples=0.8
        để tránh overfit trên dataset nhỏ.
    """
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0,
                                                   random_state=RANDOM_STATE),
        "SVM (RBF)":   SVC(kernel="rbf",    C=10, gamma="scale",
                           probability=True, random_state=RANDOM_STATE),
        "SVM (Linear)": SVC(kernel="linear", C=1,
                            probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        # RF: thắt chặt regularization để tránh overfit
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,           # giảm từ 20 → 8
            min_samples_leaf=5,    # tăng từ 3 → 5
            min_samples_split=10,  # tăng từ 6 → 10
            max_features="sqrt",
            max_samples=0.8,       # bagging: mỗi cây chỉ thấy 80% mẫu
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=RANDOM_STATE
        ),
        "Naive Bayes": GaussianNB(),
    }
    if HAS_XGB:
        base_models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
            use_label_encoder=False
        )
    # Bọc tất cả vào Pipeline
    return {name: make_pipeline(m) for name, m in base_models.items()}


# Lưới hyperparameter dùng prefix pipeline ("clf__" cho model, "pca__" cho PCA)
# RF: tập trung vào regularisation (max_depth nhỏ, min_samples lớn, max_samples < 1)
PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF)":   {"clf__C": [1, 10, 50, 100], "clf__gamma": ["scale", 0.01, 0.005]},
    "SVM (Linear)": {"clf__C": [0.01, 0.1, 1, 10]},
    "KNN": {"clf__n_neighbors": [5, 7, 9, 11, 15], "clf__weights": ["uniform", "distance"]},
    "Random Forest": {
        "pca__n_components":      [50, 100, 150],       # thử vài mức PCA
        "clf__max_depth":         [5, 8, 12],           # giới hạn độ sâu
        "clf__min_samples_leaf":  [3, 5, 8, 10],        # regularization mạnh hơn
        "clf__min_samples_split": [6, 10, 15],
        "clf__max_features":      ["sqrt", "log2"],
        "clf__max_samples":       [0.7, 0.8, 1.0],      # bagging sub-sampling
    },
    "Gradient Boosting": {
        "clf__n_estimators":  [100, 200, 300],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth":     [3, 4, 5],
        "clf__subsample":     [0.7, 0.8, 1.0],
    },
    "Naive Bayes": {},
    "XGBoost": {
        "clf__n_estimators":     [100, 200],
        "clf__learning_rate":    [0.03, 0.05, 0.1],
        "clf__max_depth":        [3, 5, 7],
        "clf__subsample":        [0.7, 0.8],
        "clf__colsample_bytree": [0.7, 0.8],
    },
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

    print("\n" + "=" * 70)
    print("BƯỚC 1b: GOM NHÓM ẢNH TRÙNG/GẦN TRÙNG (chống leakage)")
    print("=" * 70)
    groups = load_groups(paths)
    print(f"✅ Tổng số ảnh: {len(paths)} | Tổng số group: {len(np.unique(groups))}")

    # GroupShuffleSplit: đảm bảo các ảnh CÙNG group_id (trùng/gần trùng nhau)
    # luôn nằm cùng một phía (toàn bộ train hoặc toàn bộ test), không bị xé lẻ.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    groups_train = groups[train_idx]

    # Kiểm tra an toàn: không có group nào xuất hiện ở cả train và test
    overlap = set(groups_train) & set(groups[test_idx])
    if overlap:
        print(f"🔴 CẢNH BÁO: vẫn còn {len(overlap)} group bị chia vào cả train và test!")
    else:
        print("✅ Xác nhận: không có group nào bị chia vào cả train và test (an toàn, không leakage).")

    # Pipeline tích hợp scaler + PCA, nên KHÔNG cần scale riêng ở đây
    # X_train / X_test truyền vào raw feature
    print("\n" + "=" * 70)
    print("BƯỚC 2: TRAIN & SO SÁNH NHIỀU MÔ HÌNH (5-fold Cross-Validation)")
    print("="  * 70)
    print("  (Mỗi Pipeline = StandardScaler → PCA(100) → Model)")
    print("=" * 70)

    models = get_models()
    cv = GroupKFold(n_splits=5)
    results = []

    for name, model in models.items():
        # CV trên raw feature — Pipeline tự scale + PCA bên trong
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, groups=groups_train,
                                     scoring="accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))   # Train acc
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        overfit_gap = train_acc - cv_scores.mean()  # gap = dấu hiệu overfit

        results.append({
            "Model": name,
            "Train Accuracy": train_acc,
            "CV Accuracy (mean)": cv_scores.mean(),
            "CV Accuracy (std)": cv_scores.std(),
            "Test Accuracy": test_acc,
            "Test F1 (macro)": test_f1,
            "Overfit Gap": overfit_gap,
        })
        flag = "⚠️ OVERFIT" if overfit_gap > 0.08 else "✅"
        print(f"  {name:22s} | Train: {train_acc:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
              f"| Test: {test_acc:.4f} | F1: {test_f1:.4f} | Gap: {overfit_gap:+.4f} {flag}")

    results_df = pd.DataFrame(results).sort_values("Test F1 (macro)", ascending=False).reset_index(drop=True)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    print("\n" + "=" * 70)
    print("BẢNG XẾP HẠNG MODEL (theo Test F1-macro)")
    print("=" * 70)
    print(results_df[["Model", "Train Accuracy", "CV Accuracy (mean)", "CV Accuracy (std)",
                       "Test Accuracy", "Test F1 (macro)", "Overfit Gap"]].to_string(index=False))

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
        grid.fit(X_train, y_train, groups=groups_train)
        best_model = grid.best_estimator_
        print(f"✅ Best params: {grid.best_params_}")
        print(f"✅ Best CV F1-macro: {grid.best_score_:.4f}")
    else:
        best_model = base_model
        best_model.fit(X_train, y_train)

    y_pred_final = best_model.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred_final)
    final_f1 = f1_score(y_test, y_pred_final, average="macro")

    final_train_acc = accuracy_score(y_train, best_model.predict(X_train))
    final_gap = final_train_acc - final_acc
    print(f"\n📊 KẾT QUẢ CUỐI CÙNG trên Test set:")
    print(f"   Train Accuracy : {final_train_acc:.4f}")
    print(f"   Test Accuracy  : {final_acc:.4f}")
    print(f"   Test F1-macro  : {final_f1:.4f}")
    print(f"   Overfit Gap    : {final_gap:+.4f}" + (" ⚠️ còn overfit" if final_gap > 0.08 else " ✅ ổn định"))
    print("\n" + classification_report(y_test, y_pred_final, target_names=le.classes_))

    # ── Confusion matrix ──────────────────────────────────────────────────
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

    # ── Biểu đồ so sánh các model (Train vs CV vs Test) ──────────────────
    plot_df = results_df.sort_values("Test F1 (macro)")
    x = np.arange(len(plot_df))
    width = 0.28
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(x - width, plot_df["Train Accuracy"],  width, label="Train Acc",  color="#e07b54")
    ax.barh(x,          plot_df["CV Accuracy (mean)"], width, label="CV Acc",    color="#d4a017")
    ax.barh(x + width, plot_df["Test Accuracy"],   width, label="Test Acc",   color="#4caf82")
    ax.set_yticks(x)
    ax.set_yticklabels(plot_df["Model"])
    ax.set_xlabel("Accuracy")
    ax.set_title("So sánh mô hình: Train / CV / Test Accuracy")
    ax.legend()
    ax.set_xlim(0, 1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_chart.png"), dpi=150)
    plt.close()

    # ── Learning Curve cho best model (visualize overfitting) ─────────────
    print("\n" + "=" * 70)
    print(f"BƯỚC 4: VẼ LEARNING CURVE cho '{best_model_name}'")
    print("=" * 70)
    train_sizes_abs, lc_train_scores, lc_val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=cv, groups=groups_train,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 8),
        n_jobs=-1
    )
    lc_train_mean = lc_train_scores.mean(axis=1)
    lc_train_std  = lc_train_scores.std(axis=1)
    lc_val_mean   = lc_val_scores.mean(axis=1)
    lc_val_std    = lc_val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_abs, lc_train_mean, "o-", color="#e07b54", label="Train Accuracy")
    plt.fill_between(train_sizes_abs,
                     lc_train_mean - lc_train_std,
                     lc_train_mean + lc_train_std, alpha=0.15, color="#e07b54")
    plt.plot(train_sizes_abs, lc_val_mean, "s--", color="#4caf82", label="CV Accuracy")
    plt.fill_between(train_sizes_abs,
                     lc_val_mean - lc_val_std,
                     lc_val_mean + lc_val_std, alpha=0.15, color="#4caf82")
    plt.xlabel("Số lượng mẫu train")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve – {best_model_name} (tuned)\n"
              "Khoảng cách 2 đường nhỏ = model tổng quát hoá tốt")
    plt.legend(loc="lower right")
    plt.ylim(0.5, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve_best_model.png"), dpi=150)
    plt.close()
    print(f"✅ Learning curve đã lưu vào: {OUTPUT_DIR}/learning_curve_best_model.png")

    # ===================================================================
    # LƯU MODEL PIPELINE + LABEL ENCODER
    # ===================================================================
    # best_model là Pipeline (scaler + PCA + clf) — lưu 1 file duy nhất
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.pkl"))
    joblib.dump(le,         os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    print(f"\n💾 Đã lưu Pipeline ({best_model_name}) vào: {OUTPUT_DIR}/best_model.pkl")
    print(f"   (Pipeline gồm: StandardScaler → PCA → {best_model_name})")
    print(f"💾 Đã lưu label encoder vào: {OUTPUT_DIR}/label_encoder.pkl")
    print(f"📈 Biểu đồ và bảng so sánh nằm trong thư mục: {OUTPUT_DIR}/")
    print(f"\n💡 Khi dùng predict.py: chỉ cần load best_model.pkl + label_encoder.pkl")
    print(f"   (Không cần scaler.pkl riêng nữa — scaler đã tích hợp vào pipeline)")


if __name__ == "__main__":
    main()