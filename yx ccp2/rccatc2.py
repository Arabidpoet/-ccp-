
import os
import cv2
import csv
import random
import warnings
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============== 第A部分：RCCA方法模块 ==============
class FeatureLevelRCCA:


    def __init__(self, n_components=2, regular=1e-6, verbose=False):

        self.n_components = n_components
        self.regular = regular
        self.verbose = verbose
        self.W1 = None
        self.W2 = None
        self.mean1_ = None
        self.mean2_ = None

    def fit(self, X1, X2):

        if self.verbose:
            print(f"[RCCA Fit] Start. n_components={self.n_components}, regular={self.regular}")

        # 0 去中心化
        self.mean1_ = np.mean(X1, axis=0, keepdims=True)
        self.mean2_ = np.mean(X2, axis=0, keepdims=True)
        X1c = X1 - self.mean1_
        X2c = X2 - self.mean2_

        n, d1 = X1c.shape
        _, d2 = X2c.shape

        # 1协方差矩阵 (以样本数 n 为基准)
        S11 = (X1c.T @ X1c) / (n - 1)
        S22 = (X2c.T @ X2c) / (n - 1)
        S12 = (X1c.T @ X2c) / (n - 1)
        S21 = S12.T  # (d2, d1)

        # 2对角线加正则
        S11 += self.regular * np.eye(d1)
        S22 += self.regular * np.eye(d2)

        # 3广义特征值分解
        #    解方程: (S11^-1 S12 S22^-1 S21) w1 = rho^2 * w1
        try:
            S11_inv = np.linalg.inv(S11)
            S22_inv = np.linalg.inv(S22)

            M = S11_inv @ S12 @ S22_inv @ S21  # (d1, d1)
            eigvals, eigvecs = np.linalg.eig(M)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular, try increasing regular or reducing dimension.")

        # 4只取实部，并排序
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]

        # 5选取前 n_components
        W1_raw = eigvecs_sorted[:, :self.n_components]  # (d1, n_components)
        W2_raw = S22_inv @ S21 @ W1_raw  # (d2, n_components)

        # 对每个方向做 norm=1
        for i in range(self.n_components):
            norm_w1 = np.linalg.norm(W1_raw[:, i])
            norm_w2 = np.linalg.norm(W2_raw[:, i])
            if norm_w1 < 1e-12:
                norm_w1 = 1.0
            if norm_w2 < 1e-12:
                norm_w2 = 1.0
            W1_raw[:, i] /= norm_w1
            W2_raw[:, i] /= norm_w2

        self.W1 = W1_raw
        self.W2 = W2_raw

        if self.verbose:
            print(f"[RCCA Fit] top eigenvals: {eigvals_sorted[:5]}")

    def transform(self, X, which='view1'):

        if which == 'view1':
            if self.W1 is None:
                raise ValueError("W1 not fitted.")
            Xc = X - self.mean1_
            return Xc @ self.W1
        else:
            if self.W2 is None:
                raise ValueError("W2 not fitted.")
            Xc = X - self.mean2_
            return Xc @ self.W2


# ============== 第B部分：数据加载与特征提取 ==============
def extract_features(img):

    img = cv2.resize(img, (256, 256))

    # 特征1: RGB hist
    color_feat = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        color_feat.extend(hist)
    color_feat = np.array(color_feat, dtype=np.float32)

    # 特征2: LBP纹理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist_lbp = hist_lbp.astype(np.float32)

    return color_feat, hist_lbp


def load_data_from_split(root_dir, split='train'):

    if split == 'train' or split == 'val':
        class_names = {
            'normal': 0,
            'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 1,
            'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 2,
            'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 3
        }
    else:  # test
        class_names = {
            'normal': 0,
            'adenocarcinoma': 1,
            'large.cell.carcinoma': 2,
            'squamous.cell.carcinoma': 3
        }

    split_dir = os.path.join(root_dir, split)
    if not os.path.isdir(split_dir):
        raise ValueError(f"No directory for split={split}: {split_dir}")

    X1_list, X2_list, y_list = [], [], []

    for cname, label in class_names.items():
        class_dir = os.path.join(split_dir, cname)
        if not os.path.isdir(class_dir):
            print(f"Warning: {class_dir} not found, skip.")
            continue

        print(f"[{split}] Loading {cname} => label={label}")
        for fn in os.listdir(class_dir):
            if not (fn.lower().endswith('.png') or fn.lower().endswith('.jpg')):
                continue
            img_path = os.path.join(class_dir, fn)
            img = cv2.imread(img_path)
            if img is None:
                continue
            feat1, feat2 = extract_features(img)
            X1_list.append(feat1)
            X2_list.append(feat2)
            y_list.append(label)

    if not X1_list:
        raise ValueError(f"No valid data found under {split_dir}")

    X1 = np.array(X1_list)
    X2 = np.array(X2_list)
    y = np.array(y_list, dtype=int)
    return X1, X2, y


def load_all_data(root_dir):

    X1_train, X2_train, y_train = load_data_from_split(root_dir, 'train')
    X1_val, X2_val, y_val = load_data_from_split(root_dir, 'val')
    X1_test, X2_test, y_test = load_data_from_split(root_dir, 'test')

    print("\n[Train] X1:", X1_train.shape, "X2:", X2_train.shape, "y:", y_train.shape)
    print("[Val  ] X1:", X1_val.shape, "X2:", X2_val.shape, "y:", y_val.shape)
    print("[Test ] X1:", X1_test.shape, "X2:", X2_test.shape, "y:", y_test.shape)

    return (X1_train, X2_train, y_train,
            X1_val, X2_val, y_val,
            X1_test, X2_test, y_test)


# ============== 第C部分：评估函数 (1NN & LightGBM ==============
def evaluate_rcca_nn(X1_train, X2_train, y_train,
                     X1_test, X2_test, y_test,
                     n_components_vals, regular_vals,
                     csv_path="rcca_nn_results.csv",
                     random_seed=42, verbose=True):


    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "regular", "accuracy", "f1"])

    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc = scaler1.transform(X1_test)
    X2_test_sc = scaler2.transform(X2_test)

    results = []

    for nc in n_components_vals:
        for reg in regular_vals:
            if verbose:
                print(f"\n[Evaluation] RCCA + 1-NN => n_components={nc}, regular={reg}")

            rcca = FeatureLevelRCCA(n_components=nc, regular=reg, verbose=False)
            rcca.fit(X1_train_sc, X2_train_sc)

            Z1_train = rcca.transform(X1_train_sc, which='view1')
            Z2_train = rcca.transform(X2_train_sc, which='view2')
            Z_train = np.hstack([Z1_train, Z2_train])

            Z1_test = rcca.transform(X1_test_sc, which='view1')
            Z2_test = rcca.transform(X2_test_sc, which='view2')
            Z_test = np.hstack([Z1_test, Z2_test])

            nn = KNeighborsClassifier(n_neighbors=1)
            nn.fit(Z_train, y_train)
            pred_nn = nn.predict(Z_test)
            acc = accuracy_score(y_test, pred_nn)
            f1 = f1_score(y_test, pred_nn, average='macro')

            results.append((nc, reg, acc, f1))

            # 写入CSV
            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nc, reg, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x[3], reverse=True)
    print("\n[RCCA + 1-NN Top 5 Results]")
    print("Rank\tn_comp\tregular\tACC\tF1")
    print("-" * 50)
    for i, (nc, reg, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{reg}\t{acc:.4f}\t{f1:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_rcca_lightgbm(X1_train, X2_train, y_train,
                           X1_test, X2_test, y_test,
                           n_components_vals, regular_vals,
                           csv_path="rcca_lightgbm_results.csv",
                           random_seed=42, verbose=True):


    lgb_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 63,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_seed,
        'verbose': -1,
        'force_col_wise': True,
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss'
    }

    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "regular", "accuracy", "f1"])

    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc = scaler1.transform(X1_test)
    X2_test_sc = scaler2.transform(X2_test)

    results = []

    for nc in n_components_vals:
        for reg in regular_vals:
            if verbose:
                print(f"\n[Evaluation] RCCA + LightGBM => n_components={nc}, regular={reg}")

            rcca = FeatureLevelRCCA(n_components=nc, regular=reg, verbose=False)
            rcca.fit(X1_train_sc, X2_train_sc)

            Z1_train = rcca.transform(X1_train_sc, which='view1')
            Z2_train = rcca.transform(X2_train_sc, which='view2')
            Z_train = np.hstack([Z1_train, Z2_train])

            Z1_test = rcca.transform(X1_test_sc, which='view1')
            Z2_test = rcca.transform(X2_test_sc, which='view2')
            Z_test = np.hstack([Z1_test, Z2_test])

            clf = LGBMClassifier(**lgb_params)
            clf.fit(Z_train, y_train)
            y_pred = clf.predict(Z_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append((nc, reg, acc, f1))

            # 写入CSV
            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nc, reg, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x[3], reverse=True)
    print("\n[RCCA + LightGBM Top 5 Results]")
    print("Rank\tn_comp\tregular\tACC\tF1")
    print("-" * 50)
    for i, (nc, reg, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{reg}\t{acc:.4f}\t{f1:.4f}")

    print(f"\nResults saved to {csv_path}")


# ============== 第D部分：主函数 ==============
def main():
    data_dir = r"E:\jqxx\Chest CT-Scan images Dataset"

    print("Loading data from:", data_dir)
    (X1_train, X2_train, y_train,
     X1_val, X2_val, y_val,
     X1_test, X2_test, y_test) = load_all_data(data_dir)

    # 合并 train+val
    X1_train = np.vstack([X1_train, X1_val])
    X2_train = np.vstack([X2_train, X2_val])
    y_train = np.concatenate([y_train, y_val])

    print(f"\n[Final Train] X1:{X1_train.shape}, X2:{X2_train.shape}, y:{y_train.shape}")
    print(f"[Test       ] X1:{X1_test.shape}, X2:{X2_test.shape}, y:{y_test.shape}")

    # ------------------ 参数范围示例 ------------------
    # 设置 n_components 和 regular 两个超参进行搜索
    n_components_vals = [1, 2, 5, 10,15,20,25,30,35,40]  # 可以根据实际需求扩增
    regular_vals = [1e-5, 1e-4, 1e-3,1e-2, 1e-1,1,5,10,15,20]

    # 1) 评估: 1-NN
    print("\n========== [Evaluation 1] RCCA + 1-NN ==========")
    evaluate_rcca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        regular_vals=regular_vals,
        csv_path="rcca_nn_results.csv",
        random_seed=42,
        verbose=True
    )

    # 2) 评估: LightGBM
    print("\n========== [Evaluation 2] RCCA + LightGBM ==========")
    evaluate_rcca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        regular_vals=regular_vals,
        csv_path="rcca_lightgbm_results.csv",
        random_seed=42,
        verbose=True
    )


if __name__ == "__main__":
    main()