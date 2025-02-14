import os
import cv2
import csv
import random
import warnings
import numpy as np

from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============== 第A部分：KCCA方法模块 ==============
class FeatureLevelKCCA:
    """
    KCCA 实现示例：
      1) 对两视图 X1, X2 分别用零填充，拼接到相同维度 (d1 + d2)；
      2) 基于 RBF 核来构造核矩阵；
      3) 进行核居中 (Kc = H K H)，去除数据整体均值；
      4) 在居中后核矩阵上加正则项 λI 并做类似 CCA 的处理，
         通过奇异值分解 (SVD) 获得投影系数 alpha, beta；
      5) transform 时，同样对新数据零填充，再和训练集做核居中后投影。

    备注：
      - 核心计算在 fit 阶段： KxC^-1/2, KyC^-1/2, 然后做 M = KxC^-1/2 KxyC KyC^-1/2，再奇异值分解。
      - (KxyC) 是视图1 与视图2 间的居中核矩阵，也需要同样的居中操作。

    参数：
      :param sigma: RBF 核宽度 (σ)
      :param n_components: 要保留的典型相关向量个数
      :param regular: 正则化系数 (λ)
      :param verbose: 是否打印一些 debug 信息
    """

    def __init__(self, sigma=1.0, n_components=2, regular=1e-6, verbose=False):
        self.sigma = sigma
        self.n_components = n_components
        self.regular = regular
        self.verbose = verbose

        # 在 fit 后赋值
        self.alpha = None
        self.beta = None
        self.X1_train_unif = None  # 训练阶段 X1 零填充
        self.X2_train_unif = None  # 训练阶段 X2 零填充
        self.d1 = None
        self.d2 = None

        # 用于居中操作
        self.H = None  # H = I - 1/n * 11^T

    def _unify_dim_view1(self, X):
        """
        将视图1的数据 X(n, d1) 零填充到 (n, d1+d2) 。
        前 d1 列 = X，后 d2 列 = 0
        """
        n = X.shape[0]
        X_unif = np.zeros((n, self.d1 + self.d2), dtype=X.dtype)
        X_unif[:, :self.d1] = X
        return X_unif

    def _unify_dim_view2(self, X):
        """
        将视图2的数据 X(n, d2) 零填充到 (n, d1+d2) 。
        前 d1 列 = 0，后 d2 列 = X
        """
        n = X.shape[0]
        X_unif = np.zeros((n, self.d1 + self.d2), dtype=X.dtype)
        X_unif[:, self.d1:] = X
        return X_unif

    def _rbf_kernel(self, A, B):
        """
        RBF (Gaussian) 核矩阵: K[i,j] = exp( -||A[i]-B[j]||^2 / (2*sigma^2) ).
        A.shape = (nA, D), B.shape = (nB, D).
        """
        A_sq = np.sum(A ** 2, axis=1, keepdims=True)  # (nA,1)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True)  # (nB,1)
        dist_sq = A_sq + B_sq.T - 2 * (A @ B.T)
        K = np.exp(-dist_sq / (2.0 * self.sigma ** 2))
        return K

    def _center_kernel(self, K):
        """
        对核矩阵 K 做居中: Kc = H K H
        其中 H = I - 1/n · 11^T, 在 fit 时已创建 self.H
        """
        # Kc = H K H
        return self.H @ K @ self.H

    def fit(self, X1, X2):
        """
        拟合 KCCA，得到投影系数 self.alpha, self.beta.
        X1.shape=(n, d1), X2.shape=(n, d2)
        """
        # 记录 d1, d2，构造零填充后的训练数据
        self.d1 = X1.shape[1]
        self.d2 = X2.shape[1]

        n = X1.shape[0]
        if self.verbose:
            print(f"[KCCA Fit] n={n}, d1={self.d1}, d2={self.d2}, sigma={self.sigma}, regular={self.regular}")

        # 零填充
        self.X1_train_unif = self._unify_dim_view1(X1)  # (n, d1+d2)
        self.X2_train_unif = self._unify_dim_view2(X2)

        # 居中矩阵 H = I - 1/n * 11^T
        self.H = np.eye(n) - np.ones((n, n)) / n

        # ============= 1) 构造核矩阵 Kx, Ky, Kxy =============
        Kx = self._rbf_kernel(self.X1_train_unif, self.X1_train_unif)  # (n,n)
        Ky = self._rbf_kernel(self.X2_train_unif, self.X2_train_unif)  # (n,n)
        Kxy = self._rbf_kernel(self.X1_train_unif, self.X2_train_unif)  # (n,n)

        # ============= 2) 对 Kx, Ky, Kxy 分别做居中 =============
        KxC = self._center_kernel(Kx)
        KyC = self._center_kernel(Ky)
        KxyC = self._center_kernel(Kxy)  # 交差视图也要居中

        # ============= 3) 加正则项 =============
        KxC += np.eye(n) * self.regular
        KyC += np.eye(n) * self.regular

        # ============= 4) 计算 KxC^-1/2, KyC^-1/2 =============
        # 先做特征分解 (KxC, KyC 对称正定)
        eigvals_x, eigvecs_x = np.linalg.eigh(KxC)
        eigvals_y, eigvecs_y = np.linalg.eigh(KyC)

        # 防止 sqrt( < 0 ) 或 过小问题
        eps = 1e-12
        eigvals_x[eigvals_x < eps] = eps
        eigvals_y[eigvals_y < eps] = eps

        D_x_inv_sqrt = np.diag(eigvals_x ** (-0.5))
        D_y_inv_sqrt = np.diag(eigvals_y ** (-0.5))

        KxC_inv_sqrt = eigvecs_x @ D_x_inv_sqrt @ eigvecs_x.T  # (n,n)
        KyC_inv_sqrt = eigvecs_y @ D_y_inv_sqrt @ eigvecs_y.T  # (n,n)

        # ============= 5) M = KxC^-1/2 * KxyC * KyC^-1/2 =============
        M = KxC_inv_sqrt @ KxyC @ KyC_inv_sqrt

        # ============= 6) SVD 分解 M = U S V^T =============
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # 取前 n_components
        U_r = U[:, :self.n_components]  # (n, n_components)
        V_r = Vt[:self.n_components, :].T  # (n, n_components)

        # ============= 7) alpha, beta =============
        self.alpha = KxC_inv_sqrt @ U_r  # (n, n_components)
        self.beta = KyC_inv_sqrt @ V_r  # (n, n_components)

        if self.verbose:
            print("[KCCA Fit] Top singular values:", S[:5], "..." if len(S) > 5 else "")

    def transform(self, X, which='view1'):
        """
        对新样本 X(n_test, d1或d2) 投影到 (n_test, n_components).
        步骤：
          1) 同样做零填充到 (n_test, d1+d2)；
          2) 计算与对应训练视图核矩阵；
          3) 对该核矩阵做同样的居中 => Knew_c = H_test * Knew * H_train
          4) 投影到 alpha 或 beta 空间。
        """
        if self.alpha is None or self.beta is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_train = self.X1_train_unif.shape[0]  # 训练样本数
        n_test = X.shape[0]

        # 零填充
        if which == 'view1':
            X_unif = self._unify_dim_view1(X)  # (n_test, d1+d2)
            # 计算 K(X_unif, X1_train_unif)
            K_new = self._rbf_kernel(X_unif, self.X1_train_unif)
            K_new_mean = np.mean(K_new, axis=0, keepdims=True)
            K_train_mean = np.mean(self._rbf_kernel(self.X1_train_unif, self.X1_train_unif), axis=0, keepdims=True)
            K_new_centered = K_new - K_new_mean - K_train_mean + np.mean(K_train_mean)

            Z = K_new_centered @ self.alpha  # (n_test, n_components)
        else:
            X_unif = self._unify_dim_view2(X)
            K_new = self._rbf_kernel(X_unif, self.X2_train_unif)

            # 居中处理
            K_new_mean = np.mean(K_new, axis=0, keepdims=True)
            K_train_mean = np.mean(self._rbf_kernel(self.X2_train_unif, self.X2_train_unif), axis=0, keepdims=True)
            K_new_centered = K_new - K_new_mean - K_train_mean + np.mean(K_train_mean)

            Z = K_new_centered @ self.beta

        return Z


# ============== 第B部分：数据加载与特征提取 ==============
def extract_features(img):
    """
    图像特征提取:
      - RGB直方图(3*32=96维)
      - LBP纹理(radius=3, uniform => ~26维)
    """
    img = cv2.resize(img, (256, 256))

    # 特征1: RGB直方图
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


# ============== 第C部分：评估函数 (1NN & LightGBM) ==============
def evaluate_kcca_nn(X1_train, X2_train, y_train,
                     X1_test, X2_test, y_test,
                     n_components_vals,
                     sigma_vals,
                     csv_path="nn_results_kcca.csv",
                     random_seed=42, verbose=True):
    """
    用 1-NN 来评估 KCCA 投影后的分类性能。
    遍历 n_components_vals、sigma_vals，记录结果到 CSV。
    """
    np.random.seed(random_seed)

    # 先对原始特征做标准化
    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc = scaler1.transform(X1_test)
    X2_test_sc = scaler2.transform(X2_test)

    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "sigma", "accuracy", "f1"])

    results = []
    for nc in n_components_vals:
        for s in sigma_vals:
            if verbose:
                print(f"\n[KCCA + 1-NN] n_components={nc}, sigma={s}")
            kcca = FeatureLevelKCCA(sigma=s, n_components=nc, regular=1e-6, verbose=False)
            kcca.fit(X1_train_sc, X2_train_sc)

            # 投影
            Z1_train = kcca.transform(X1_train_sc, which='view1')
            Z2_train = kcca.transform(X2_train_sc, which='view2')
            Z_train = np.hstack([Z1_train, Z2_train])

            Z1_test = kcca.transform(X1_test_sc, which='view1')
            Z2_test = kcca.transform(X2_test_sc, which='view2')
            Z_test = np.hstack([Z1_test, Z2_test])

            # 1-NN
            nn = KNeighborsClassifier(n_neighbors=1)
            nn.fit(Z_train, y_train)
            pred_nn = nn.predict(Z_test)
            acc = accuracy_score(y_test, pred_nn)
            f1 = f1_score(y_test, pred_nn, average='macro')

            results.append((nc, s, acc, f1))
            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nc, s, f"{acc:.4f}", f"{f1:.4f}"])

    # 按 F1 排序并显示前5
    results.sort(key=lambda x: x[3], reverse=True)
    print("\n[1-NN + KCCA Top 5 Results]")
    print("Rank\tn_comp\tsigma\tACC\tF1")
    print("-" * 40)
    for i, (nc, s, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{s}\t{acc:.4f}\t{f1:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_kcca_lightgbm(X1_train, X2_train, y_train,
                           X1_test, X2_test, y_test,
                           n_components_vals,
                           sigma_vals,
                           csv_path="lightgbm_results_kcca.csv",
                           random_seed=42, verbose=True):
    """
    用 LightGBM 来评估 KCCA 投影后的分类性能。
    遍历 n_components_vals、sigma_vals，记录到 CSV。
    """
    np.random.seed(random_seed)

    # LightGBM 参数
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

    # 标准化
    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc = scaler1.transform(X1_test)
    X2_test_sc = scaler2.transform(X2_test)

    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "sigma", "accuracy", "f1"])

    results = []
    for nc in n_components_vals:
        for s in sigma_vals:
            if verbose:
                print(f"\n[KCCA + LightGBM] n_components={nc}, sigma={s}")

            kcca = FeatureLevelKCCA(sigma=s, n_components=nc, regular=1e-6, verbose=False)
            kcca.fit(X1_train_sc, X2_train_sc)

            Z1_train = kcca.transform(X1_train_sc, which='view1')
            Z2_train = kcca.transform(X2_train_sc, which='view2')
            Z_train = np.hstack([Z1_train, Z2_train])

            Z1_test = kcca.transform(X1_test_sc, which='view1')
            Z2_test = kcca.transform(X2_test_sc, which='view2')
            Z_test = np.hstack([Z1_test, Z2_test])

            clf = LGBMClassifier(**lgb_params)
            clf.fit(Z_train, y_train)
            y_pred = clf.predict(Z_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append((nc, s, acc, f1))
            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nc, s, f"{acc:.4f}", f"{f1:.4f}"])

    # 按 F1 排序并显示前5
    results.sort(key=lambda x: x[3], reverse=True)
    print("\n[LightGBM + KCCA Top 5 Results]")
    print("Rank\tn_comp\tsigma\tACC\tF1")
    print("-" * 40)
    for i, (nc, s, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{s}\t{acc:.4f}\t{f1:.4f}")

    print(f"\nResults saved to {csv_path}")


# ============== 第D部分：主函数 ==============
def main():
    # 修改为你的数据根目录
    data_dir = r"E:\jqxx\Chest CT-Scan images Dataset"

    print("Loading data from:", data_dir)
    (X1_train, X2_train, y_train,
     X1_val, X2_val, y_val,
     X1_test, X2_test, y_test) = load_all_data(data_dir)

    # 合并 train + val
    X1_train = np.vstack([X1_train, X1_val])
    X2_train = np.vstack([X2_train, X2_val])
    y_train = np.concatenate([y_train, y_val])

    print(f"\n[Final Train] X1:{X1_train.shape}, X2:{X2_train.shape}, y:{y_train.shape}")
    print(f"[Test       ] X1:{X1_test.shape}, X2:{X2_test.shape}, y:{y_test.shape}")

    # 参数范围
    n_components_vals = [1, 2, 5, 10,15,20,25,30]
    sigma_vals = [0.1, 0.5, 1.0, 5.0,10]

    # 1) 用 1-NN 做评价
    print("\n========== [Evaluation 1] 1-NN + KCCA ==========")
    evaluate_kcca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="nn_results_kcca.csv",
        random_seed=42,
        verbose=True
    )

    # 2) 用 LightGBM 做评价
    print("\n========== [Evaluation 2] LightGBM + KCCA ==========")
    evaluate_kcca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="lightgbm_results_kcca.csv",
        random_seed=42,
        verbose=True
    )


if __name__ == "__main__":
    main()