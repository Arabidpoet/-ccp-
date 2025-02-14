
import os
import csv
import warnings
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============== 第A部分：KCCA核心类 ==============
def rbf_kernel(X, Y=None, sigma=1.0):
    """
    计算 RBF 核矩阵:
      K(i, j) = exp(-||X_i - Y_j||^2 / (2*sigma^2))
    X.shape = (N, Dx), Y.shape = (M, Dx).
    若 Y=None，则默认为 Y=X，返回 NxN 自核矩阵。
    """
    if Y is None:
        Y = X
    # 计算欧几里得距离的平方
    X_sq = np.sum(X**2, axis=1, keepdims=True)  # (N,1)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T  # (1,M)
    dist_sq = X_sq + Y_sq - 2.0 * np.dot(X, Y.T)  # (N, M)
    dist_sq = np.maximum(dist_sq, 0.0)

    # 防止指数溢出
    max_exp = 100.0
    dist_sq_clip = np.minimum(dist_sq / (2.0 * sigma**2), max_exp)
    K = np.exp(-dist_sq_clip)
    return K.astype(np.float32)

def center_kernel(K):
    """
    对核矩阵做去中心化: Kc = H K H，其中 H = I - 1/n·11^T
    """
    n = K.shape[0]
    one_n = np.ones((n, n), dtype=np.float32) / n
    I = np.eye(n, dtype=np.float32)
    H = I - one_n
    return H @ K @ H

def matrix_inv_sqrt(M, reg=1e-12):
    """
    对矩阵 M 做 (M + reg·I)^(-1/2) 运算，用于白化。
    """
    M_reg = M + reg * np.eye(M.shape[0], dtype=M.dtype)
    U, S, Vt = np.linalg.svd(M_reg, full_matrices=False)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + reg))
    return (U @ S_inv_sqrt @ Vt).astype(np.float32)


class FeatureLevelKCCA:
    """
    标准 KCCA 实现，在「样本层面」做 RBF 核，允许 X1, X2 具有不同特征维度。
    步骤：
      1) K1 = K(X1, X1)，K2 = K(X2, X2) -> 去中心化 -> 得到 K1c, K2c
      2) A = (K1c + regI)^(-1/2)，B = (K2c + regI)^(-1/2)
      3) M = A (K1c * K2c) B
      4) SVD分解 M，保留前 n_components 列
      5) W1 = A·U_r, W2 = B·V_r，用于投影
    transform 时，对新样本先构造与训练集的核，再去中心化，然后右乘 W1 或 W2。
    """

    def __init__(self, sigma=1.0, n_components=2, center=True, reg=1e-12):
        self.sigma = sigma
        self.n_components = n_components
        self.center = center
        self.reg = reg

        self.X1_train = None
        self.X2_train = None

        self.W1 = None
        self.W2 = None
        self.K1_centerer = None  # 用于transform时重现 K1c 的中心化
        self.K2_centerer = None  # 用于transform时重现 K2c 的中心化

    def fit(self, X1, X2):
        """
        训练阶段：X1, X2 均是 (N, Dx) 形式，N 条样本，不同视图特征数可不同。
        """
        N = X1.shape[0]
        if X2.shape[0] != N:
            raise ValueError("X1, X2 样本数不一致，无法做 KCCA")

        # 记住训练集，以便 transform 时计算核
        self.X1_train = X1
        self.X2_train = X2

        # (1) 计算自核 K1, K2
        K1 = rbf_kernel(X1, sigma=self.sigma)  # (N,N)
        K2 = rbf_kernel(X2, sigma=self.sigma)  # (N,N)

        # (2) 去中心化
        if self.center:
            K1c = center_kernel(K1)
            K2c = center_kernel(K2)
        else:
            K1c = K1
            K2c = K2

        # 把去中心化后的 K1c, K2c 暂存，后续 transform 新数据时要再次对核做相同中心化
        self.K1_centerer = K1c
        self.K2_centerer = K2c

        # (3) 白化
        A = matrix_inv_sqrt(K1c, reg=self.reg)  # NxN
        B = matrix_inv_sqrt(K2c, reg=self.reg)  # NxN

        # (4) 在核空间中计算“跨视图协方差” ~ K1c * K2c
        #     M = A (K1c @ K2c) B -> NxN
        M = A @ (K1c @ K2c) @ B

        # (5) SVD
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        U_r = U[:, :self.n_components]      # Nx n_components
        V_r = Vt[:self.n_components, :].T   # Nx n_components

        # (6) 投影方向
        self.W1 = A @ U_r  # Nx n_components
        self.W2 = B @ V_r  # Nx n_components

    def transform(self, X, view=1):
        """
        将新样本 X 投影到 KCCA 的低维子空间：
        1) 先与对应视图的训练集做核 (N_test, N_train)
        2) 若 center=True，需要做相同的去中心化
        3) 再乘以投影矩阵 W1/W2，得到 (N_test, n_components)
        """
        if self.W1 is None or self.W2 is None:
            raise ValueError("尚未 fit()，无法 transform()。")

        if view == 1:
            # 先与 X1_train 做核
            Kx = rbf_kernel(X, self.X1_train, sigma=self.sigma)  # (N_test, N_train)
            if self.center:
                # 与训练时相同的中心化：K1c = H1 * K1 * H1
                # 这里需要对 Kx 的行/列均做类似变换。最简单办法：再减去 K1, Kx 的均值项
                Kx = self._center_test_kernel(Kx, self.K1_centerer)
            return Kx @ self.W1

        else:
            # 与 X2_train 做核
            Kx = rbf_kernel(X, self.X2_train, sigma=self.sigma)  # (N_test, N_train)
            if self.center:
                Kx = self._center_test_kernel(Kx, self.K2_centerer)
            return Kx @ self.W2

    def _center_test_kernel(self, Kx, Ktrain_centered):

        # 训练集 NxN
        N = Ktrain_centered.shape[0]

        row_mean = np.mean(Kx, axis=1, keepdims=True)
        # 列均值 => shape (1,N_train)
        col_mean = np.mean(Kx, axis=0, keepdims=True)
        # 整体均值 => scalar
        grand_mean = np.mean(Kx)

        Kx_c = Kx - row_mean - col_mean + grand_mean
        return Kx_c


# ============== 第B部分：数据加载与特征提取（与之前相同） ==============
def extract_rgb_histogram(img, bins=16):
    h_r = np.histogram(img.getchannel('R'), bins=bins, range=(0, 255))[0]
    h_g = np.histogram(img.getchannel('G'), bins=bins, range=(0, 255))[0]
    h_b = np.histogram(img.getchannel('B'), bins=bins, range=(0, 255))[0]
    hist = np.concatenate([h_r, h_g, h_b], axis=0)
    hist = hist.astype(np.float32) / (img.size[0] * img.size[1])
    return hist

def extract_gray_histogram(img, bins=16):
    gray_img = img.convert('L')
    h = np.histogram(gray_img, bins=bins, range=(0, 255))[0]
    h = h.astype(np.float32) / (img.size[0] * img.size[1])
    return h

def load_image_data_as_samples(
    dataset_path,
    img_size=(224, 224),
    sample_ratio=1.0
):
    """
    修改后的数据加载函数，适应Br35H数据集结构
    """
    if not 0 < sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0,1]")

    category_mapping = {
        'no': 0,  # 非肿瘤
        'yes': 1  # 肿瘤
    }

    X1_list, X2_list, labels_list = [], [], []
    for category, label in category_mapping.items():
        cat_path = os.path.join(dataset_path, category)
        if not os.path.exists(cat_path):
            print(f"【警告】类别文件夹 {cat_path} 不存在，跳过...")
            continue

        image_files = [f for f in os.listdir(cat_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        n_samples = int(len(image_files) * sample_ratio)
        if sample_ratio < 1.0 and len(image_files) > 0:
            image_files = np.random.choice(image_files, size=n_samples, replace=False)

        print(f"类别 {category}：共 {len(image_files)} 个图像文件被读取...")
        for fn in image_files:
            fp = os.path.join(cat_path, fn)
            try:
                img = Image.open(fp).convert('RGB')
                img = img.resize(img_size)
                feat1 = extract_rgb_histogram(img)  # (48,) for bins=16
                feat2 = extract_gray_histogram(img) # (16,)
                X1_list.append(feat1)
                X2_list.append(feat2)
                labels_list.append(label)
            except Exception as e:
                print(f"读取图像 {fp} 失败: {e}")

    X1 = np.array(X1_list, dtype=np.float32)  # (N, 48)
    X2 = np.array(X2_list, dtype=np.float32)  # (N, 16)
    labels = np.array(labels_list, dtype=np.int32)
    return X1, X2, labels


# ============== 第C部分：评估函数及主流程 ==============
def train_test_split_on_samples(X1, X2, labels, test_size, random_state=42):
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        X1, X2, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    return X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te


def evaluate_kcca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, sigma_vals,
        csv_path="nn_results_kcca.csv",
        random_seed=42,
        verbose=True
):
    """
    1-NN评估KCCA投影效果，并记录Accuracy和F1指标。
    """
    results = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "sigma", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for s in sigma_vals:
            if verbose:
                print(f"\n[Evaluation] 1-NN (KCCA) => n_components={n_comp}, sigma={s}")

            kcca = FeatureLevelKCCA(sigma=s, n_components=n_comp, center=True, reg=1e-6)
            kcca.fit(X1_train, X2_train)

            # 投影到KCCA子空间
            Z1_train = kcca.transform(X1_train, view=1)  # (N_train,n_comp)
            Z2_train = kcca.transform(X2_train, view=2)  # (N_train,n_comp)
            X_train_kcca = np.hstack([Z1_train, Z2_train])  # (N_train, 2*n_comp)

            Z1_test = kcca.transform(X1_test, view=1)
            Z2_test = kcca.transform(X2_test, view=2)
            X_test_kcca = np.hstack([Z1_test, Z2_test])    # (N_test, 2*n_comp)

            # 1-NN
            nnclf = KNeighborsClassifier(n_neighbors=1)
            nnclf.fit(X_train_kcca, y_train)
            y_pred = nnclf.predict(X_test_kcca)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'n_components': n_comp,
                'sigma': s,
                'accuracy': acc,
                'f1': f1
            })

            with open(csv_path, "a", newline='') as f2:
                writer2 = csv.writer(f2)
                writer2.writerow([n_comp, s, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[1-NN (KCCA) Top 5 Results]")
    print("Rank\tn_comp\tsigma\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['sigma']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"Results saved to {csv_path}")


def evaluate_kcca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, sigma_vals,
        csv_path="lightgbm_results_kcca.csv",
        random_seed=42,
        verbose=True
):
    """
    LightGBM评估KCCA投影效果，记录Accuracy和F1指标。
    """
    lgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_seed,
        'verbose': -1,
        'force_col_wise': True
    }

    results = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "sigma", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for s in sigma_vals:
            if verbose:
                print(f"\n[Evaluation] LightGBM (KCCA) => n_components={n_comp}, sigma={s}")

            kcca = FeatureLevelKCCA(sigma=s, n_components=n_comp, center=True, reg=1e-6)
            kcca.fit(X1_train, X2_train)

            Z1_train = kcca.transform(X1_train, view=1)
            Z2_train = kcca.transform(X2_train, view=2)
            X_train_kcca = np.hstack([Z1_train, Z2_train])

            Z1_test = kcca.transform(X1_test, view=1)
            Z2_test = kcca.transform(X2_test, view=2)
            X_test_kcca = np.hstack([Z1_test, Z2_test])

            clf = LGBMClassifier(**lgb_params)
            clf.fit(X_train_kcca, y_train)
            y_pred = clf.predict(X_test_kcca)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'n_components': n_comp,
                'sigma': s,
                'accuracy': acc,
                'f1': f1
            })

            with open(csv_path, "a", newline='') as f2:
                writer2 = csv.writer(f2)
                writer2.writerow([n_comp, s, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[LightGBM (KCCA) Top 5 Results]")
    print("Rank\tn_comp\tsigma\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['sigma']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


def main():
    """
    主函数：加载数据 -> 划分训练/测试 -> 使用 KCCA -> 评估 1-NN & LightGBM
    """
    dataset_path = r"E:\jqxx\dataset\Br35H"
    sample_ratio = 1.0
    test_ratio = 0.2
    random_seed = 42

    print(f"正在从 {dataset_path} 加载数据...")
    X1_samples, X2_samples, labels = load_image_data_as_samples(
        dataset_path=dataset_path,
        img_size=(224, 224),
        sample_ratio=sample_ratio
    )

    print("\n数据加载完成。信息：")
    print("X1 (样本为行) shape:", X1_samples.shape)  # (N, 48)
    print("X2 (样本为行) shape:", X2_samples.shape)  # (N, 16)
    print("labels shape:", labels.shape)            # (N,)
    print("标签分布：", np.bincount(labels))

    # 样本级划分
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split_on_samples(
        X1_samples, X2_samples, labels,
        test_size=test_ratio,
        random_state=random_seed
    )

    # 这里 X1_tr, X2_tr 形状如 (2400,48), (2400,16)，满足 (N, D1), (N, D2)

    # 超参数范围
    n_components_vals = [2, 4, 8, 12, 15]
    sigma_vals = [0.05, 0.1, 0.2, 0.3, 0.5]

    # KCCA + 1-NN
    print("\n========== [Evaluation 1] 1-NN (KCCA) ==========")
    evaluate_kcca_nn(
        X1_tr, X2_tr, y_tr,
        X1_te, X2_te, y_te,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="nn_results_kcca.csv",
        random_seed=random_seed,
        verbose=True
    )

    # KCCA + LightGBM
    print("\n========== [Evaluation 2] LightGBM (KCCA) ==========")
    evaluate_kcca_lightgbm(
        X1_tr, X2_tr, y_tr,
        X1_te, X2_te, y_te,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="lightgbm_results_kcca.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n>>> Done.")


if __name__ == "__main__":
    main()