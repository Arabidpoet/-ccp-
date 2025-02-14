
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


# ============== 第A部分：KCCA方法模块 ==============

def rbf_kernel_features(X, Y=None, sigma=1.0):
    """
    构造 RBF 核矩阵
      X: shape = (d_X, n)  —— 每一列是一个样本，d_X 为特征维度
      Y: shape = (d_Y, n) ，默认为 None 时取 Y = X
    返回:
      K: shape = (d_X, d_Y)
      注意：对于两个不同视图，d_X 与 d_Y 可能不同，此时 np.dot(X, Y.T) 得到 (d_X, d_Y)。
    """
    if Y is None:
        Y = X
    X_sq = np.sum(X**2, axis=1, keepdims=True)  # shape (d_X, 1)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T  # shape (1, d_Y)
    dist_sq = X_sq + Y_sq - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    max_exp = 100  # clip 以防下溢
    dist_sq_clip = np.minimum(dist_sq / (2.0 * sigma**2), max_exp)
    K = np.exp(-dist_sq_clip)
    return K.astype(np.float32)


def center_kernel_square(K):
    """
    对方阵 K (d×d) 进行中心化: K' = H K H, 其中 H = I_d - (1/d)*11^T
    """
    d = K.shape[0]
    one_d = np.ones((d, d), dtype=K.dtype) / d
    I = np.eye(d, dtype=K.dtype)
    H = I - one_d
    return H @ K @ H


def center_cross_kernel(K12):
    """
    对矩阵 K12 (d1×d2) 做“双边中心化”：
      K12' = H1 * K12 * H2
    其中：
      H1 = I_d1 - 1/d1 * (1_{d1} 1_{d1}^T)
      H2 = I_d2 - 1/d2 * (1_{d2} 1_{d2}^T)
    """
    d1, d2 = K12.shape
    H1 = np.eye(d1, dtype=K12.dtype) - np.ones((d1, d1), dtype=K12.dtype) / d1
    H2 = np.eye(d2, dtype=K12.dtype) - np.ones((d2, d2), dtype=K12.dtype) / d2
    return H1 @ K12 @ H2


def matrix_inv_sqrt(M, reg=1e-12):
    """
    对称正定矩阵 M （加上 regI 后）的逆平方根。
    """
    M_reg = M + reg * np.eye(M.shape[0], dtype=M.dtype)
    U, S, Vt = np.linalg.svd(M_reg, full_matrices=False)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + reg))
    return (U @ S_inv_sqrt @ Vt).astype(np.float32)


class FeatureLevelKCCA:
    """
    核 CCA 实现：
      - 采用数据排列为 (d, n) ：d 为特征维度，n 为样本数
      - 训练时构造 K11, K22, K12 分别为两个视图的 RBF 核矩阵，
        对 K11, K22 做中心化，对 K12 做双边中心化，
        再正则化、求逆平方根、构造 M = A K12 B 后做 SVD，
        得到投影矩阵 W1 = A U, W2 = B V。
      - 新样本投影时：Z = (W.T @ X).T
    """

    def __init__(self, sigma=1.0, n_components=2, center=True, reg=1e-12):
        """
        :param sigma: RBF核宽度
        :param n_components: 投影目标维度
        :param center: 是否对核矩阵做中心化
        :param reg: 防止奇异的正则化项
        """
        self.sigma = sigma
        self.n_components = n_components
        self.center = center
        self.reg = reg
        self.W1 = None  # 对应 view1 的投影矩阵，形状 (d1, n_components)
        self.W2 = None  # 对应 view2 的投影矩阵，形状 (d2, n_components)

    def fit(self, X1, X2):
        """
        根据两个视图的训练数据（排列为 (d, n)）训练 KCCA 模型。
        流程：
          1) 构造核矩阵：K11, K22, K12
          2) 中心化（若 center=True）：分别对 K11, K22 和 K12 做中心化
          3) 正则化：K11_reg = K11 + reg*I, K22_reg = K22 + reg*I
          4) 求逆平方根：A = (K11_reg)^(-1/2), B = (K22_reg)^(-1/2)
          5) M = A * K12 * B，SVD(M)
          6) 得到投影矩阵：W1 = A * U_r, W2 = B * V_r  （U_r, V_r 为截断 SVD 得到的前 n_components 列）
        """
        # 1) 构造核矩阵
        K11 = rbf_kernel_features(X1, X1, sigma=self.sigma)  # shape (d1, d1)
        K22 = rbf_kernel_features(X2, X2, sigma=self.sigma)  # shape (d2, d2)
        K12 = rbf_kernel_features(X1, X2, sigma=self.sigma)  # shape (d1, d2)

        # 2) 中心化
        if self.center:
            K11 = center_kernel_square(K11)
            K22 = center_kernel_square(K22)
            K12 = center_cross_kernel(K12)

        # 3) 加正则项
        K11_reg = K11 + self.reg * np.eye(K11.shape[0], dtype=K11.dtype)
        K22_reg = K22 + self.reg * np.eye(K22.shape[0], dtype=K22.dtype)

        # 4) 逆平方根
        A = matrix_inv_sqrt(K11_reg, reg=0.0)
        B = matrix_inv_sqrt(K22_reg, reg=0.0)

        # 5) 构造 M 并做 SVD
        M = A @ K12 @ B  # shape (d1, d2)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        U_r = U[:, :self.n_components]     # shape (d1, n_components)
        V_r = Vt[:self.n_components, :].T    # shape (d2, n_components)

        # 6) 得到投影矩阵
        self.W1 = A @ U_r  # (d1, n_components)
        self.W2 = B @ V_r  # (d2, n_components)

    def transform(self, X, view=1):
        """
        将新样本 X 映射到低维空间。
        输入：
          X: 数据矩阵，排列为 (d, n_new)
          view: 1 或 2，指明使用哪个视图的投影矩阵
        返回：
          投影后数据，形状 (n_new, n_components)
        """
        if view == 1:
            if self.W1 is None:
                raise ValueError("模型尚未训练，请先调用 fit()。")
            return (self.W1.T @ X).T
        else:
            if self.W2 is None:
                raise ValueError("模型尚未训练，请先调用 fit()。")
            return (self.W2.T @ X).T


# ============== 第B部分：数据加载与特征提取 ==============

def extract_rgb_histogram(img, bins=16):
    """
    提取 RGB 三通道直方图，归一化后返回 3*bins 维向量。
    """
    h_r = np.histogram(img.getchannel('R'), bins=bins, range=(0, 255))[0]
    h_g = np.histogram(img.getchannel('G'), bins=bins, range=(0, 255))[0]
    h_b = np.histogram(img.getchannel('B'), bins=bins, range=(0, 255))[0]
    hist = np.concatenate([h_r, h_g, h_b], axis=0)
    hist = hist.astype(np.float32) / (img.size[0] * img.size[1])
    return hist


def extract_gray_histogram(img, bins=16):
    """
    提取灰度直方图，归一化后返回 bins 维向量。
    """
    gray_img = img.convert('L')
    h = np.histogram(gray_img, bins=bins, range=(0, 255))[0]
    h = h.astype(np.float32) / (img.size[0] * img.size[1])
    return h


def load_image_data_as_samples(dataset_path, categories, img_size=(224, 224), sample_ratio=1.0):
    """
    加载图像数据，每张图像作为一个样本。
    返回：
      X1: RGB 直方图数组，形状 (n_samples, 48)
      X2: 灰度直方图数组，形状 (n_samples, 16)
      labels: (n_samples,)
    """
    if not 0 < sample_ratio <= 1.0:
        raise ValueError("sample_ratio 必须在 (0, 1] 内")

    X1_list, X2_list, labels_list = [], [], []
    for label, category in enumerate(categories):
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
                feat1 = extract_rgb_histogram(img)  # 48 维向量
                feat2 = extract_gray_histogram(img)   # 16 维向量
                X1_list.append(feat1)
                X2_list.append(feat2)
                labels_list.append(label)
            except Exception as e:
                print(f"读取图像 {fp} 失败: {e}")

    X1 = np.array(X1_list, dtype=np.float32)  # (n_samples, 48)
    X2 = np.array(X2_list, dtype=np.float32)  # (n_samples, 16)
    labels = np.array(labels_list, dtype=np.int32)
    return X1, X2, labels


def train_test_split_on_samples(X1, X2, labels, test_size, random_state=42):
    """
    对样本进行拆分，保证 X1, X2 及标签均同步拆分。
    """
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        X1, X2, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    return X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te


# ============== 第C部分：评估函数 (1-NN 与 LightGBM) ==============

def evaluate_kcca_nn(X1_train, X2_train, y_train,
                     X1_test, X2_test, y_test,
                     n_components_vals, sigma_vals,
                     csv_path="kcca_nn_results.csv",
                     random_seed=42,
                     verbose=True):
    """
    利用 1-NN 评估 KCCA 投影效果，并记录 Accuracy 与 F1 指标。
    """
    results = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "sigma", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for s in sigma_vals:
            if verbose:
                print(f"\n[Evaluation] KCCA 1-NN => n_components={n_comp}, sigma={s}")

            kcca = FeatureLevelKCCA(
                sigma=s,
                n_components=n_comp,
                center=True,
                reg=1e-6
            )
            kcca.fit(X1_train, X2_train)

            # 对训练样本投影：注意输入需排列为 (d, n)
            Z1_train = kcca.transform(X1_train, view=1)  # (n_train, n_components)
            Z2_train = kcca.transform(X2_train, view=2)  # (n_train, n_components)
            X_train_kcca = np.hstack([Z1_train, Z2_train])

            Z1_test = kcca.transform(X1_test, view=1)  # (n_test, n_components)
            Z2_test = kcca.transform(X2_test, view=2)  # (n_test, n_components)
            X_test_kcca = np.hstack([Z1_test, Z2_test])

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

            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n_comp, s, f"{acc:.4f}", f"{f1:.4f}"])

    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[1-NN on KCCA: Top 5 Results]")
    print("Rank\tn_components\tsigma\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['sigma']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_kcca_lightgbm(X1_train, X2_train, y_train,
                           X1_test, X2_test, y_test,
                           n_components_vals, sigma_vals,
                           csv_path="kcca_lightgbm_results.csv",
                           random_seed=42,
                           verbose=True):
    """
    利用 LightGBM 评估 KCCA 投影效果，并记录 Accuracy 与 F1 指标。
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
                print(f"\n[Evaluation] KCCA LightGBM => n_components={n_comp}, sigma={s}")

            kcca = FeatureLevelKCCA(
                sigma=s,
                n_components=n_comp,
                center=True,
                reg=1e-6
            )
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

            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n_comp, s, f"{acc:.4f}", f"{f1:.4f}"])

    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[LightGBM on KCCA: Top 5 Results]")
    print("Rank\tn_components\tsigma\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['sigma']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


# ============== 第D部分：主函数 ==============

def main():
    # 修改为你自己的数据集路径
    dataset_path = r"E:\jqxx\dataset\archive1"
    categories = ["COVID", "non-COVID"]

    sample_ratio = 1.0
    test_ratio = 0.2
    random_seed = 42

    print(f"正在从 {dataset_path} 加载数据...")
    X1_samples, X2_samples, labels = load_image_data_as_samples(
        dataset_path=dataset_path,
        categories=categories,
        img_size=(224, 224),
        sample_ratio=sample_ratio
    )

    print("\n数据加载完成。信息：")
    print("X1 (样本为行) shape:", X1_samples.shape)  # (n, 48)
    print("X2 (样本为行) shape:", X2_samples.shape)  # (n, 16)
    print("labels shape:", labels.shape)

    # 样本划分（此时数据排列为 (n, d)）
    X1_tr_samp, X1_te_samp, X2_tr_samp, X2_te_samp, y_train, y_test = train_test_split_on_samples(
        X1_samples, X2_samples, labels,
        test_size=test_ratio,
        random_state=random_seed
    )
    print("\ntrain / test 划分后:")
    print("X1_train (未转置) shape =", X1_tr_samp.shape)
    print("X1_test (未转置) shape  =", X1_te_samp.shape)
    print("y_train.shape =", y_train.shape)
    print("y_test.shape  =", y_test.shape)

    # 注意：FeatureLevelKCCA 要求数据排列为 (d, n)，因此需要转置
    X1_train = X1_tr_samp.T  # (48, n_train)
    X2_train = X2_tr_samp.T  # (16, n_train)
    X1_test = X1_te_samp.T   # (48, n_test)
    X2_test = X2_te_samp.T   # (16, n_test)
    print("\n转置后:")
    print("X1_train.shape =", X1_train.shape)
    print("X2_train.shape =", X2_train.shape)
    print("X1_test.shape  =", X1_test.shape)
    print("X2_test.shape  =", X2_test.shape)

    # 设定超参数搜索范围
    n_components_vals = [2, 4, 8, 12]
    sigma_vals = [0.01, 0.1, 1.0, 10.0]

    # 评估：1-NN
    print("\n========== [Evaluation 1] 1-NN (KCCA) ==========")
    evaluate_kcca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="kcca_nn_results.csv",
        random_seed=random_seed,
        verbose=True
    )

    # 评估：LightGBM
    print("\n========== [Evaluation 2] LightGBM (KCCA) ==========")
    evaluate_kcca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="kcca_lightgbm_results.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n>>> KCCA 实验结束，Done.")


if __name__ == "__main__":
    main()