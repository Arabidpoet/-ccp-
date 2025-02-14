
import os
import csv
import warnings
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    normalized_mutual_info_score, precision_score, recall_score
)
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============== 第A部分：CCP方法模块 ==============
def rbf_kernel_features(X, Y=None, sigma=1.0):
    """
    RBF kernel (Gaussian)，针对 "以特征为行" 的矩阵:
      - X.shape = (dX, n)
      - Y.shape = (dY, n)
    X[i,:] 表示第 i 个特征在 n 个样本上的取值 => 向量 ∈ ℝ^n
    则返回 K.shape = (dX, dY)，
    K[i,j] = exp(-||X[i,:]-Y[j,:]||^2 / (2*sigma^2))
    """
    if Y is None:
        Y = X
    X_sq = np.sum(X**2, axis=1, keepdims=True)  # shape(dX,1)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T  # shape(1,dY)
    dist_sq = X_sq + Y_sq - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)  # clip < 0
    max_exp = 100
    dist_sq_clip = np.minimum(dist_sq / (2.0 * sigma**2), max_exp)
    K = np.exp(-dist_sq_clip)
    return K.astype(np.float32)


def center_kernel(K):
    """
    对特征核矩阵 K (d × d) 做中心化:
    K' = H K H, 其中 H = I_d - (1/d)*11^T
    """
    d = K.shape[0]
    one_d = np.ones((d, d), dtype=np.float32) / d
    I = np.eye(d, dtype=np.float32)
    H = I - one_d
    return H @ K @ H


def matrix_inv_sqrt(M, reg=1e-12):
    """
    对称正定矩阵 M 的(加上 regI 后)逆平方根.
    """
    M_reg = M + reg * np.eye(M.shape[0], dtype=M.dtype)
    U, S, Vt = np.linalg.svd(M_reg, full_matrices=False)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + reg))
    return (U @ S_inv_sqrt @ Vt).astype(np.float32)


class FeatureLevelCCP:

    def __init__(self, sigma=1.0, n_components=2, center=True, reg=1e-12):
        """
        :param sigma: RBF核宽度
        :param n_components: 降维目标维度
        :param center: 是否对核矩阵做中心化
        :param reg: 防止奇异的正则化项
        """
        self.sigma = sigma
        self.n_components = n_components
        self.center = center
        self.reg = reg
        self.W1 = None
        self.W2 = None

    def fit(self, X1, X2):
        """
        计算 W1, W2.
        X1, X2 格式: (d1, n), (d2, n)，即 "以特征为行"。
        """
        # 1) 构造 F-intraset & F-interset 核矩阵
        K11 = rbf_kernel_features(X1, X1, sigma=self.sigma)  # (d1,d1)
        K22 = rbf_kernel_features(X2, X2, sigma=self.sigma)  # (d2,d2)
        K12 = rbf_kernel_features(X1, X2, sigma=self.sigma)  # (d1,d2)

        # 2) (可选)对 K11, K22 做中心化
        if self.center:
            K11 = center_kernel(K11)
            K22 = center_kernel(K22)
            # K12 是否双边中心化

        # 3) 对 K11, K22 求逆平方根
        A = matrix_inv_sqrt(K11, reg=self.reg)  # (d1,d1)
        B = matrix_inv_sqrt(K22, reg=self.reg)  # (d2,d2)

        # 4) 构造 M = A * K12 * B
        M = A @ K12 @ B  # shape(d1, d2)

        # 5) 对 M 做 SVD
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        U_r = U[:, :self.n_components]     # (d1, n_components)
        V_r = Vt[:self.n_components, :].T  # (d2, n_components)

        # 6) 得到投影矩阵
        self.W1 = A @ U_r  # (d1, n_components)
        self.W2 = B @ V_r  # (d2, n_components)

    def transform(self, X, view=1):
        """
        将 (d, n) 数据投影到 (n, n_components).
        """
        if view == 1:
            if self.W1 is None:
                raise ValueError("W1 is not fitted yet. Call fit() first.")
            # W1.shape = (d1, n_components)
            # 输出 => (n, n_components)
            return (self.W1.T @ X).T
        else:
            if self.W2 is None:
                raise ValueError("W2 is not fitted yet. Call fit() first.")
            return (self.W2.T @ X).T


# ============== 第B部分：数据加载与特征提取 ==============
def extract_rgb_histogram(img, bins=16):
    """
    提取 RGB 三通道的直方图特征，并进行归一化
    """
    h_r = np.histogram(img.getchannel('R'), bins=bins, range=(0, 255))[0]
    h_g = np.histogram(img.getchannel('G'), bins=bins, range=(0, 255))[0]
    h_b = np.histogram(img.getchannel('B'), bins=bins, range=(0, 255))[0]
    hist = np.concatenate([h_r, h_g, h_b], axis=0)
    hist = hist.astype(np.float32) / (img.size[0] * img.size[1])
    return hist


def extract_gray_histogram(img, bins=16):
    """
    提取 Gray 单通道的直方图特征，并进行归一化
    """
    gray_img = img.convert('L')
    h = np.histogram(gray_img, bins=bins, range=(0, 255))[0]
    h = h.astype(np.float32) / (img.size[0] * img.size[1])
    return h


def load_image_data_as_samples(
    dataset_path,
    categories,
    img_size=(224, 224),
    sample_ratio=1.0
):
    """
    加载图像数据, 在初始阶段将每张图当作一条样本, 得到:
      X1: RGB histogram 特征 => shape = (n_samples, dim1)
      X2: Gray histogram 特征 => shape = (n_samples, dim2)
      labels => shape = (n_samples,)
    """
    if not 0 < sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0,1]")

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
                feat1 = extract_rgb_histogram(img)   # e.g. 48-dim
                feat2 = extract_gray_histogram(img)  # e.g. 16-dim
                X1_list.append(feat1)
                X2_list.append(feat2)
                labels_list.append(label)
            except Exception as e:
                print(f"读取图像 {fp} 失败: {e}")

    X1 = np.array(X1_list, dtype=np.float32)  # shape=(n_samples, dim1)
    X2 = np.array(X2_list, dtype=np.float32)  # shape=(n_samples, dim2)
    labels = np.array(labels_list, dtype=np.int32)
    return X1, X2, labels


def train_test_split_on_samples(X1, X2, labels, test_size, random_state=42):
    """
    对"样本"进行切分：保持 X1, X2 同样顺序拆分.
    返回:
      X1_train, X1_test, X2_train, X2_test, y_train, y_test
    """
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        X1, X2, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    return X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te


# ============== 第C部分：评估函数 (1-NN & LightGBM) ==============
def evaluate_ccp_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, sigma_vals,
        csv_path="nn_results.csv",
        random_seed=42,
        verbose=True
):
    """
    1-NN评估CCP投影效果，并记录Accuracy和F1指标。
    """
    results = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "sigma", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for s in sigma_vals:
            if verbose:
                print(f"\n[Evaluation] 1-NN => n_components={n_comp}, sigma={s}")

            ccp = FeatureLevelCCP(sigma=s, n_components=n_comp, center=True, reg=1e-6)
            ccp.fit(X1_train, X2_train)

            Z1_train = ccp.transform(X1_train, view=1)
            Z2_train = ccp.transform(X2_train, view=2)
            X_train_ccp = np.hstack([Z1_train, Z2_train])

            Z1_test = ccp.transform(X1_test, view=1)
            Z2_test = ccp.transform(X2_test, view=2)
            X_test_ccp = np.hstack([Z1_test, Z2_test])

            # 1-NN
            nnclf = KNeighborsClassifier(n_neighbors=1)
            nnclf.fit(X_train_ccp, y_train)
            y_pred = nnclf.predict(X_test_ccp)

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

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[1-NN Top 5 Results]")
    print("Rank\tn_comp\tsigma\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['sigma']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_ccp_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, sigma_vals,
        csv_path="lightgbm_results.csv",
        random_seed=42,
        verbose=True
):
    """
    LightGBM评估CCP投影效果，记录Accuracy和F1指标。
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
                print(f"\n[Evaluation] LightGBM => n_components={n_comp}, sigma={s}")

            ccp = FeatureLevelCCP(sigma=s, n_components=n_comp, center=True, reg=1e-6)
            ccp.fit(X1_train, X2_train)

            Z1_train = ccp.transform(X1_train, view=1)
            Z2_train = ccp.transform(X2_train, view=2)
            X_train_ccp = np.hstack([Z1_train, Z2_train])

            Z1_test = ccp.transform(X1_test, view=1)
            Z2_test = ccp.transform(X2_test, view=2)
            X_test_ccp = np.hstack([Z1_test, Z2_test])

            clf = LGBMClassifier(**lgb_params)
            clf.fit(X_train_ccp, y_train)
            y_pred = clf.predict(X_test_ccp)

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

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[LightGBM Top 5 Results]")
    print("Rank\tn_comp\tsigma\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['sigma']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


# ============== 第D部分：主函数 ==============
def main():
    dataset_path = r"E:\jqxx\dataset\archive1"  # <-- 请根据实际路径修改
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
    print("X1 (样本为行) shape:", X1_samples.shape)
    print("X2 (样本为行) shape:", X2_samples.shape)
    print("labels shape:", labels.shape)

    # 1) 样本级划分
    X1_tr_samp, X1_te_samp, X2_tr_samp, X2_te_samp, y_train, y_test = train_test_split_on_samples(
        X1_samples, X2_samples, labels,
        test_size=test_ratio,
        random_state=random_seed
    )
    print("\ntrain / test after split:")
    print("X1_train.shape =", X1_tr_samp.shape)
    print("X1_test.shape  =", X1_te_samp.shape)
    print("y_train.shape  =", y_train.shape)
    print("y_test.shape   =", y_test.shape)

    # 2) 转置(以特征为行, 样本为列)
    X1_train = X1_tr_samp.T  # shape(d1, nTrain)
    X2_train = X2_tr_samp.T  # shape(d2, nTrain)
    X1_test = X1_te_samp.T  # shape(d1, nTest)
    X2_test = X2_te_samp.T  # shape(d2, nTest)

    print("\n转置后:")
    print("X1_train.shape =", X1_train.shape)
    print("X2_train.shape =", X2_train.shape)
    print("X1_test.shape  =", X1_test.shape)
    print("X2_test.shape  =", X2_test.shape)

    # 3) 超参数范围
    n_components_vals = [8,9,10,11,12,13,14,15,24,25,26,]
    sigma_vals = [0.1,0.2,0.3,0.4,0.5,0.7,]

    # 4) 评估: 1-NN
    print("\n========== [Evaluation 1] 1-NN ==========")
    evaluate_ccp_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="nn_results.csv",
        random_seed=random_seed,
        verbose=True
    )

    # 5) 评估: LightGBM
    print("\n========== [Evaluation 2] LightGBM ==========")
    evaluate_ccp_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="lightgbm_results.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n>>> Done.")


if __name__ == "__main__":
    main()