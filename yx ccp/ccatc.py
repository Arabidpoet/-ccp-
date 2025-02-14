
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
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============== 第A部分：CCA方法模块 ==============
class CCA:
    """
    1) 对输入数据 X1, X2 进行标准化
    2) 计算 C11, C22, C12
    3) 计算 C11^-1/2 和 C22^-1/2
    4) 对 M = C11^-1/2 * C12 * C22^-1/2 做 SVD
    5) 得到投影矩阵 w1, w2，用于将 X1, X2 映射到各自的典型相关空间
    """

    def __init__(self, n_components=2, reg=1e-4):
        """
        参数:
        ----
        n_components: int
            需要提取的典型相关分量数
        reg: float
            用于在协方差矩阵上加的对角线上正则化系数
        """
        self.n_components = n_components
        self.reg = reg
        self.w1 = None
        self.w2 = None
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()

    def fit(self, X1, X2):
        """
        计算 CCA 投影矩阵 w1, w2。

        参数:
        ----
        X1: ndarray of shape (d1, n)
            第一视图数据，d1 为特征维度，n 为样本数
        X2: ndarray of shape (d2, n)
            第二视图数据，d2 为特征维度，n 为样本数
        """
        # 1. Standardize，注意这里 X1, X2 的 shape = (d, n)。
        X1_std = self.scaler1.fit_transform(X1.T).T  # shape 仍然是 (d1, n)
        X2_std = self.scaler2.fit_transform(X2.T).T  # shape 仍然是 (d2, n)

        n = X1_std.shape[1]  # 样本数

        # 2. 计算加了正则项的协方差矩阵
        C11 = (X1_std @ X1_std.T) / (n - 1) + self.reg * np.eye(X1_std.shape[0])
        C22 = (X2_std @ X2_std.T) / (n - 1) + self.reg * np.eye(X2_std.shape[0])
        C12 = (X1_std @ X2_std.T) / (n - 1)

        # 3. 计算 C11^-1/2 和 C22^-1/2
        inv_sqrt_C11 = matrix_inv_sqrt(C11)
        inv_sqrt_C22 = matrix_inv_sqrt(C22)

        # 4. 对 M = C11^-1/2 * C12 * C22^-1/2 做 SVD
        M = inv_sqrt_C11 @ C12 @ inv_sqrt_C22
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # 5. 得到投影矩阵 w1, w2
        self.w1 = inv_sqrt_C11 @ U[:, :self.n_components]         # shape = (d1, n_components)
        self.w2 = inv_sqrt_C22 @ Vt.T[:, :self.n_components]       # shape = (d2, n_components)

    def transform(self, X, view=1):
        """
        将数据投影到 CCA 空间。

        参数:
        ----
        X: ndarray of shape (d, n)
            要投影的数据
        view: int
            如果 view=1，则使用 w1 进行投影；如果 view=2，则使用 w2 进行投影。

        返回:
        ----
        Z: ndarray of shape (n, n_components)
            投影后的特征
        """
        if view == 1:
            if self.w1 is None:
                raise ValueError("CCA 模型尚未 fit。请先调用 fit() 方法。")
            X_std = self.scaler1.transform(X.T).T
            return (self.w1.T @ X_std).T
        else:
            if self.w2 is None:
                raise ValueError("CCA 模型尚未 fit。请先调用 fit() 方法。")
            X_std = self.scaler2.transform(X.T).T
            return (self.w2.T @ X_std).T


def matrix_inv_sqrt(M):
    """
    计算矩阵的逆平方根 (M^-1/2)。
    """
    # 在此不再添加 reg，保持一次正则即可。
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S))
    return (U @ S_inv_sqrt @ Vt).astype(np.float32)


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
    从 dataset_path 路径中依次读取指定的类别文件夹，按类别加载图像。
    返回对每张图像提取到的特征 X1, X2，以及对应的标签。
    """
    if not 0 < sample_ratio <= 1.0:
        raise ValueError("sample_ratio 必须在 (0,1] 区间内。")

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
                feat1 = extract_rgb_histogram(img)
                feat2 = extract_gray_histogram(img)
                X1_list.append(feat1)
                X2_list.append(feat2)
                labels_list.append(label)
            except Exception as e:
                print(f"读取图像 {fp} 失败: {e}")

    X1 = np.array(X1_list, dtype=np.float32)
    X2 = np.array(X2_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    return X1, X2, labels


def train_test_split_on_samples(X1, X2, labels, test_size, random_state=42):
    """
    使用 sklearn 中的 train_test_split 将 X1, X2 以及标签进行拆分。
    返回对应的 X1_train, X1_test, X2_train, X2_test, y_train, y_test
    """
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        X1, X2, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    return X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te


# ============== 第C部分：评估函数 (1-NN & LightGBM) ==============
def evaluate_cca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, reg_vals,
        csv_path="nn_results_cca.csv",
        random_seed=42,
        verbose=True
):
    """
    评估：在不同的 n_components 和 reg 下，做 CCA 后使用 1-NN 分类器。
    并将结果记录在 csv 文件中。
    """
    results = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "reg", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for reg in reg_vals:
            if verbose:
                print(f"\n[Evaluation] 1-NN => n_components={n_comp}, reg={reg}")

            cca = CCA(n_components=n_comp, reg=reg)
            cca.fit(X1_train, X2_train)

            Z1_train = cca.transform(X1_train, view=1)
            Z2_train = cca.transform(X2_train, view=2)
            X_train_cca = np.hstack([Z1_train, Z2_train])

            Z1_test = cca.transform(X1_test, view=1)
            Z2_test = cca.transform(X2_test, view=2)
            X_test_cca = np.hstack([Z1_test, Z2_test])

            nnclf = KNeighborsClassifier(n_neighbors=1)
            nnclf.fit(X_train_cca, y_train)
            y_pred = nnclf.predict(X_test_cca)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'n_components': n_comp,
                'reg': reg,
                'accuracy': acc,
                'f1': f1
            })

            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n_comp, reg, f"{acc:.4f}", f"{f1:.4f}"])

    # 按照 f1 由高到低排序后打印 top 5
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[1-NN Top 5 Results]")
    print("Rank\tn_comp\treg\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['reg']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_cca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, reg_vals,
        csv_path="lightgbm_results_cca.csv",
        random_seed=42,
        verbose=True
):
    """
    评估：在不同的 n_components 和 reg 下，做 CCA 后使用 LightGBM 分类器。
    并将结果记录在 csv 文件中。
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
        writer.writerow(["n_components", "reg", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for reg in reg_vals:
            if verbose:
                print(f"\n[Evaluation] LightGBM => n_components={n_comp}, reg={reg}")

            cca = CCA(n_components=n_comp, reg=reg)
            cca.fit(X1_train, X2_train)

            Z1_train = cca.transform(X1_train, view=1)
            Z2_train = cca.transform(X2_train, view=2)
            X_train_cca = np.hstack([Z1_train, Z2_train])

            Z1_test = cca.transform(X1_test, view=1)
            Z2_test = cca.transform(X2_test, view=2)
            X_test_cca = np.hstack([Z1_test, Z2_test])

            clf = LGBMClassifier(**lgb_params)
            clf.fit(X_train_cca, y_train)
            y_pred = clf.predict(X_test_cca)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'n_components': n_comp,
                'reg': reg,
                'accuracy': acc,
                'f1': f1
            })

            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n_comp, reg, f"{acc:.4f}", f"{f1:.4f}"])

    # 按照 f1 由高到低排序后打印 top 5
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[LightGBM Top 5 Results]")
    print("Rank\tn_comp\treg\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['reg']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


# ============== 第D部分：主函数 ==============
def main():

    # 数据集路径
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

    # 拆分训练集、测试集
    X1_tr_samp, X1_te_samp, X2_tr_samp, X2_te_samp, y_train, y_test = train_test_split_on_samples(
        X1_samples, X2_samples, labels,
        test_size=test_ratio,
        random_state=random_seed
    )

    # 注意：CCA 的实现使用了 (d, n) 形式，因此需要转置
    X1_train = X1_tr_samp.T
    X2_train = X2_tr_samp.T
    X1_test = X1_te_samp.T
    X2_test = X2_te_samp.T

    print("\n转置后:")
    print("X1_train.shape =", X1_train.shape)
    print("X2_train.shape =", X2_train.shape)
    print("X1_test.shape  =", X1_test.shape)
    print("X2_test.shape  =", X2_test.shape)

    # CCA参数范围
    n_components_vals = [2, 4, 8, 12, 16]
    reg_vals = [1e-6, 1e-4, 1e-2]

    print("\n========== [Evaluation 1] 1-NN with CCA ==========")
    evaluate_cca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        reg_vals=reg_vals,
        csv_path="nn_results_cca.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n========== [Evaluation 2] LightGBM with CCA ==========")
    evaluate_cca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        reg_vals=reg_vals,
        csv_path="lightgbm_results_cca.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n>>> Done.")


if __name__ == "__main__":
    main()