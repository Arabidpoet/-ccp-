
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


# ============== 第A部分：RCCA方法模块 ==============
class FeatureLevelRCCA:
    """
    基于特征级（Feature-level）的正则化典型相关分析 (RCCA) 算法实现。
    使用了 Tikhonov 正则 (C + reg*I)，并采用 (n - 1) 计算样本协方差。
    """

    def __init__(self, n_components=2, reg1=0.1, reg2=0.1):
        """
        :param n_components: 降维目标维度（即要提取的典型相关对数）
        :param reg1: 第一个视图的正则化参数
        :param reg2: 第二个视图的正则化参数
        """
        self.n_components = n_components
        self.reg1 = reg1
        self.reg2 = reg2
        self.W1 = None
        self.W2 = None
        self.mean1 = None
        self.mean2 = None

    def fit(self, X1, X2):
        """
        计算投影矩阵 W1, W2。
        X1, X2 格式: (d1, n), (d2, n)，即 “以特征为行”，样本为列。
        """
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("X1 和 X2 的样本数量（列数）必须相同。")

        # 1) 计算并减去均值
        self.mean1 = np.mean(X1, axis=1, keepdims=True)
        self.mean2 = np.mean(X2, axis=1, keepdims=True)
        X1_centered = X1 - self.mean1  # shape = (d1, n)
        X2_centered = X2 - self.mean2  # shape = (d2, n)

        n = X1.shape[1]  # 样本数量

        # 2) 计算加上正则项的协方差矩阵
        #    注意在统计中常使用 1/(n-1)，与 1/n 相比更贴近“样本方差”
        #    下面在矩阵主对角线上加上 reg 调整，实现岭回归式正则
        C11 = np.dot(X1_centered, X1_centered.T) / (n - 1) + self.reg1 * np.eye(X1.shape[0])
        C22 = np.dot(X2_centered, X2_centered.T) / (n - 1) + self.reg2 * np.eye(X2.shape[0])
        C12 = np.dot(X1_centered, X2_centered.T) / (n - 1)

        # 3) 计算广义特征值问题所需的矩阵
        inv_C11 = np.linalg.inv(C11)
        inv_C22 = np.linalg.inv(C22)

        M1 = inv_C11 @ C12 @ inv_C22 @ C12.T
        M2 = inv_C22 @ C12.T @ inv_C11 @ C12

        # 4) 求解特征值与特征向量（因为都是对称矩阵，用 np.linalg.eigh 更稳妥）
        eig_vals1, eig_vecs1 = np.linalg.eigh(M1)
        eig_vals2, eig_vecs2 = np.linalg.eigh(M2)

        # 5) 选择最大的 n_components 个特征值对应的特征向量 (从大到小排序)
        idx1 = np.argsort(eig_vals1)[::-1][:self.n_components]
        idx2 = np.argsort(eig_vals2)[::-1][:self.n_components]

        W1_raw = eig_vecs1[:, idx1]  # shape = (d1, n_components)
        W2_raw = eig_vecs2[:, idx2]  # shape = (d2, n_components)

        # 6) 对投影矩阵进行归一化，使得每个投影向量的 L2 范数为 1
        #    这样可以避免向量过大或过小的问题，但不改变其方向。
        self.W1 = W1_raw / np.sqrt(np.sum(W1_raw ** 2, axis=0, keepdims=True))
        self.W2 = W2_raw / np.sqrt(np.sum(W2_raw ** 2, axis=0, keepdims=True))

    def transform(self, X, view=1):
        """
        将 (d, n) 形状的数据投影到 (n, n_components) 空间。
        :param view: 选择使用哪个视图的投影矩阵 (1 or 2)
        :return: 投影后得到的 (n, n_components) 矩阵
        """
        if view == 1:
            if self.W1 is None:
                raise ValueError("W1 未被 fit() 初始化。请先调用 fit()。")
            X_centered = X - self.mean1
            return (self.W1.T @ X_centered).T  # shape => (n, n_components)
        else:
            if self.W2 is None:
                raise ValueError("W2 未被 fit() 初始化。请先调用 fit()。")
            X_centered = X - self.mean2
            return (self.W2.T @ X_centered).T  # shape => (n, n_components)


# ============== 第B部分：数据加载与特征提取 ==============
def extract_rgb_histogram(img, bins=16):
    """
    提取 RGB 三通道的直方图特征，并进行归一化 (除以像素总数)。
    返回 shape = (3*bins, ) 的一维向量。
    """
    h_r = np.histogram(img.getchannel('R'), bins=bins, range=(0, 255))[0]
    h_g = np.histogram(img.getchannel('G'), bins=bins, range=(0, 255))[0]
    h_b = np.histogram(img.getchannel('B'), bins=bins, range=(0, 255))[0]
    hist = np.concatenate([h_r, h_g, h_b], axis=0)
    hist = hist.astype(np.float32) / (img.size[0] * img.size[1])
    return hist


def extract_gray_histogram(img, bins=16):
    """
    提取 Gray 单通道的直方图特征，并进行归一化 (除以像素总数)。
    返回 shape = (bins, ) 的一维向量。
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
    加载图像数据, 将每张图作为一条样本, 得到:
      X1: RGB histogram 特征 => shape = (n_samples, 3*bins)
      X2: Gray histogram 特征 => shape = (n_samples, bins)
      labels => shape = (n_samples,)
    参数 sample_ratio 用于随机采样一定比例的数据。
    """
    if not 0 < sample_ratio <= 1.0:
        raise ValueError("sample_ratio 必须在 (0,1] 范围内")

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

                feat1 = extract_rgb_histogram(img)  # e.g. 48-dim
                feat2 = extract_gray_histogram(img)  # e.g. 16-dim

                X1_list.append(feat1)
                X2_list.append(feat2)
                labels_list.append(label)
            except Exception as e:
                print(f"读取图像 {fp} 失败: {e}")

    X1 = np.array(X1_list, dtype=np.float32)  # shape = (n_samples, 3*bins)
    X2 = np.array(X2_list, dtype=np.float32)  # shape = (n_samples, bins)
    labels = np.array(labels_list, dtype=np.int32)
    return X1, X2, labels


def train_test_split_on_samples(X1, X2, labels, test_size, random_state=42):
    """
    对"样本"进行切分：保持 X1, X2 同样的拆分顺序.
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


# ============== 第C部分：评估函数 ==============
def evaluate_rcca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, reg_vals,
        csv_path="nn_results_rcca.csv",
        random_seed=42,
        verbose=True
):
    """
    使用 1-NN（1-最近邻）来评估 RCCA 投影后的分类效果，并记录 ACC 和 F1 指标。
    会遍历给定的 n_components_vals 和 reg_vals，然后将结果保存到 CSV。
    """
    results = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "reg", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for reg in reg_vals:
            if verbose:
                print(f"\n[Evaluation] 1-NN => n_components={n_comp}, reg={reg}")

            # 定义 RCCA (以相同的正则参数简化处理)
            rcca = FeatureLevelRCCA(n_components=n_comp, reg1=reg, reg2=reg)

            # (d, n) 的转置 => (n, d)
            # 注意：我们的数据当前是 (n_samples, dim) 形状，所以需转置成 (dim, n)
            rcca.fit(X1_train.T, X2_train.T)

            # 投影
            Z1_train = rcca.transform(X1_train.T, view=1)
            Z2_train = rcca.transform(X2_train.T, view=2)
            X_train_rcca = np.hstack([Z1_train, Z2_train])  # shape=(n_samples, 2*n_components)

            Z1_test = rcca.transform(X1_test.T, view=1)
            Z2_test = rcca.transform(X2_test.T, view=2)
            X_test_rcca = np.hstack([Z1_test, Z2_test])

            # 1-NN
            nnclf = KNeighborsClassifier(n_neighbors=1)
            nnclf.fit(X_train_rcca, y_train)
            y_pred = nnclf.predict(X_test_rcca)

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

    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[1-NN Top 5 Results]")
    print("Rank\tn_comp\treg\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['reg']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_rcca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, reg_vals,
        csv_path="lightgbm_results_rcca.csv",
        random_seed=42,
        verbose=True
):
    """
    使用 LightGBM 来评估 RCCA 投影后的分类效果，并记录 ACC 和 F1 指标。
    同样遍历 n_components_vals 和 reg_vals，将结果保存到 CSV。
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

            rcca = FeatureLevelRCCA(n_components=n_comp, reg1=reg, reg2=reg)
            rcca.fit(X1_train.T, X2_train.T)

            Z1_train = rcca.transform(X1_train.T, view=1)
            Z2_train = rcca.transform(X2_train.T, view=2)
            X_train_rcca = np.hstack([Z1_train, Z2_train])

            Z1_test = rcca.transform(X1_test.T, view=1)
            Z2_test = rcca.transform(X2_test.T, view=2)
            X_test_rcca = np.hstack([Z1_test, Z2_test])

            clf = LGBMClassifier(**lgb_params)
            clf.fit(X_train_rcca, y_train)
            y_pred = clf.predict(X_test_rcca)

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

    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[LightGBM Top 5 Results]")
    print("Rank\tn_comp\treg\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['reg']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

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

    X1_tr_samp, X1_te_samp, X2_tr_samp, X2_te_samp, y_train, y_test = train_test_split_on_samples(
        X1_samples, X2_samples, labels,
        test_size=test_ratio,
        random_state=random_seed
    )

    # RCCA 的实现中，X1, X2 的形状是 (d, n)（特征为行，样本为列）。
    # 而上面拆分好的 X1_tr_samp, X2_tr_samp 等是 (n, d)（样本为行，特征为列）。
    # 因此在调用 fit() 或 transform() 前，需要进行转置。

    n_components_vals = [8, 9, 10, 12, 15, 24]
    reg_vals = [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5]

    print("\n========== [Evaluation 1] 1-NN with RCCA ==========")
    evaluate_rcca_nn(
        X1_tr_samp, X2_tr_samp, y_train,
        X1_te_samp, X2_te_samp, y_test,
        n_components_vals=n_components_vals,
        reg_vals=reg_vals,
        csv_path="nn_results_rcca.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n========== [Evaluation 2] LightGBM with RCCA ==========")
    evaluate_rcca_lightgbm(
        X1_tr_samp, X2_tr_samp, y_train,
        X1_te_samp, X2_te_samp, y_test,
        n_components_vals=n_components_vals,
        reg_vals=reg_vals,
        csv_path="lightgbm_results_rcca.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n>>> Done.")


if __name__ == "__main__":
    main()