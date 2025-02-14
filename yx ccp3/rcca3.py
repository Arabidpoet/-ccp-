
import os
import csv
import warnings
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
)
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============== 第A部分：rCCA方法模块 ==============
class FeatureLevelRCCA:
    """
    标准的、带有正则化的 Canonical Correlation Analysis（rCCA）实现。

    输入:
        n_components: 用于投影的潜在维度个数 (int)
        alpha: 正则化系数，取值>0 (float)

    用法:
        1) rcca = FeatureLevelRCCA(n_components=2, alpha=1e-3)
        2) rcca.fit(X1, X2)
        3) Z1 = rcca.transform(X1, view=1)
           Z2 = rcca.transform(X2, view=2)

    说明:
        假设 X1.shape = (d1, n), X2.shape = (d2, n)
          - d1, d2 分别是特征维度
          - n 是样本数（需确保两视图的样本数一致）
        输出 Z1.shape = (n, n_components), Z2.shape = (n, n_components)
        这里我们使用的是典型的正则化 CCA。
    """

    def __init__(self, n_components=2, alpha=1e-3):
        self.n_components = n_components
        self.alpha = alpha
        self.W1 = None
        self.W2 = None
        self.mean1_ = None
        self.mean2_ = None

    def fit(self, X1, X2):
        """
        训练 rCCA 模型
        X1: shape = (d1, n)
        X2: shape = (d2, n)
        """
        # 1) 零均值化
        self.mean1_ = np.mean(X1, axis=1, keepdims=True)
        self.mean2_ = np.mean(X2, axis=1, keepdims=True)
        X1_centered = X1 - self.mean1_
        X2_centered = X2 - self.mean2_

        # 2) 计算协方差矩阵
        #    S11 = (X1_centered @ X1_centered.T) / (n - 1) + alpha * I_d1
        #    S22 = (X2_centered @ X2_centered.T) / (n - 1) + alpha * I_d2
        #    S12 = (X1_centered @ X2_centered.T) / (n - 1)
        n = X1.shape[1]
        d1 = X1.shape[0]
        d2 = X2.shape[0]

        S11 = (X1_centered @ X1_centered.T) / (n - 1)
        S22 = (X2_centered @ X2_centered.T) / (n - 1)
        S12 = (X1_centered @ X2_centered.T) / (n - 1)

        # 正则化
        S11 += self.alpha * np.eye(d1)
        S22 += self.alpha * np.eye(d2)

        # 3) 某种等价的解法：S11^{-1/2} * S12 * S22^{-1/2} 做 SVD
        #    M = S11^{-1/2} * S12 * S22^{-1/2}
        #    M = U * Sigma * V^T
        #    W1 = S11^{-1/2} * U_r
        #    W2 = S22^{-1/2} * V_r
        #    注意: 这里需要 S11, S22 可逆；正则化保证了数值稳定。
        U1, D1, Vt1 = np.linalg.svd(S11, full_matrices=False)
        U2, D2, Vt2 = np.linalg.svd(S22, full_matrices=False)

        D1_inv_sqrt = np.diag(1.0 / np.sqrt(D1))
        D2_inv_sqrt = np.diag(1.0 / np.sqrt(D2))

        # 转为 float32 可选，若追求精度可去掉 .astype(np.float32)
        S11_inv_sqrt = (U1 @ D1_inv_sqrt @ Vt1).astype(np.float32)
        S22_inv_sqrt = (U2 @ D2_inv_sqrt @ Vt2).astype(np.float32)

        M = S11_inv_sqrt @ S12 @ S22_inv_sqrt
        U, Sigma, Vt = np.linalg.svd(M, full_matrices=False)

        U_r = U[:, :self.n_components]
        V_r = Vt[:self.n_components, :].T  # shape => (d2, n_components)

        # 先得到未归一化的投影矩阵
        W1_unnorm = S11_inv_sqrt @ U_r  # shape (d1, n_components)
        W2_unnorm = S22_inv_sqrt @ V_r  # shape (d2, n_components)

        # ========== 修正：CCA归一化 (使每个canonical weight在各自S11, S22度量下范数为 1) ==========
        W1 = np.zeros_like(W1_unnorm)
        W2 = np.zeros_like(W2_unnorm)
        for i in range(self.n_components):
            w1_i = W1_unnorm[:, i]
            w2_i = W2_unnorm[:, i]
            # 计算在原协方差 S11, S22 下的范数
            norm1 = np.sqrt(w1_i.T @ S11 @ w1_i)
            norm2 = np.sqrt(w2_i.T @ S22 @ w2_i)
            # 避免数值问题
            if norm1 < 1e-15 or norm2 < 1e-15:
                raise ValueError("出现奇异解：请提高 alpha 或检查数据。")
            # 归一化
            w1_i /= norm1
            w2_i /= norm2
            W1[:, i] = w1_i
            W2[:, i] = w2_i

        self.W1 = W1
        self.W2 = W2

    def transform(self, X, view=1):
        """
        将原始特征X投影到 rCCA 学到的潜在空间
        X: shape (d, n)
        view=1 或 2 指定视图
        输出: shape (n, n_components)
        """
        if view == 1:
            if self.W1 is None:
                raise ValueError("W1 is not fitted yet. Call fit() first.")
            X_centered = X - self.mean1_
            return (self.W1.T @ X_centered).T
        else:
            if self.W2 is None:
                raise ValueError("W2 is not fitted yet. Call fit() first.")
            X_centered = X - self.mean2_
            return (self.W2.T @ X_centered).T


# ============== 第B部分：数据加载与特征提取 ==============
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
    修改后的数据加载函数，适应Br35H数据集结构。其余部分保持不变。
    """
    if not 0 < sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0,1]")

    # 为Br35H数据集定义类别映射
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


# ============== 评估函数部分==============
def train_test_split_on_samples(X1, X2, labels, test_size, random_state=42):
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        X1, X2, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    return X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te


def evaluate_rcca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, alpha_vals,
        csv_path="nn_rcca_results.csv",
        random_seed=42,
        verbose=True
):
    """
    1-NN评估rCCA投影效果，并记录Accuracy和F1指标。
    与原 evaluate_ccp_nn 类似，但超参替换为 n_components_vals & alpha_vals。
    """
    results = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "alpha", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for alpha in alpha_vals:
            if verbose:
                print(f"\n[Evaluation] 1-NN => n_components={n_comp}, alpha={alpha}")

            rcca = FeatureLevelRCCA(n_components=n_comp, alpha=alpha)
            rcca.fit(X1_train, X2_train)

            Z1_train = rcca.transform(X1_train, view=1)
            Z2_train = rcca.transform(X2_train, view=2)
            X_train_rcca = np.hstack([Z1_train, Z2_train])

            Z1_test = rcca.transform(X1_test, view=1)
            Z2_test = rcca.transform(X2_test, view=2)
            X_test_rcca = np.hstack([Z1_test, Z2_test])

            # 1-NN
            nnclf = KNeighborsClassifier(n_neighbors=1)
            nnclf.fit(X_train_rcca, y_train)
            y_pred = nnclf.predict(X_test_rcca)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'n_components': n_comp,
                'alpha': alpha,
                'accuracy': acc,
                'f1': f1
            })

            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n_comp, alpha, f"{acc:.4f}", f"{f1:.4f}"])

    # 按 F1 分数排序，显示前 5 个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[1-NN (rCCA) Top 5 Results]")
    print("Rank\tn_comp\talpha\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['alpha']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_rcca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, alpha_vals,
        csv_path="lightgbm_rcca_results.csv",
        random_seed=42,
        verbose=True
):
    """
    LightGBM评估rCCA投影效果，记录Accuracy和F1指标。
    与原 evaluate_ccp_lightgbm 类似，但超参替换为 n_components_vals & alpha_vals。
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
        writer.writerow(["n_components", "alpha", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for alpha in alpha_vals:
            if verbose:
                print(f"\n[Evaluation] LightGBM => n_components={n_comp}, alpha={alpha}")

            rcca = FeatureLevelRCCA(n_components=n_comp, alpha=alpha)
            rcca.fit(X1_train, X2_train)

            Z1_train = rcca.transform(X1_train, view=1)
            Z2_train = rcca.transform(X2_train, view=2)
            X_train_rcca = np.hstack([Z1_train, Z2_train])

            Z1_test = rcca.transform(X1_test, view=1)
            Z2_test = rcca.transform(X2_test, view=2)
            X_test_rcca = np.hstack([Z1_test, Z2_test])

            clf = LGBMClassifier(**lgb_params)
            clf.fit(X_train_rcca, y_train)
            y_pred = clf.predict(X_test_rcca)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'n_components': n_comp,
                'alpha': alpha,
                'accuracy': acc,
                'f1': f1
            })

            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n_comp, alpha, f"{acc:.4f}", f"{f1:.4f}"])

    # 按 F1 分数排序，显示前 5 个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[LightGBM (rCCA) Top 5 Results]")
    print("Rank\tn_comp\talpha\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['alpha']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


# ============== 主函数 ==============
def main():
    dataset_path = r"E:\jqxx\dataset\Br35H"  # Br35H数据集路径
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
    print("X1 (样本为行) shape:", X1_samples.shape)
    print("X2 (样本为行) shape:", X2_samples.shape)
    print("labels shape:", labels.shape)
    print("标签分布：", np.bincount(labels))

    # 1) 样本级划分
    X1_tr_samp, X1_te_samp, X2_tr_samp, X2_te_samp, y_train, y_test = train_test_split_on_samples(
        X1_samples, X2_samples, labels,
        test_size=test_ratio,
        random_state=random_seed
    )

    # 2) 转置(以特征为行)
    X1_train = X1_tr_samp.T
    X2_train = X2_tr_samp.T
    X1_test = X1_te_samp.T
    X2_test = X2_te_samp.T

    # 3) 超参数范围（给出 rCCA 的建议调参区间）
    n_components_vals = [2, 5, 8, 15, 16]
    alpha_vals = [ 1e-1, 1,5,10]

    # 4) 评估: 1-NN
    print("\n========== [Evaluation 1] rCCA + 1-NN ==========")
    evaluate_rcca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        alpha_vals=alpha_vals,
        csv_path="nn_rcca_results.csv",
        random_seed=random_seed,
        verbose=True
    )

    # 5) 评估: LightGBM
    print("\n========== [Evaluation 2] rCCA + LightGBM ==========")
    evaluate_rcca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        alpha_vals=alpha_vals,
        csv_path="lightgbm_rcca_results.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n>>> Done.")


if __name__ == "__main__":
    main()