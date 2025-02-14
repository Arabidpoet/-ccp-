
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

# 新增的 CCA 所需包
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============== 第A部分：FeatureLevelCCA方法模块 ==============
class FeatureLevelCCA:


    def __init__(self, n_components=2, center=True, max_iter=5000, tol=1e-6):
        """
        :param n_components: CCA投影维度
        :param center: 是否在 CCA 前进行去中心化（在这里体现为归一化+去均值）
        :param max_iter: CCA求解最大迭代次数（sklearn默认500，如需更严格可适当提高）
        :param tol: CCA迭代收敛阈值
        """
        self.n_components = n_components
        self.center = center
        self.max_iter = max_iter
        self.tol = tol

        # CCA的内部模型与各自的Scaler
        self.cca = None
        self.scaler_x = None
        self.scaler_y = None

    def fit(self, X1, X2):
        """
        拟合CCA模型。输入X1, X2形状为 (n_features, n_samples)。
        内部会转置为 (n_samples, n_features) 并进行标准化，然后进行CCA拟合。
        """
        # 转置 => (n_samples, n_features)
        X1 = X1.T
        X2 = X2.T

        # 是否对两个视图分别做StandardScaler
        if self.center:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
            X1 = self.scaler_x.fit_transform(X1)
            X2 = self.scaler_y.fit_transform(X2)

        # 初始化并拟合CCA
        self.cca = CCA(
            n_components=self.n_components,
            scale=False,  # 我们已经手动做了标准化
            max_iter=self.max_iter,
            tol=self.tol
        )
        self.cca.fit(X1, X2)

    def transform(self, X, view=1):

        if self.cca is None:
            raise ValueError("CCA模型尚未fit，请先调用fit()。")

        # 转置 => (n_samples, n_features)
        X = X.T

        # 分别使用视图对应的Scaler
        if self.center:
            if view == 1 and self.scaler_x is not None:
                X = self.scaler_x.transform(X)
            elif view == 2 and self.scaler_y is not None:
                X = self.scaler_y.transform(X)

        if view == 1:
            # 投影到第一视图的协方差空间
            return X @ self.cca.x_weights_
        else:
            # 投影到第二视图的协方差空间
            return X @ self.cca.y_weights_


# ============== 第B部分：数据加载与特征提取（保持不变） ==============
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


# ============== 评估函数部分 ==============
def train_test_split_on_samples(X1, X2, labels, test_size, random_state=42):
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        X1, X2, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    return X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te


def evaluate_cca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, sigma_vals,
        csv_path="nn_results_cca.csv",
        random_seed=42,
        verbose=True
):
    """
    1-NN评估CCA投影效果，并记录Accuracy和F1指标。
    这里保留了sigma_vals是为了与原CCP评估保持形式一致，实际CCA并不需要sigma。
    """
    results = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "sigma", "accuracy", "f1"])

    for n_comp in n_components_vals:
        for s in sigma_vals:  # sigma在CCA中不会实际用到，这里仅保持占位
            if verbose:
                print(f"\n[Evaluation] 1-NN (CCA) => n_components={n_comp}, (ignore sigma={s})")

            # 初始化 CCA。如果 n_comp 超过了 X1 或 X2 的特征数最小值，就会报错
            # 因此需要确保 n_comp <= min(X1.shape[0], X2.shape[0])
            cca = FeatureLevelCCA(n_components=n_comp, center=True, max_iter=2000, tol=1e-5)
            try:
                cca.fit(X1_train, X2_train)
            except ValueError as err:
                print(f"CCA fit 出错: {err}")
                # 若报错，则跳过本轮
                continue

            Z1_train = cca.transform(X1_train, view=1)
            Z2_train = cca.transform(X2_train, view=2)
            X_train_cca = np.hstack([Z1_train, Z2_train])

            Z1_test = cca.transform(X1_test, view=1)
            Z2_test = cca.transform(X2_test, view=2)
            X_test_cca = np.hstack([Z1_test, Z2_test])

            # 1-NN
            nnclf = KNeighborsClassifier(n_neighbors=1)
            nnclf.fit(X_train_cca, y_train)
            y_pred = nnclf.predict(X_test_cca)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'n_components': n_comp,
                'sigma': s,
                'accuracy': acc,
                'f1': f1
            })

            with open(csv_path, "a", newline='') as f_csv:
                writer_csv = csv.writer(f_csv)
                writer_csv.writerow([n_comp, s, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[1-NN (CCA) Top 5 Results]")
    print("Rank\tn_comp\tsigma\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['sigma']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_cca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, sigma_vals,
        csv_path="lightgbm_results_cca.csv",
        random_seed=42,
        verbose=True
):
    """
    LightGBM评估CCA投影效果，记录Accuracy和F1指标。
    同样地，这里保留sigma_vals仅是为了与原代码保持一致，占位
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
                print(f"\n[Evaluation] LightGBM (CCA) => n_components={n_comp}, (ignore sigma={s})")

            cca = FeatureLevelCCA(n_components=n_comp, center=True, max_iter=2000, tol=1e-5)
            try:
                cca.fit(X1_train, X2_train)
            except ValueError as err:
                print(f"CCA fit 出错: {err}")
                # 若出错，跳过并继续
                continue

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
                'sigma': s,
                'accuracy': acc,
                'f1': f1
            })

            with open(csv_path, "a", newline='') as f_csv:
                writer_csv = csv.writer(f_csv)
                writer_csv.writerow([n_comp, s, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[LightGBM (CCA) Top 5 Results]")
    print("Rank\tn_comp\tsigma\tACC\tF1")
    print("-" * 40)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['sigma']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

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

    # 2) 转置(以特征为行)，与原CCP流程保持一致
    X1_train = X1_tr_samp.T
    X2_train = X2_tr_samp.T
    X1_test = X1_te_samp.T
    X2_test = X2_te_samp.T

    # 3) 超参数范围（n_components不得超过 min(X1_feature维度, X2_feature维度)）

    n_components_vals = [2, 4, 8, 12, 15]  # 不能超过
    sigma_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]  # 占位，保留接口

    # 4) 评估: 1-NN (使用CCA)
    print("\n========== [Evaluation 1] 1-NN (CCA) ==========")
    evaluate_cca_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="nn_results_cca.csv",
        random_seed=random_seed,
        verbose=True
    )

    # 5) 评估: LightGBM (使用CCA)
    print("\n========== [Evaluation 2] LightGBM (CCA) ==========")
    evaluate_cca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        sigma_vals=sigma_vals,
        csv_path="lightgbm_results_cca.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n>>> Done.")


if __name__ == "__main__":
    main()