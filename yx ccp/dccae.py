
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============== 第A部分：DCCAE方法模块 ==============
class DCCAE(nn.Module):
    """
    Deep Canonical Correlation Autoencoder (DCCAE) 类。
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim=512, out_dim=2, lr=1e-4):
        super(DCCAE, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        self.lr = lr

    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        return z1, z2

    def loss_func(self, z1, z2, use_all_singular_values=False, loss_type="l2"):
        r1 = 1e-4
        r2 = 1e-4

        z1 = z1 - torch.mean(z1, 0)
        z2 = z2 - torch.mean(z2, 0)

        c = torch.matmul(z1.t(), z2)

        cost = self.cca_loss(c, r1=r1, r2=r2, use_all_singular_values=use_all_singular_values, loss_type=loss_type)
        return cost

    def cca_loss(self, c, r1=1e-4, r2=1e-4, use_all_singular_values=False, loss_type="l2"):
        c11 = torch.matmul(c, c.t()) + r1 * torch.eye(c.shape[0], dtype=c.dtype, device=c.device)
        c22 = torch.matmul(c.t(), c) + r2 * torch.eye(c.shape[1], dtype=c.dtype, device=c.device)
        c_inv1 = torch.inverse(c11)
        c_inv2 = torch.inverse(c22)
        cca_corr = torch.trace(torch.matmul(torch.matmul(c_inv1, c), c_inv2.t()))
        if loss_type == "l2":
            return torch.square(cca_corr)
        elif loss_type == "l1":
            return torch.abs(cca_corr)
        return cca_corr

    def fit(self, X1, X2, epochs=100, batch_size=32, use_all_singular_values=False, loss_type="l2"):
        X1_tensor = torch.tensor(X1, dtype=torch.float32)
        X2_tensor = torch.tensor(X2, dtype=torch.float32)
        dataset = TensorDataset(X1_tensor, X2_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X1, batch_X2 in dataloader:
                optimizer.zero_grad()
                z1, z2 = self(batch_X1, batch_X2)
                loss = -self.loss_func(z1, z2, use_all_singular_values=use_all_singular_values, loss_type=loss_type)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    def transform(self, X, view=1):
        """
        将数据投影到指定视图的潜在空间。
        :param X: 输入数据 shape(n, dim)
        :param view: 1 或 2. 根据不同视图使用 self.encoder1 或 self.encoder2
        :return: shape(n, out_dim)
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            if view == 1:
                Z = self.encoder1(X_tensor)
            elif view == 2:
                Z = self.encoder2(X_tensor)
            else:
                raise ValueError("View must be 1 or 2.")
        return Z.numpy()


class FeatureLevelDCCAE:
    """
    DCCAE 类.
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim=512, n_components=2, lr=1e-4):
        """
        :param input_dim1: 第一视图的输入特征维度
        :param input_dim2: 第二视图的输入特征维度
        :param hidden_dim: 中间隐藏层的维度
        :param n_components: 降维目标维度
        :param lr: 学习率
        """
        self.dccae = DCCAE(input_dim1, input_dim2, hidden_dim=hidden_dim, out_dim=n_components, lr=lr)
        self.X_train_dims = (input_dim1, input_dim2)
        self.n_components = n_components
        self.W1 = None
        self.W2 = None

    def fit(self, X1, X2, epochs=100, batch_size=32):
        """
        训练 DCCAE 模型.
        X1, X2 格式: (n, d1), (n, d2)，即 "以样本为行".
        """
        self.dccae.fit(X1, X2, epochs=epochs, batch_size=batch_size)

    def transform(self, X, view=1):
        """
        将 (n, d) 数据投影到 (n, n_components).
        :param view: 1 or 2. 根据不同视图使用 self.dccae 的相应编码器
        :return: shape(n, n_components)
        """
        if view == 1:
            self.W1 = self.dccae.encoder1
            return self.dccae.transform(X, view=1)
        else:
            self.W2 = self.dccae.encoder2
            return self.dccae.transform(X, view=2)


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
def evaluate_dccae_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, input_dim1, input_dim2,
        csv_path="nn_results.csv",
        random_seed=42,
        verbose=True
):
    """
    1-NN评估DCCAE投影效果，并记录Accuracy和F1指标。
    """
    results = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "accuracy", "f1"])

    for n_comp in n_components_vals:
        if verbose:
            print(f"\n[Evaluation] 1-NN => n_components={n_comp}")

        dccae = FeatureLevelDCCAE(input_dim1=input_dim1, input_dim2=input_dim2, n_components=n_comp, lr=1e-4)
        dccae.fit(X1_train, X2_train, epochs=100, batch_size=32)

        Z1_train = dccae.transform(X1_train, view=1)
        Z2_train = dccae.transform(X2_train, view=2)
        X_train_dccae = np.hstack([Z1_train, Z2_train])

        Z1_test = dccae.transform(X1_test, view=1)
        Z2_test = dccae.transform(X2_test, view=2)
        X_test_dccae = np.hstack([Z1_test, Z2_test])

        # 1-NN
        nnclf = KNeighborsClassifier(n_neighbors=1)
        nnclf.fit(X_train_dccae, y_train)
        y_pred = nnclf.predict(X_test_dccae)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        results.append({
            'n_components': n_comp,
            'accuracy': acc,
            'f1': f1
        })

        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([n_comp, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[1-NN Top 5 Results]")
    print("Rank\tn_comp\tACC\tF1")
    print("-" * 30)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_dccae_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals, input_dim1, input_dim2,
        csv_path="lightgbm_results.csv",
        random_seed=42,
        verbose=True
):
    """
    LightGBM评估DCCAE投影效果，记录Accuracy和F1指标。
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
        writer.writerow(["n_components", "accuracy", "f1"])

    for n_comp in n_components_vals:
        if verbose:
            print(f"\n[Evaluation] LightGBM => n_components={n_comp}")

        dccae = FeatureLevelDCCAE(input_dim1=input_dim1, input_dim2=input_dim2, n_components=n_comp, lr=1e-4)
        dccae.fit(X1_train, X2_train, epochs=100, batch_size=32)

        Z1_train = dccae.transform(X1_train, view=1)
        Z2_train = dccae.transform(X2_train, view=2)
        X_train_dccae = np.hstack([Z1_train, Z2_train])

        Z1_test = dccae.transform(X1_test, view=1)
        Z2_test = dccae.transform(X2_test, view=2)
        X_test_dccae = np.hstack([Z1_test, Z2_test])

        clf = LGBMClassifier(**lgb_params)
        clf.fit(X_train_dccae, y_train)
        y_pred = clf.predict(X_test_dccae)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        results.append({
            'n_components': n_comp,
            'accuracy': acc,
            'f1': f1
        })

        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([n_comp, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x['f1'], reverse=True)
    print("\n[LightGBM Top 5 Results]")
    print("Rank\tn_comp\tACC\tF1")
    print("-" * 30)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}\t{res['n_components']}\t{res['accuracy']:.4f}\t{res['f1']:.4f}")

    print(f"\nResults saved to {csv_path}")


# ============== 第D部分：主函数 ==============
def main():
    dataset_path =r"D:\\archive"   # <-- 请根据实际路径修改
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
    # 这里暂时不需要转置，因为 DCCAE 的输入是以样本为行的。
    X1_train = X1_tr_samp  # shape(nTrain, d1)
    X2_train = X2_tr_samp  # shape(nTrain, d2)
    X1_test = X1_te_samp  # shape(nTest, d1)
    X2_test = X2_te_samp  # shape(nTest, d2)

    print("\n转换后:")
    print("X1_train.shape =", X1_train.shape)
    print("X2_train.shape =", X2_train.shape)
    print("X1_test.shape  =", X1_test.shape)
    print("X2_test.shape  =", X2_test.shape)

    # 3) 超参数范围
    n_components_vals = [2, 4, 8, 12]
    input_dim1 = X1_train.shape[1]  # shape=(n_samples, dim1) => dim1
    input_dim2 = X2_train.shape[1]  # shape=(n_samples, dim2) => dim2

    # 4) 评估: 1-NN
    print("\n========== [Evaluation 1] 1-NN ==========")
    evaluate_dccae_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        input_dim1=input_dim1,
        input_dim2=input_dim2,
        csv_path="nn_results_dccae.csv",
        random_seed=random_seed,
        verbose=True
    )

    # 5) 评估: LightGBM
    print("\n========== [Evaluation 2] LightGBM ==========")
    evaluate_dccae_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        input_dim1=input_dim1,
        input_dim2=input_dim2,
        csv_path="lightgbm_results_dccae.csv",
        random_seed=random_seed,
        verbose=True
    )

    print("\n>>> Done.")


if __name__ == "__main__":
    main()