# -*- coding: utf-8 -*-

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
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============== 第A部分：DCCAE方法模块 ==============
class FeatureLevelDCCAE:
    """
    独立封装的 DCCAE 类，包含:
    1) 构建自编码器
    2) 训练自编码器
    3) transform 方法做特征提取
    """

    def __init__(self, n_components=2, regularizer=1e-4, learning_rate=0.001, epochs=100, batch_size=32, verbose=True):
        """
        :param n_components: 降维维度
        :param regularizer: 正则化系数
        :param learning_rate: 学习率
        :param epochs: 训练迭代次数
        :param batch_size: 批次大小
        :param verbose: 是否打印 debug 信息
        """
        self.n_components = n_components
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.autoencoder, self.encoder1, self.encoder2 = self._build_autoencoder()

    def _build_autoencoder(self):
        """
        构建自编码器模型以及编码器部分
        """
        input_dim1 = 96  # 假设feat1的维度是96
        input_dim2 = 26  # 假设feat2的维度是26

        # 编码器1
        input1 = Input(shape=(input_dim1,))
        x1 = Dense(64, activation='relu', kernel_regularizer=l2(self.regularizer))(input1)
        x1 = Dense(32, activation='relu', kernel_regularizer=l2(self.regularizer))(x1)
        latent1 = Dense(self.n_components, activation='tanh', kernel_regularizer=l2(self.regularizer))(x1)

        # 解码器1
        x1 = Dense(32, activation='relu', kernel_regularizer=l2(self.regularizer))(latent1)
        x1 = Dense(64, activation='relu', kernel_regularizer=l2(self.regularizer))(x1)
        decoded1 = Dense(input_dim1, activation=None, kernel_regularizer=l2(self.regularizer))(x1)

        # 编码器2
        input2 = Input(shape=(input_dim2,))
        x2 = Dense(64, activation='relu', kernel_regularizer=l2(self.regularizer))(input2)
        x2 = Dense(32, activation='relu', kernel_regularizer=l2(self.regularizer))(x2)
        latent2 = Dense(self.n_components, activation='tanh', kernel_regularizer=l2(self.regularizer))(x2)

        # 解码器2
        x2 = Dense(32, activation='relu', kernel_regularizer=l2(self.regularizer))(latent2)
        x2 = Dense(64, activation='relu', kernel_regularizer=l2(self.regularizer))(x2)
        decoded2 = Dense(input_dim2, activation=None, kernel_regularizer=l2(self.regularizer))(x2)

        # 自编码器的两个部分
        autoencoder1 = Model(input1, decoded1)
        autoencoder2 = Model(input2, decoded2)

        # 编码器部分
        encoder1 = Model(input1, latent1)
        encoder2 = Model(input2, latent2)

        # 合并编码器
        combined_input = Input(shape=(input_dim1 + input_dim2,))
        x = Dense(32, activation='relu')(combined_input)
        x = Dropout(0.05)(x)
        x = Dense(self.n_components, activation='tanh')(x)
        x = Dropout(0.05)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.05)(x)
        combined_latent = Dense(self.n_components, activation='tanh')(x)

        self.encoder_combined = Model(combined_input, combined_latent)

        # 编译
        autoencoder1.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae')
        autoencoder2.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae')
        autoencoder = Model(inputs=[input1, input2], outputs=[decoded1, decoded2])
        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae')

        return autoencoder, encoder1, encoder2

    def fit(self, X1, X2):
        """
        主流程: 训练自编码器
        X1.shape=(n, d1), X2.shape=(n, d2)
        """
        if self.verbose:
            print(f"[DCCAE Fit] Start. n_components={self.n_components}")

        self.autoencoder.fit([X1, X2], [X1, X2],
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             validation_split=0.1,
                             verbose=1 if self.verbose else 0)

        if self.verbose:
            print("[DCCAE Fit] Done.")

    def transform(self, X1, X2):
        """
        将 X1(n,d1), X2(n,d2) 输入到组合编码器，提取到 (n, n_components).
        """
        # 合并特征
        X_combined = np.hstack([X1, X2])
        return self.encoder_combined.predict(X_combined)


# ============== 第B部分：数据加载与特征提取 ==============
def extract_features(img):
    """
    图像特征提取:
     - RGB直方图(3*32=96维)
     - LBP纹理(radius=3, uniform)
    返回 feat1(96维), feat2(26维) 视数据而定
    """
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
    """
    root_dir/ train/ [4类文件夹], val/[4类], test/[4类]
    类名与您给出的代码相同.
    """
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
    """
    返回 (X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test)
    """
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
def evaluate_dccae_nn(X1_train, X2_train, y_train,
                      X1_test, X2_test, y_test,
                      n_components_vals,
                      csv_path="nn_results.csv",
                      random_seed=42, verbose=True):
    """
    1-NN评估DCCAE提取效果。
    结果存入CSV文件(csv_path)，包含:
      n_components, accuracy, f1_score
    """
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "accuracy", "f1"])

    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc = scaler1.transform(X1_test)
    X2_test_sc = scaler2.transform(X2_test)

    results = []

    for nc in n_components_vals:
        if verbose:
            print(f"\n[Evaluation] 1-NN => n_components={nc}")

        # 构造DCCAE
        dccae = FeatureLevelDCCAE(n_components=nc, verbose=verbose)
        dccae.fit(X1_train_sc, X2_train_sc)

        Z_train = dccae.transform(X1_train_sc, X2_train_sc)
        Z_test = dccae.transform(X1_test_sc, X2_test_sc)

        # 1-NN
        nn = KNeighborsClassifier(n_neighbors=1)
        nn.fit(Z_train, y_train)
        pred_nn = nn.predict(Z_test)
        acc = accuracy_score(y_test, pred_nn)
        f1 = f1_score(y_test, pred_nn, average='macro')

        results.append((nc, acc, f1))

        # 写入CSV
        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([nc, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x[2], reverse=True)
    print("\n[1-NN Top 5 Results]")
    print("Rank\tn_comp\tACC\tF1")
    print("-" * 40)
    for i, (nc, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{acc:.4f}\t{f1:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_dccae_lightgbm(X1_train, X2_train, y_train,
                            X1_test, X2_test, y_test,
                            n_components_vals,
                            csv_path="lightgbm_results.csv",
                            random_seed=42, verbose=True):
    """
    LightGBM评估DCCAE提取效果。
    结果存入CSV文件(csv_path)，包含:
      n_components, accuracy, f1_score
    """
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
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss'
    }

    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "accuracy", "f1"])

    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc = scaler1.transform(X1_test)
    X2_test_sc = scaler2.transform(X2_test)

    results = []

    for nc in n_components_vals:
        if verbose:
            print(f"\n[Evaluation] LightGBM => n_components={nc}")

        # 构造DCCAE
        dccae = FeatureLevelDCCAE(n_components=nc, verbose=verbose)
        dccae.fit(X1_train_sc, X2_train_sc)

        Z_train = dccae.transform(X1_train_sc, X2_train_sc)
        Z_test = dccae.transform(X1_test_sc, X2_test_sc)

        clf = LGBMClassifier(**lgb_params)
        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        results.append((nc, acc, f1))

        # 写入CSV
        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([nc, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x[2], reverse=True)
    print("\n[LightGBM Top 5 Results]")
    print("Rank\tn_comp\tACC\tF1")
    print("-" * 40)
    for i, (nc, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{acc:.4f}\t{f1:.4f}")

    print(f"\nResults saved to {csv_path}")


# ============== 第D部分：主函数 ==============
def main():
    data_dir = r"D:\Chest CT-Scan images Dataset"

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

    # 参数范围
    n_components_vals = [2, 5, 10, 20, 25, 30]

    # 1) 评估: 1-NN
    print("\n========== [Evaluation 1] 1-NN ==========")
    evaluate_dccae_nn(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        csv_path="nn_results.csv",
        random_seed=42,
        verbose=True
    )

    # 2) 评估: LightGBM
    print("\n========== [Evaluation 2] LightGBM ==========")
    evaluate_dccae_lightgbm(
        X1_train, X2_train, y_train,
        X1_test, X2_test, y_test,
        n_components_vals=n_components_vals,
        csv_path="lightgbm_results.csv",
        random_seed=42,
        verbose=True
    )


if __name__ == "__main__":
    main()