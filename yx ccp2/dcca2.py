
import os
import cv2
import csv
import random
import warnings
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============== 第A部分：DCCA方法模块（独立封装，便于后续替换/对比） ==============
class FeatureLevelDCCA:
    """
    独立封装的 DCCA 类，包含:
    1) 计算协方差矩阵
    2) 根据DCCA方法求解投影矩阵
    3) transform 函数做投影变换
    """

    def __init__(self, reg_param=1e-5, n_components=2, verbose=False):
        """
        :param reg_param: 正则化参数
        :param n_components: 降维维度
        :param verbose: 是否打印 debug 信息
        """
        self.reg_param = reg_param
        self.n_components = n_components
        self.verbose = verbose
        self.W1 = None
        self.W2 = None

    def fit(self, X1, X2):
        """
        主流程: 根据DCCA方法构造 W1, W2
        X1.shape=(n, d1), X2.shape=(n, d2)
        """
        if self.verbose:
            print(f"[DCCA Fit] Start. reg_param={self.reg_param}, n_components={self.n_components}")

        n = X1.shape[0]

        # 中心化数据
        X1_centered = X1 - X1.mean(axis=0)
        X2_centered = X2 - X2.mean(axis=0)

        # 计算协方差矩阵
        C11 = (X1_centered.T @ X1_centered) / (n - 1)
        C22 = (X2_centered.T @ X2_centered) / (n - 1)
        C12 = (X1_centered.T @ X2_centered) / (n - 1)

        # 添加正则化项
        C11 += self.reg_param * np.eye(C11.shape[0])
        C22 += self.reg_param * np.eye(C22.shape[0])

        # 计算矩阵的逆平方根
        def inv_sqrt(mat):
            eigvals, eigvecs = np.linalg.eigh(mat)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        C11_inv_sqrt = inv_sqrt(C11)
        C22_inv_sqrt = inv_sqrt(C22)

        # 计算T矩阵并进行SVD分解
        T = C11_inv_sqrt @ C12 @ C22_inv_sqrt
        U, S, Vt = np.linalg.svd(T, full_matrices=False)

        # 取前k个奇异向量
        U_k = U[:, :self.n_components]
        V_k = Vt[:self.n_components, :].T

        # 计算投影矩阵
        self.W1 = C11_inv_sqrt @ U_k
        self.W2 = C22_inv_sqrt @ V_k

        if self.verbose:
            print(f"  [Fit done] Top singular values={S[:self.n_components]}")

    def transform(self, X, which='view1'):
        """
        将 X(n,d) 投影到 (n, n_components)
        which='view1' or 'view2' 决定用哪个投影矩阵
        """
        if which == 'view1':
            if self.W1 is None:
                raise ValueError("W1 not fitted.")
            return X @ self.W1
        else:
            if self.W2 is None:
                raise ValueError("W2 not fitted.")
            return X @ self.W2


# ============== 第B部分：数据加载与特征提取 ==============
def extract_features(img):
    """
    图像特征提取:
     - RGB直方图(3*32=96维)
     - LBP纹理(radius=3, uniform)
    返回 feat1(96维), feat2(若 radius=3 则 bins ~ 26左右) 视数据而定
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
    print("[Val  ] X1:", X1_val.shape,   "X2:", X2_val.shape,   "y:", y_val.shape)
    print("[Test ] X1:", X1_test.shape,  "X2:", X2_test.shape,  "y:", y_test.shape)

    return (X1_train, X2_train, y_train,
            X1_val, X2_val, y_val,
            X1_test, X2_test, y_test)


# ============== 第C部分：评估函数 (1NN & LightGBM) ==============
def evaluate_dcca_nn(X1_train, X2_train, y_train,
                    X1_test,  X2_test,  y_test,
                    n_components_vals, reg_params,
                    csv_path="nn_results.csv",
                    random_seed=42, verbose=True):
    """
    1-NN评估DCCA投影效果。
    结果存入CSV文件(csv_path)，包含:
      n_components, reg_param, accuracy, f1_score
    """
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "reg_param", "accuracy", "f1"])

    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc  = scaler1.transform(X1_test)
    X2_test_sc  = scaler2.transform(X2_test)

    results = []

    for nc in n_components_vals:
        for reg in reg_params:
            if verbose:
                print(f"\n[Evaluation] 1-NN => n_components={nc}, reg_param={reg}")

            # 使用DCCA
            dcca = FeatureLevelDCCA(reg_param=reg, n_components=nc, verbose=False)
            dcca.fit(X1_train_sc, X2_train_sc)

            # 投影处理
            Z1_train = dcca.transform(X1_train_sc, 'view1')
            Z2_train = dcca.transform(X2_train_sc, 'view2')
            Z_train = np.hstack([Z1_train, Z2_train])

            Z1_test = dcca.transform(X1_test_sc, 'view1')
            Z2_test = dcca.transform(X2_test_sc, 'view2')
            Z_test = np.hstack([Z1_test, Z2_test])

            # 1-NN
            nn = KNeighborsClassifier(n_neighbors=1)
            nn.fit(Z_train, y_train)
            pred_nn = nn.predict(Z_test)
            acc = accuracy_score(y_test, pred_nn)
            f1 = f1_score(y_test, pred_nn, average='macro')

            results.append((nc, reg, acc, f1))

            # 写入CSV
            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nc, reg, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x[3], reverse=True)
    print("\n[1-NN Top 5 Results]")
    print("Rank\tn_comp\treg_param\tACC\tF1")
    print("-" * 40)
    for i, (nc, reg, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{reg}\t{acc:.4f}\t{f1:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_dcca_lightgbm(X1_train, X2_train, y_train,
                          X1_test,  X2_test,  y_test,
                          n_components_vals, reg_params,
                          csv_path="lightgbm_results.csv",
                          random_seed=42, verbose=True):
    """
    LightGBM评估DCCA投影效果。
    结果存入CSV文件(csv_path)，包含:
      n_components, reg_param, accuracy, f1
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
        'force_col_wise': True,
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss'
    }

    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "reg_param", "accuracy", "f1"])

    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc  = scaler1.transform(X1_test)
    X2_test_sc  = scaler2.transform(X2_test)

    results = []

    for nc in n_components_vals:
        for reg in reg_params:
            if verbose:
                print(f"\n[Evaluation] LightGBM => n_components={nc}, reg_param={reg}")

            # 使用DCCA
            dcca = FeatureLevelDCCA(reg_param=reg, n_components=nc, verbose=False)
            dcca.fit(X1_train_sc, X2_train_sc)

            Z1_train = dcca.transform(X1_train_sc, which='view1')
            Z2_train = dcca.transform(X2_train_sc, which='view2')
            Z_train = np.hstack([Z1_train, Z2_train])

            Z1_test = dcca.transform(X1_test_sc, which='view1')
            Z2_test = dcca.transform(X2_test_sc, which='view2')
            Z_test = np.hstack([Z1_test, Z2_test])

            clf = LGBMClassifier(**lgb_params)
            clf.fit(Z_train, y_train)
            y_pred = clf.predict(Z_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append((nc, reg, acc, f1))

            # 写入CSV
            with open(csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nc, reg, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x[3], reverse=True)
    print("\n[LightGBM Top 5 Results]")
    print("Rank\tn_comp\treg_param\tACC\tF1")
    print("-" * 40)
    for i, (nc, reg, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{reg}\t{acc:.4f}\t{f1:.4f}")

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
    reg_params = [1e-6, 1e-5, 1e-4, 1e-3]

    # 1) 评估: 1-NN
    print("\n========== [Evaluation 1] 1-NN ==========")
    evaluate_dcca_nn(
        X1_train, X2_train, y_train,
        X1_test,  X2_test,  y_test,
        n_components_vals=n_components_vals,
        reg_params=reg_params,
        csv_path="nn_results.csv",
        random_seed=42,
        verbose=True
    )

    # 2) 评估: LightGBM
    print("\n========== [Evaluation 2] LightGBM ==========")
    evaluate_dcca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test,  X2_test,  y_test,
        n_components_vals=n_components_vals,
        reg_params=reg_params,
        csv_path="lightgbm_results.csv",
        random_seed=42,
        verbose=True
    )


if __name__ == "__main__":
    main()