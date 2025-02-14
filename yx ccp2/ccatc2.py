
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
from sklearn.cross_decomposition import CCA  # 关键：使用CCA

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============== 第A部分：CCA方法模块==============
class FeatureLevelCCA:
    """
    CCA 类，包含:
    1) 使用 sklearn.cross_decomposition.CCA 做特征级的多视图关联分析
    2) fit: 拟合并找到投影方向
    3) transform: 将X投影到相应的低维空间
    """
    def __init__(self, n_components=2, scale=True, max_iter=500, verbose=False):
        """
        :param n_components: 降维维度
        :param scale: 是否让 CCA 对数据执行标准化（内置）
        :param max_iter: CCA 的最大迭代次数
        :param verbose: 是否打印 debug 信息
        """
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.verbose = verbose
        self.cca_model = None

    def fit(self, X1, X2):
        """
        使用 CCA 拟合 X1, X2
        X1.shape=(n, d1), X2.shape=(n, d2)
        """
        if self.verbose:
            print(f"[CCA Fit] Start. n_components={self.n_components}, scale={self.scale}, max_iter={self.max_iter}")
        # 使用sklearn的CCA
        self.cca_model = CCA(n_components=self.n_components, scale=self.scale, max_iter=self.max_iter)
        self.cca_model.fit(X1, X2)
        return self

    def transform(self, X1, X2):
        """
        将 X1, X2 投影到 (n, n_components).
        返回 Z1, Z2
        """
        if self.cca_model is None:
            raise ValueError("请先调用 fit() 拟合后再 transform()。")
        Z1, Z2 = self.cca_model.transform(X1, X2)
        return Z1, Z2


# ============== 第B部分：数据加载与特征提取(与原CCP实验保持一致) ==============
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
    与原CCP实验相同的方式读取数据。
    root_dir/ train/ [4类文件夹], val/[4类], test/[4类]
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


# ============== 第C部分：评估函数 (1NN & LightGBM)==============
def evaluate_cca_nn(X1_train, X2_train, y_train,
                    X1_test,  X2_test,  y_test,
                    n_components_vals,
                    csv_path="nn_results_cca.csv",
                    random_seed=42, verbose=True):
    """
    1-NN评估 CCA 投影效果。
    结果存入CSV文件(csv_path)，包含:
      n_components, accuracy, f1_score
    （此时CCA只有n_components一个关键超参数）
    """
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "accuracy", "f1"])

    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc  = scaler1.transform(X1_test)
    X2_test_sc  = scaler2.transform(X2_test)

    results = []

    for nc in n_components_vals:
        if verbose:
            print(f"\n[Evaluation] 1-NN => n_components={nc}")

        # 构造CCA
        cca = FeatureLevelCCA(n_components=nc, scale=False, max_iter=1000, verbose=False)
        cca.fit(X1_train_sc, X2_train_sc)

        Z1_train, Z2_train = cca.transform(X1_train_sc, X2_train_sc)
        Z_train = np.hstack([Z1_train, Z2_train])

        Z1_test, Z2_test = cca.transform(X1_test_sc, X2_test_sc)
        Z_test = np.hstack([Z1_test, Z2_test])

        # 1-NN
        nn = KNeighborsClassifier(n_neighbors=1)
        nn.fit(Z_train, y_train)
        pred_nn = nn.predict(Z_test)
        acc = accuracy_score(y_test, pred_nn)
        f1 = f1_score(y_test, pred_nn, average='macro')

        results.append((nc, acc, f1))

        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([nc, f"{acc:.4f}", f"{f1:.4f}"])

    # 按F1分数排序，显示前5个最佳结果
    results.sort(key=lambda x: x[2], reverse=True)
    print("\n[1-NN + CCA Top 5 Results]")
    print("Rank\tn_comp\tACC\tF1")
    print("-" * 40)
    for i, (nc, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{acc:.4f}\t{f1:.4f}")

    print(f"\nResults saved to {csv_path}")


def evaluate_cca_lightgbm(X1_train, X2_train, y_train,
                          X1_test,  X2_test,  y_test,
                          n_components_vals,
                          csv_path="lightgbm_results_cca.csv",
                          random_seed=42, verbose=True):
    """
    LightGBM评估 CCA 投影效果。
    结果存入CSV文件(csv_path)，包含:
      n_components, accuracy, f1
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
        'num_class': None,  # 在fit时自动设置
        'metric': 'multi_logloss'
    }

    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "accuracy", "f1"])

    scaler1 = StandardScaler().fit(X1_train)
    scaler2 = StandardScaler().fit(X2_train)
    X1_train_sc = scaler1.transform(X1_train)
    X2_train_sc = scaler2.transform(X2_train)
    X1_test_sc  = scaler1.transform(X1_test)
    X2_test_sc  = scaler2.transform(X2_test)

    results = []

    for nc in n_components_vals:
        if verbose:
            print(f"\n[Evaluation] LightGBM => n_components={nc}")

        # 构造CCA
        cca = FeatureLevelCCA(n_components=nc, scale=False, max_iter=1000, verbose=False)
        cca.fit(X1_train_sc, X2_train_sc)

        Z1_train, Z2_train = cca.transform(X1_train_sc, X2_train_sc)
        Z_train = np.hstack([Z1_train, Z2_train])

        Z1_test, Z2_test = cca.transform(X1_test_sc, X2_test_sc)
        Z_test = np.hstack([Z1_test, Z2_test])

        # LightGBM
        num_classes = len(np.unique(y_train))
        lgb_params['num_class'] = num_classes
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
    print("\n[LightGBM + CCA Top 5 Results]")
    print("Rank\tn_comp\tACC\tF1")
    print("-" * 40)
    for i, (nc, acc, f1) in enumerate(results[:5], 1):
        print(f"{i}\t{nc}\t{acc:.4f}\t{f1:.4f}")

    print(f"\nResults saved to {csv_path}")


# ============== 第D部分：主函数（仅CCA方式） ==============
def main():
    data_dir = r"E:\jqxx\Chest CT-Scan images Dataset"

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

    # 获取 X1 和 X2 的最小特征维度
    max_components = min(X1_train.shape[1], X2_train.shape[1])

    # 设置 n_components 值（不超过 max_components）
    n_components_vals = [1, 2, 5, 10, 20, 25, max_components]

    # 1) 评估: 1-NN
    print("\n========== [Evaluation 1] 1-NN + CCA ==========")
    evaluate_cca_nn(
        X1_train, X2_train, y_train,
        X1_test,  X2_test,  y_test,
        n_components_vals=n_components_vals,
        csv_path="nn_results_cca.csv",
        random_seed=42,
        verbose=True
    )

    # 2) 评估: LightGBM
    print("\n========== [Evaluation 2] LightGBM + CCA ==========")
    evaluate_cca_lightgbm(
        X1_train, X2_train, y_train,
        X1_test,  X2_test,  y_test,
        n_components_vals=n_components_vals,
        csv_path="lightgbm_results_cca.csv",
        random_seed=42,
        verbose=True
    )


if __name__ == "__main__":
    main()