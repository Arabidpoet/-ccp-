

### 此代码库为使用 **CCP（Cross-view Correlation Projection）方法** 对图像特征进行处理及后续分类的对比实验，以验证模型在三种图像医学数据集上的表现，并输出相应的统计评估结果（Accuracy、F1-score）。


## 1. 代码主要结构和介绍
1. **CCP方法模块（第A部分）**  
2. **数据加载与特征提取（第B部分）**  
3. **评估函数（第C部分）**  
4. **主函数（第D部分）**
   （其余方法实验代码结构也与此类似）
   
### 三个名为yx cpp1 2 3 的文件夹分别对应数据集123的实验代码和实验数据（csv格式），实验数据文件夹中名为 1 2 3的word文件也分别直观地保留了六种方法（ccp;cca;kcca;rcca;dcca;dccae)的实验数据。###

## 数据集介绍
1.SARS-COV-2 Ct-Scan数据集，SARS-COV-2 Ct-Scan 数据集包含 2482 张 CT 肺部扫描图像。此数据集分为两类。
网址：https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/SARS-COV-2.md
2.Chest CT-Scan images数据集，是人类胸部癌检测的 2D-CT 图像数据集。1,000 张 CT 图像，分为四类。
网址：https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/SARS-COV-2.md
3.Br35H数据集：3,000 张脑部 MRI 图像，分为两类：非肿瘤和肿瘤。
网址：https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/Br35H.md
三种数据集均已经上传到百度网盘，方便下载：
链接: https://pan.baidu.com/s/1UZUF6MYHifsKALbf6wFsBw?pwd=ei8g 提取码: ei8g
## 2. 使用方法

1. **数据准备**  
   - 将数据集放在合适的目录下，并在代码中修改 `dataset_path`
2. **环境配置**  
```plaintext
Python 3.8+ (建议使用 3.8 或更高版本)
numpy == 1.20.0  或以上版本
pandas == 1.2.0  或以上版本
scikit-learn == 0.24.0 或以上版本
lightgbm == 3.2.0     或以上版本
Pillow == 8.1.0       或以上版本
```

3. **快速运行**  
   - 在终端或 Python IDE 中运行该脚本：  
     ```bash
     python your_script_name.py
     ```
   - 主函数默认会执行所有步骤，包括数据加载、训练、评估，并输出结果到 `nn_results.csv` 与 `lightgbm_results.csv`。  



