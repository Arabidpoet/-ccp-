

# 此代码库为使用 **CCP（Cross-view Correlation Projection）方法** 对图像特征进行处理及后续分类的实验，以验证模型在三种图像医学数据集上的表现，并输出相应的统计评估结果（Accuracy、F1-score）。


## 1. 主要结构

1. **CCP方法模块（第A部分）**  
2. **数据加载与特征提取（第B部分）**  
3. **评估函数（第C部分）**  
4. **主函数（第D部分）**  


## 2. 使用方法

1. **数据准备**  
   - 将数据集放在合适的目录下，并在代码中修改 `dataset_path`
2. **环境配置**  
 Python 3.8+ (建议使用 3.8 或更高版本)
 numpy == 1.20.0  或以上版本
 pandas == 1.2.0  或以上版本
 scikit-learn == 0.24.0 或以上版本
 lightgbm == 3.2.0     或以上版本
 Pillow == 8.1.0       或以上版本
3. **快速运行**  
   - 在终端或 Python IDE 中运行该脚本：  
     ```bash
     python your_script_name.py
     ```
   - 主函数默认会执行所有步骤，包括数据加载、训练、评估，并输出结果到 `nn_results.csv` 与 `lightgbm_results.csv`。  



