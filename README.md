# 水声信号机器学习实战 (Hydroacoustic ML Lab)

基于 FFT 频谱特征的水下声信号分类与聚类，使用 scikit-learn 完整管线。

## 数据集

Mendeley 水声数据集，CSV 格式：

| 项目 | 详情 |
|------|------|
| 样本数 | 54,000 |
| 特征数 | 256（`fft_bin_0` ~ `fft_bin_255`，FFT 频谱幅值，已归一化到 0~1） |
| 类别数 | 7 |
| 类别 | Whale (17000), Dolphin_Porpoise (10000), Ship (8000), Human_Activity (8000), Seal_Walrus_Manatee (5000), Natural_Sound (4000), Marine_Life_Sound (2000) |

## 项目结构

```
hydroacoustic-ml-lab/
├── data/
│   ├── dataset.csv          # 数据集（未提交 git）
│   └── README.md            # 数据集说明
├── src/
│   ├── data_loader.py       # 数据加载
│   └── utils.py             # 评估指标、可视化工具
├── experiments/
│   ├── 01_knn.py            # KNN 分类
│   ├── 02_randomforest.py   # 随机森林分类
│   └── 03_K-means.py        # K-means 聚类
├── results/                 # 实验结果图表
│   ├── 01_knn/
│   ├── 02_randomforest/
│   └── 03_K-means/
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行实验
python experiments/01_knn.py
python experiments/02_randomforest.py
python experiments/03_K-means.py
```

## 实验设计

每个实验对比三种预处理方案：

| 方案 | 说明 |
|------|------|
| 原始数据 | 不做任何变换 |
| 标准化 | `StandardScaler` 零均值单位方差 |
| PCA 降维 | 保留 80% 方差后标准化 |

### 实验 1：KNN 分类

- 模型：`KNeighborsClassifier`
- 调参：`GridSearchCV` 搜索最优 k 值
- 评估：准确率 + 混淆矩阵 + 多分类 ROC 曲线

### 实验 2：随机森林分类

- 模型：`RandomForestClassifier`
- 调参：`GridSearchCV` 搜索 `n_estimators`、`max_depth`、`min_samples_leaf`
- 评估：准确率 + 混淆矩阵 + ROC + 特征重要性

### 实验 3：K-means 聚类

- **实验 A**：肘部法则 + 轮廓系数确定最佳 k 值
- **实验 B**：PCA 降维到 2 维，聚类结果 vs 真实标签可视化
- 评估：轮廓系数、ARI、NMI

## 评估指标说明

| 指标 | 用途 | 范围 |
|------|------|------|
| Accuracy | 分类整体准确率 | 0~1 |
| Precision / Recall / F1 | 各类别分类表现 | 0~1 |
| ROC AUC | 分类器区分能力 | 0~1 |
| Silhouette Score | 聚类紧密度与分离度 | -1~1 |
| ARI | 聚类与真实标签一致性 | -1~1 |
| NMI | 聚类与真实标签互信息 | 0~1 |

## 技术栈

- Python 3.x
- scikit-learn（模型训练、评估）
- pandas（数据处理）
- matplotlib + seaborn（可视化）
- numpy

## 待扩展

- [ ] SVM 分类实验
- [ ] 类别不平衡处理（SMOTE、class_weight）
- [ ] 原始音频波形特征提取（替代预提取的 FFT）
