''' 本实验使用K-means算法对水声信号数据集进行聚类分析，评估聚类效果，并可视化结果。  
实验 A：用肘部法则确定最佳 k 值，并比较不同预处理方法（原始数据、标准化数据、PCA降维数据）对聚类效果的影响。
！！标准化对计算聚类欧氏距离很重要，PCA降维后可能会丢失一些信息，但也可能有助于去除噪声和冗余特征，从而提高聚类效果。
实验 B：PCA 降维到 2 维，可视化聚类结果
'''

import sys
import os
# 把项目根目录加入 Python 搜索路径，这样才能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data  # 导入数据加载函数
from src.utils import set_chinese_font  # 导入字体设置函数
from sklearn.decomposition import PCA  # 主成分分析（PCA）用于降维
from sklearn.preprocessing import StandardScaler  # 用于数据标准化
from sklearn.cluster import KMeans  # K-means聚类算法
from sklearn.metrics import  silhouette_score, adjusted_rand_score, normalized_mutual_info_score  # 用于评估聚类效果
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告

def find_best_k(x, name, k_range=range(6, 10)):
    print(f"--- {name} ---")
    loss = []
    silscore = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=50)  # 固定random_state保证可复现
        model.fit(x)
        loss.append(model.inertia_)
        sample_idx = np.random.choice(len(x), size=5000, replace=False)  # 采样计算，否则复杂度太高
        x_np = np.asarray(x)  # 统一转成 numpy，DataFrame 也兼容
        silscore.append(silhouette_score(x_np[sample_idx], model.labels_[sample_idx]))

    # 可视化肘部图
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # 左：肘部法则
    ax1.plot(k_range, loss, 'b-o', label='Inertia (肘部法则)')
    ax1.set_xlabel('聚类数K')
    ax1.set_ylabel('惯性值(Inertia)', color='b')
    # 右：轮廓系数
    ax2 = ax1.twinx()
    ax2.plot(k_range, silscore, 'r-s', label='Silhouette Score')
    ax2.set_ylabel('轮廓系数(Silhouette Score)', color='r')
    plt.title(f'肘部法则选择最优k - {name}')
    plt.grid(alpha=0.5)
    plt.show()

    best_k = k_range[np.argmax(silscore)]
    print(f'最佳K值为：{best_k}')
    return best_k


def evaluate_kmeans(x, y, best_k):
    kmodel = KMeans(n_clusters=best_k, random_state=42, n_init=50)  # 设置n_init=50以提高稳定性
    kmodel.fit(x) 
    cluster_labels = kmodel.labels_  # 获取聚类标签
    #评估聚类效果
    silhouette = silhouette_score(x, cluster_labels)  # 计算轮廓系数，衡量聚类的紧密度和分离度，值越接近1表示聚类效果越好
    print(f"轮廓系数: {silhouette:.4f}")
    print(f"ARI: {adjusted_rand_score(y, cluster_labels):.4f}")# 计算调整兰德指数（ARI），衡量聚类结果与真实标签的一致性，值越接近1表示聚类效果越好
    print(f"NMI: {normalized_mutual_info_score(y, cluster_labels):.4f}")# 计算归一化互信息（NMI），衡量聚类结果与真实标签之间的互信息，值越接近1表示聚类效果越好
    return kmodel

def main():
    set_chinese_font()
    
    # ========== 实验 A==========
    # 加载数据
    x, y, _, _, _, _, feature_names = load_data('data/dataset.csv')
    
    # 标准化数据
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # PCA降维+标准化
    pca = PCA(n_components=0.8)  # 保留80%的方差
    x_pca = pca.fit_transform(x)
    print(f"PCA降维后特征维度: {x_pca.shape[1]}")  # 输出降维后的特征维度
    x_pca_scaled = scaler.fit_transform(x_pca)

   #模型测试
    results = {}
    print("\n" + "=" * 50)
    print("K-Means不同预处理方案对比")
    print("=" * 50)

    best_k1 = find_best_k(x, "K-Means+原始数据", k_range=range(6, 10))
    results['KMeans-原始'] = evaluate_kmeans(x, y, best_k1)
    best_k2 = find_best_k(x_scaled, "K-Means+标准化数据", k_range=range(6, 10))
    results['KMeans-标准化'] = evaluate_kmeans(x_scaled, y, best_k2)
    best_k3 = find_best_k(x_pca_scaled, "K-Means+PCA降维数据", k_range=range(6, 10))
    results['KMeans-PCA降维'] = evaluate_kmeans(x_pca_scaled, y, best_k3)

    # ========== 实验 B: PCA 降维到 2 维，可视化 ==========
    print("\n" + "=" * 50)
    print("PCA 2D 可视化")
    print("=" * 50)

    pca_2d = PCA(n_components=2)
    x_2d = pca_2d.fit_transform(x_scaled)  # 标准化数据降维
    km_2d = KMeans(n_clusters=7, random_state=42, n_init=50)
    labels_2d = km_2d.fit_predict(x_2d)
    centers = km_2d.cluster_centers_ #聚类中心
    #计算每个真实类别的数据中心（各类样本的坐标均值）
    true_centers = np.array([x_2d[y == label].mean(axis=0) for label in np.unique(y)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 左图：聚类结果
    # 统计每个簇里各类别的数量
    cluster_label_map = {}
    for cluster_id in np.unique(labels_2d):
        # 取出这个簇中所有样本的真实标签
        true_in_cluster = y[labels_2d == cluster_id]
        # 找出现最多的真实类别
        dominant = pd.Series(true_in_cluster).value_counts().idxmax()
        dominant_count = pd.Series(true_in_cluster).value_counts().max()
        total = len(true_in_cluster)
        cluster_label_map[cluster_id] = f'{dominant}\n({dominant_count}/{total})'
    
    for cluster_id in np.unique(labels_2d):
        mask = labels_2d == cluster_id
        ax1.scatter(x_2d[mask, 0], x_2d[mask, 1], label=cluster_label_map[cluster_id], alpha=0.6, s=10)
    ax1.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=200, label='聚类中心')
    ax1.set_title("K-Means 聚类结果")
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 右图：真实标签
    for name in np.unique(y):
        mask = y == name
        ax2.scatter(x_2d[mask, 0], x_2d[mask, 1], label=name, alpha=0.6, s=10)
    ax2.scatter(true_centers[:, 0], true_centers[:, 1], c='red', marker='*', s=200, label='类别中心')
    ax2.set_title("真实标签")
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()