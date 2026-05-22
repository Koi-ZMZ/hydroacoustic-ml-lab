'''KNN算法实验分类水声信号:
1. 原始数据（无标准化） 2. 标准化后  3. PCA 降维后'''

import sys
import os
# 把项目根目录加入 Python 搜索路径，这样才能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data  # 导入数据加载函数
from src.utils import plot_confusion_matrix, set_chinese_font, ROC_curve  # 导入模型评估和可视化函数
from sklearn.preprocessing import StandardScaler  # 用于数据标准化
from sklearn.neighbors import KNeighborsClassifier  # K近邻算法
from sklearn.decomposition import PCA  # 主成分分析（PCA）用于降维
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#网格搜索+评估管线
def evaluate_knn(x_train, y_train, x_test, y_test,name):
    # 使用网格搜索和交叉验证找到合适的参数
    knn = KNeighborsClassifier()
    param = {"n_neighbors": list(range(40, 50, 2))}
    grid_search = GridSearchCV(knn, param_grid=param, cv=4, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    # 训练集准确率
    accuracy_train = grid_search.score(x_train, y_train)
    # 验证集准确率
    accuracy_test = grid_search.score(x_test, y_test)
    # 分类报告
    y_pred = grid_search.predict(x_test)
    print(f"\n--- {name} ---")
    print("在交叉验证当中验证的最好结果：", grid_search.best_score_)
    print("grid_search选择了的模型参数是：", grid_search.best_estimator_)
    print(f"模型在训练集上的准确率: {accuracy_train:.4f}")
    print(f"模型在测试集上的准确率: {accuracy_test:.4f}")
    #print("\n分类报告:")
    #print(classification_report(y_test, y_pred))
    plot_confusion_matrix(grid_search.best_estimator_, x_test, y_test, name)# 绘制混淆矩阵
    ROC_curve(grid_search.best_estimator_, x_test, y_test, name)# 绘制ROC曲线
    return grid_search.best_estimator_

def main():
    set_chinese_font()
    #加载数据(无标准化)
    _,_,x_train, x_test, y_train, y_test,_ = load_data('data/dataset.csv')

    # 标准化后
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # PCA 降维后
    pca = PCA(n_components=0.5)  # 保留50%的方差
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    print(f"PCA降维后特征维度: {x_train_pca.shape[1]}")  # 输出降维后的特征维度
   
    # 评估不同预处理方法下的模型性能
    results = {}

    print("\n" + "=" * 50)
    print("KNN不同预处理方案对比")
    print("=" * 50)
    results['KNN-原始'] = evaluate_knn(x_train, y_train, x_test, y_test, "KNN+原始数据")
    results['KNN-标准化'] = evaluate_knn(x_train_scaled, y_train, x_test_scaled, y_test, "KNN+标准化数据")
    results['KNN-PCA降维'] = evaluate_knn(x_train_pca, y_train, x_test_pca, y_test, "KNN+PCA降维数据")

if __name__ == '__main__':
    main()