"""用随机森林进行水声信号分类，并分析特征重要性。
1. 原始数据（无标准化） 2. 标准化后  3. PCA 降维后
"""

import sys
import os
# 把项目根目录加入 Python 搜索路径，这样才能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data  # 导入数据加载函数
from src.utils import plot_confusion_matrix, set_chinese_font, ROC_curve, plot_feature_importance  # 导入模型评估和可视化函数
from sklearn.preprocessing import StandardScaler  # 用于数据标准化
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.decomposition import PCA  # 主成分分析（PCA）用于降维
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#网格搜索+评估管线
def evaluate_randomforest(x_train, y_train, x_test, y_test, name):
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)# 设置n_jobs=-1以利用所有CPU核心加速训练;
    param = {"n_estimators": [200],'max_depth': [20],"min_samples_leaf": [50]}  #所有参数暂时均已选出（后续可自行修改）
    grid_search = GridSearchCV(rf, param_grid=param, cv=3, n_jobs=-1)
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
    #加载数据（无标准化）
    _,_,x_train, x_test, y_train, y_test, feature_names = load_data('data/dataset.csv')
    
    # 标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # PCA 降维
    pca = PCA(n_components=0.5)  # 保留50%的方差
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    print(f"PCA降维后特征维度: {x_train_pca.shape[1]}")  # 输出降维后的特征维度

    # 评估不同预处理方法下的模型性能
    results = {}

    print("\n" + "=" * 50)
    print("随机森林不同预处理方案对比")
    print("=" * 50)
    results['随机森林-原始'] = evaluate_randomforest(x_train, y_train, x_test, y_test, "RF+原始数据")
    results['随机森林-标准化'] = evaluate_randomforest(x_train_scaled, y_train, x_test_scaled, y_test, "RF+标准化数据")
    results['随机森林-PCA降维'] = evaluate_randomforest(x_train_pca, y_train, x_test_pca, y_test, "RF+PCA降维数据")

    # 绘制特征重要性图
    model = results['随机森林-原始']  # 获取原始数据训练的模型
    plot_feature_importance(model, feature_names, "RF+原始数据", top_n=20)# 绘制特征重要性图

if __name__ == '__main__':
    main()    
 