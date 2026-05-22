## 加载水声数据集(CSV格式)

import pandas as pd
from sklearn.model_selection import train_test_split  # 用于数据分割

# 加载数据集并进行预处理
def load_data(file_path,test_size=0.3):
    data = pd.read_csv(file_path)
    data = data.drop(columns=['filename'], axis = 1)  # 删除不必要的列
    x = data.drop(columns=['species'], axis = 1)  # 特征数据
    y = data['species']  # 目标数据
    feature_names = x.columns.tolist()  # 获取特征名称列表
    print(f"数据集加载完成，类别数量：{y.nunique()}, \n类别名称：{y.unique().tolist()},\n特征维度: {x.shape[1]}, 样本数量: {x.shape[0]}")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42,stratify=y) #stratify=y 确保训练集和测试集中的类别分布与原始数据集相同
    print(f"训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")
    return x, y, x_train, x_test, y_train, y_test, feature_names

