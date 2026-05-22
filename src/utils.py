## 工具函数——字体显示、模型评估、可视化、模型保存

from sklearn.metrics import (classification_report,confusion_matrix,roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def set_chinese_font():
    """配置 matplotlib 字体：中文宋体，英文和数字 Times New Roman
    依赖：Windows 系统自带 STSong/SimSun
    Linux/macOS 需安装 Noto Serif SC 或更换字体名"""
    plt.rcParams['font.family'] = ['serif', 'sans-serif']          # 先后搜索两个字体族
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']  # 英文 → TNR
    plt.rcParams['font.sans-serif'] = ['STSong', 'SimSun', 'Microsoft YaHei']  # 中文 → 宋体
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式匹配 Times 风格
    plt.rcParams['axes.unicode_minus'] = False

def model_evaluation(model,x_train,y_train, x_test,y_test):
    # 训练集准确率
    accuracy = model.score(x_train, y_train)
    print(f"模型在训练集上的准确率: {accuracy:.4f}")
    # 验证集准确率
    accuracy = model.score(x_test, y_test)
    print(f"模型在测试集上的准确率: {accuracy:.4f}")
    # 分类报告
    y_pred = model.predict(x_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

def plot_confusion_matrix(model, x_test, y_test, name):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = model.classes_  # 模型训练时自动记住了真实标签名
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    #plt.xlabel('Predicted')
    #plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

def ROC_curve(model, x_test, y_test, name):
    # 二分类或多分类都适用
    y_score = model.predict_proba(x_test)  # 获取预测概率
    labels = model.classes_
    y_test_bin = label_binarize(y_test, classes=labels)  # 二值化标签

    plt.figure(figsize=(8, 6))
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {labels[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    plt.show()

def plot_feature_importance(model, feature_names, name, top_n):
    if hasattr(model, 'feature_importances_'):    #检查模型是否具有特征重要性属性
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]  # 获取前top_n个重要特征的索引,argsort默认升序，取后top_n个再反转为降序
        plt.figure(figsize=(8, 6))
        plt.title(f'Feature Importances - {name}')
        plt.barh([feature_names[i] for i in indices], importances[indices], align='center')
        #plt.yticks(range(top_n), [feature_names[i] for i in indices])# 去掉rotation=90，不需要旋转
        plt.gca().invert_yaxis()  # 翻转 y 轴，最重要的排到最上面
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    else:
        print("模型不支持特征重要性评估。")