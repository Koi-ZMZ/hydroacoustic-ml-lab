# 数据目录

将下载的 CSV 数据集放入此目录。

## 数据格式要求

CSV 文件，包含：
- 特征列（数值型）
- 标签列（分类目标）

## 示例

```
feature_1,feature_2,...,feature_n,label
0.12,0.34,...,0.56,class_a
0.23,0.45,...,0.67,class_b
```

## 推荐数据集

- Mendeley Data: [Lobster Sound Dataset](https://data.mendeley.com/datasets/wsm4vfxvsf)
  - MFCC 特征已提取，CSV 格式
  - CC BY 4.0 许可

## 数据集描述

- 一共有256个特征，格式为feature_1,feature_2,...,feature_256,species,filename

  54000 样本 × 256 特征（fft_bin_0 ~ fft_bin_255，FFT 频谱幅值，已归一化到 0~1）
  标签列：species（物种名称）
  文件名列：filename（wav 源文件）
  
  7 类水下声音分类：鲸鱼(17000)、海豚(10000)、船只(8000)、人类活动(8000)、海豹/海象/海牛(5000)、自然声音(4000)、海洋生物(2000)。
