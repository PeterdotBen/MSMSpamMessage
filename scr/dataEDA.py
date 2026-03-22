# from config import Config
from collections import Counter  # 分组聚合操作
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 清除可能的缓存
plt.rcParams.update(plt.rcParamsDefault)

# 2. 设置字体列表 (Windows 下 SimHei 最稳，Mac 下 Arial Unicode MS 最稳)
# 注意：如果是在 Linux 服务器，需要安装中文字体才能生效
system_fonts = ['SimHei', 'Microsoft YaHei', 'Microsoft YaHei UI', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['font.sans-serif'] = system_fonts
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def data_EDA(path):
    # todo:1-获取数据源
    df = pd.read_csv(path, sep=',',
                     encoding='cp1252',
                     header=0,
                     usecols=[0, 1],
                     names=['label', 'message'])
    # todo:2-数据特征
    # print(df)
    # print(df.shape,len( df))
    # print(df.info())

    # todo:3统计不同标签的样本数,样本占比
    label_count = Counter(df['label'])
    # print('label_count--->', label_count)
    # exit()
    for label, count in label_count.items():
        print(f"标签:{label}对应的样本数:{count}")
    total_samples = len(df)
    for label, count in label_count.items():
        percent = count / total_samples * 100
        print(f"标签:{label}对应的样本占比:{percent:.2f}%")
    # todo:3.3-统计文本长度,最大长度,最小长度,平均长度
    df['word_len'] = df['message'].str.split().str.len()
    print(df.head)
    print(df['word_len'].describe())
    print(df['word_len'].max())
    #
    # print('longest message --->',df.loc[df['word_len'].idxmax(),'message'])
    # print('longest message --->',df.loc[df['word_len'].idxmin(),'message'])
    sorted_word_len_counts = sorted(Counter(df['word_len']).items(), key=lambda x: x[0], reverse=False)
    print('sorted_word_len_counts--->', sorted_word_len_counts)

    plt.style.use('seaborn-v0_8-whitegrid')  # 设置风格

    # 创建画布，大小为 (12, 5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 直方图 + 密度曲线 (看整体分布)
    sns.histplot(data=df, x='word_len', bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Histogram', fontsize=14)
    axes[0].set_xlabel('Characters Length')
    axes[0].set_ylabel('Count')
    # 标记出 95% 分位线，辅助决策截断位置
    q95 = df['word_len'].quantile(0.95)
    axes[0].axvline(q95, color='red', linestyle='--', label=f'95% quantile line ({q95:.0f})')
    axes[0].legend()

    # 图2: 箱线图 (看异常值和集中趋势)
    sns.boxplot(y=df['word_len'], ax=axes[1], color='lightgreen')
    axes[1].set_title('Boxplot', fontsize=14)
    axes[1].set_ylabel('Characters')

    # 在箱线图旁标注具体数值
    median_val = df['word_len'].median()
    axes[1].text(0.5, median_val, f'median_val: {median_val}',
                 ha='center', va='bottom', color='darkgreen', fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_EDA('../data/spam.csv')
