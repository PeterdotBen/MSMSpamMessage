from os import remove

import pandas as pd
from sklearn.model_selection import train_test_split
# from torch.nn.functional import dropout


# path = r'D:\Workspace\stage10\MSMSpamMessage\data\spam.csv'
# df = pd.read_csv(path, sep=',',
#                       encoding='cp1252',
#                       header=0,
#                       usecols=[0, 1],
#                       names=['label', 'message'])
def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    将数据集划分 train_data,val_data 和test_data
    :param df: 原始数据df
    :param test_size:测试集占比 (例如 0.2 表示 20%)
    :param val_size:验证集占 "剩余部分" 的比例 (注意不是占总数的比例)
    :param random_state:随机种子，保证每次运行结果一致
    :return:train_df, val_df, test_df
    """

    # 【重要】防御性编程：确保没有空消息，防止模型训练报错
    original_len = len(df)
    df = df.dropna(subset=['message'])
    df = df[df['message'].str.strip() != '']
    if len(df) < original_len:
        print(f" 发现 {original_len - len(df)} 条空消息，已自动移除。")

    # todo:1- 先分出[test]和[val和train(临时集)]
    temp_df, test_df = train_test_split(df,
                                        test_size=test_size,
                                        stratify=df['label'],
                                        shuffle=True,
                                        random_state=random_state)
    # todo:2- 再分出[val]和[train]在临时集[temp_df]中
    # 假设val_size = 0.2
    actual_val_ratio = val_size / (1 - test_size)

    train_df, val_df = train_test_split(temp_df,
                                        test_size=actual_val_ratio,
                                        stratify=temp_df['label'],
                                        shuffle=True,
                                        random_state=random_state)

    print("-" * 30)
    print("数据集划分完成！")
    print(f"原始数据总量: {len(df)}")
    print(f"训练集 (Train): {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"验证集 (Val):   {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"测试集 (Test):  {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
    print("-" * 30)

    # 可选：打印各集合的标签分布，确认是否均衡
    print("\n标签分布检查 (比例应大致相同):")
    for name, data in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        dist = data['label'].value_counts(normalize=True) * 100
        print(f"{name}: 正常={dist.get('ham', 0):.1f}%, 垃圾={dist.get('spam', 0):.1f}%")

    return train_df, val_df, test_df


if __name__ == '__main__':
    # 1. 读取并清洗数据
    path = '../data/spam_cleaned.csv'
    df = pd.read_csv(path, sep=',', encoding='utf-8-sig')

    # 2. 执行划分 (比例设为 8:1:1)
    # 即：20% 测试，剩下的 80% 里再分 25% 做验证 (0.25*0.8 = 0.2)，剩下 60% 做训练
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.2)

    # 3. 保存为新的 CSV 文件 (可选，方便后续直接加载)
    train_df.to_csv('../data/train.csv', index=False)
    val_df.to_csv('../data/val.csv', index=False)
    test_df.to_csv('../data/test.csv', index=False)
    print("\n💾 已保存为 train.csv, val.csv, test.csv")
