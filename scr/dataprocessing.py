import pandas as pd
import re
from html import unescape
import os

def clean_and_filter_data(input_path, output_path, max_words = 50):
    """
    数据清洗全流程：
    1. 读取数据
    2. HTML 实体解码 (&lt; -> <)
    3. 正则移除 <...> 标签内容
    4. 清理多余空格
    5. 计算词数并过滤 (word_count <= max_words)
    6. 【新增】保存清洗后的数据到指定路径
    :param input_file:输入文件路径 (例如: '../data/spam.csv')
    :param output_file:输出文件路径 (例如: '../data/spam_cleaned.csv')
    :param max_word:允许的最大词数阈值
    :return: df_clean: 清洗后的 DataFrame
    """

    # --- 1. 读取数据 ---
    # --- 1. 读取数据 ---
    try:
        # 注意：spam.csv 通常使用 cp1252 编码，且没有表头，需要手动指定列名
        df = pd.read_csv(
            input_path,
            sep=',',
            encoding='cp1252',
            encoding_errors='ignore',
            header=0,
            usecols=[0, 1],
            names=['label', 'message']
        )
        print(f" 成功读取数据，原始行数: {len(df)}")
    except FileNotFoundError:
        print(f" 错误：找不到文件 {input_path}")
        return None
    except Exception as e:
        print(f"读取失败: {e}")
        return None

    # --- 2. 基础预处理 ---
    # 填充空值，防止后续操作报错
    df['message'] = df['message'].fillna('').astype(str)
    original_count = len(df)

    # --- 3. 核心清洗逻辑 ---

    # Step A: HTML 实体解码 (把 &lt;DECIMAL&gt; 变成 <DECIMAL>)
    # apply + unescape 是最稳妥的方法
    df['message'] = df['message'].apply(lambda x: unescape(x))

    # Step B: 移除所有 <...> 形式的标签 (包括 <DECIMAL>, <#>, 等)
    # 正则解释: < 开头，中间任意非 > 字符，> 结尾
    df['message'] = df['message'].str.replace(r'<[^>]+>', '', regex=True)

    # Step C: 清理多余空格 (多个空格变一个，去除首尾空格)
    df['message'] = df['message'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Step D: 再次检查是否有空字符串 (有些行可能全是标签，清洗后变空了)
    df = df[df['message'].str.len() > 0].reset_index(drop=True)

    # --- 4. 长度过滤 ---

    # 计算词数 (按空格分割)
    df['word_count'] = df['message'].str.split().str.len()

    # 记录过滤前的数量
    count_before_filter = len(df)

    # 执行过滤：只保留 word_count <= max_words 的行
    df = df[df['word_count'] <= max_words]

    # 重置索引
    df = df.reset_index(drop=True)

    # 删除临时的 word_count 列 (如果不希望输出文件包含这一列)
    # 如果想保留统计列供后续分析，可以注释掉下面这行
    df_final = df.drop(columns=['word_count'])

    # --- 5. 统计与报告 ---
    final_count = len(df_final)
    removed_by_tag = original_count - count_before_filter  # 因全标签被清空的
    removed_by_len = count_before_filter - final_count  # 因长度过长被过滤的

    print("\n--- 清洗报告---")
    print(f"原始样本数:      {original_count}")
    print(f"因内容为空移除:   {removed_by_tag} (清洗后变为空字符串)")
    print(f"因长度>={max_words}移除: {removed_by_len}")
    print(f"最终保留样本数:   {final_count}")
    print(f"总保留比例:       {(final_count / original_count) * 100:.2f}%")

    # --- 6. 【关键】存储数据 ---

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f" 创建输出目录: {output_dir}")

    # 保存为 CSV (使用 utf-8-sig 编码，Excel 打开不会乱码)
    try:
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f" 数据已成功保存至: {output_path}")
    except Exception as e:
        print(f" 保存文件失败: {e}")
        return None

    # 打印前几行预览
    print("\n--- 数据预览 (前 3 行) ---")
    print(df_final.head(3))

    return df_final


if __name__ == '__main__':
    # 配置路径
    INPUT_FILE = '../data/spam.csv'  # 你的原始文件
    OUTPUT_FILE = '../data/spam_cleaned.csv'  # 清洗后要保存的文件名
    MAX_WORDS = 50  # 长度阈值

    # 执行清洗
    cleaned_df = clean_and_filter_data(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        max_words=MAX_WORDS
    )

    if cleaned_df is not None:
        print(" 处理全部完成！下一步可以使用 spam_cleaned.csv 进行训练。")
    else:
        print(" 处理过程中断，请检查上述错误信息。")