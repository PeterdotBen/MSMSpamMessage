# print("Hello World")
import shutil

import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

print("Path to dataset files:", path)
target_path = r'D:\Workspace\stage10\MSMSpamMessage\data'
os.makedirs(target_path, exist_ok=True)

for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join(target_path, file)
    shutil.copy2(src,dst)

print(f"文件已复制到: {target_path}")
