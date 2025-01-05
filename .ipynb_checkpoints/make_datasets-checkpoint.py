import os
import shutil
import random
import json

# 数据集文件夹路径
dataset_folder = "data"
# 训练集文件夹路径
train_folder = "dataset/train_folder"
# 测试集文件夹路径
test_folder = "dataset/test_folder"
# metadata 文件路径
train_metadata_file = "dataset/train_folder/metadata.jsonl"
test_metadata_file = "dataset/test_folder/metadata.jsonl"

# 创建训练集和测试集文件夹（如果不存在）
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# 获取数据集中所有文件的列表
file_list = os.listdir(dataset_folder)
random.shuffle(file_list)

# 计算划分的索引
split_index = int(len(file_list) * 0.8)

# 用于存储元数据的信息
train_metadata = []
test_metadata = []
additional_feature = "a traditional Chinese New Year painting style of a warrior figure, vibrant colors, detailed lines"

# 拆分文件并复制到相应文件夹
for i, file_name in enumerate(file_list):
    source_path = os.path.join(dataset_folder, file_name)
    if i < split_index:
        file_name = f"train_{i}.{file_name.split('.')[-1]}"
        destination_path = os.path.join(train_folder, file_name)
        train_metadata.append({
            "file_name": file_name,
            "text": additional_feature
        })
    else:
        file_name = f"test_{i - split_index}.{file_name.split('.')[-1]}"
        destination_path = os.path.join(test_folder, file_name)
        test_metadata.append({
            "file_name": file_name,
            "text": additional_feature
        })
    shutil.copy2(source_path, destination_path)


# 将元数据写入 metadata.jsonl 文件
with open(train_metadata_file, 'w') as f:
    for item in train_metadata:
        f.write(json.dumps(item) + "\n")

with open(test_metadata_file, 'w') as f:
    for item in test_metadata:
        f.write(json.dumps(item) + "\n")

print("数据集拆分和封装完成，metadata.jsonl 文件已生成。")
