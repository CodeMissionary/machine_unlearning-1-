import os
import shutil

# 源文件夹路径
source_folder = '/home/osa/all_data/all_data/train'

# 目标文件夹路径的基础名称
target_folder_base = '/home/osa/all_data/all_data/train_split_by_patient'

# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    # 源文件路径
    source_file = os.path.join(source_folder, filename)

    # # 根据文件名的patientid确定目标文件夹路径
    patientid = filename.split('-')[0]
    # target_folder = target_folder_base+patientid
    target_folder = os.path.join(target_folder_base, patientid)
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 目标文件路径
    target_file = os.path.join(target_folder, filename)

    # 移动文件
    shutil.move(source_file, target_file)
