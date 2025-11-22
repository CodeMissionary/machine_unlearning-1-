import os
import shutil

# ===== 路径配置 =====
# 当前文件所在目录: .../machine_unlearning/data/process_data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录: .../machine_unlearning
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# 特征根目录: .../machine_unlearning/data/all_data
FEAT_ROOT = os.path.join(PROJECT_ROOT, "data", "all_data")

# 源文件夹：train 特征所在目录
source_folder = os.path.join(FEAT_ROOT, "train")

# 目标根目录：按病人划分后的目录
target_folder_base = os.path.join(FEAT_ROOT, "train_split_by_patient")
# ====================

# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    # 源文件路径
    source_file = os.path.join(source_folder, filename)

    # 根据文件名的 patientid 确定目标文件夹路径
    patientid = filename.split('-')[0]
    target_folder = os.path.join(target_folder_base, patientid)

    # 如果目标文件夹不存在，则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 目标文件路径
    target_file = os.path.join(target_folder, filename)

    # 移动文件
    shutil.move(source_file, target_file)
