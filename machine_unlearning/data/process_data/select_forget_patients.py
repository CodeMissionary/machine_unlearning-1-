from train import setup_seed
import random
import numpy as np

# 设置全局随机种子
seed_value = 42
setup_seed(seed_value)
with open('patient_ids.txt', 'r') as f:
    patient_ids = f.read().splitlines()

# 随机选择10个患者
forget_patients = random.sample(patient_ids, 10)

# 将forget_patients转换为NumPy数组
forget_patients_np = np.array(forget_patients)

# 保存为.npy文件
np.save('forget_patients.npy', forget_patients_np)

# 选取的遗忘患者id['00001436' '00001369' '00001217' '00001318' '00001357' '00001182' '00001312' '00001288' '00001400' '00001258']
