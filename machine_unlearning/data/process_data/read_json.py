import json
import numpy as np

# if __name__ == "__main__":
#     with open("../train_meta_data_streaming4train_step20sec.json", 'r') as f:
#         data = json.load(f)
#         # 提取所有患者ID
#         patient_ids = set()
#         for item in data["meta_data"]:
#             path_info = item["path"]
#             path_parts = path_info.split('-')
#             patient_id = path_parts[0]
#             patient_ids.add(patient_id)
#     np.save("patient_ids.npy", np.array(list(patient_ids)))
    # # 将患者ID保存到文件
    # with open('patient_ids.txt', 'w') as f:
    #     for patient_id in patient_ids:
    #         f.write(patient_id + '\n')


if __name__ == "__main__":
    import json

    # 读取数据
    with open("../test_meta_data_streaming4train_step20sec.json", 'r') as f:
        data = json.load(f)

        # 创建字典来存储每个患者的数据数量
        patient_data_count = {}

        # 提取所有患者ID
        for item in data["meta_data"]:
            path_info = item["path"]
            path_parts = path_info.split('-')
            patient_id = path_parts[0]

            # 如果患者ID不在字典中，将其添加并初始化为0
            if patient_id not in patient_data_count:
                patient_data_count[patient_id] = 0

            # 对应患者ID的数据数量加1
            patient_data_count[patient_id] += 1
    print("总的患者数量{}".format(len(patient_data_count)))
    counts=0
    # 打印每个患者的数据数量
    for patient_id, count in patient_data_count.items():
        counts+=count
        print(f"Patient ID: {patient_id}, Data Count: {count}")
    print("总的记录数量{}".format(counts))
