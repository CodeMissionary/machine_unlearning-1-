import os
import json

PROJECT_ROOT = "/home/mzk/machine_unlearning"
META_DIR = os.path.join(PROJECT_ROOT, "data")

# 只保留这些病人 ID（文件名前面的那部分）
KEEP_PATIENTS = ["00000995"]  # 你可以把需要的 ID 写在这里，比如 ["00000995", "00001234"]

splits = {
    "train": "train_meta_data_streaming4train_step20sec.json",
    "dev": "dev_meta_data_streaming4train_step20sec.json",
    "test": "test_meta_data_streaming4train_step20sec.json",
}

for split, fname in splits.items():
    meta_path = os.path.join(META_DIR, fname)
    print(f"\n=== 处理 {split}: {meta_path} ===")

    with open(meta_path, "r") as f:
        data = json.load(f)

    meta_data = data["meta_data"]
    print(f"原来样本数: {len(meta_data)}")

    kept = []
    for item in meta_data:
        wav_name = item["path"]  # 例如 "00000995-100507[001].wav"
        patient_id = wav_name.split('-')[0]
        if patient_id in KEEP_PATIENTS:
            kept.append(item)

    print(f"保留下来的样本数: {len(kept)}（只保留 {KEEP_PATIENTS}）")

    data["meta_data"] = kept

    out_path = os.path.join(META_DIR, f"{split}_meta_00000995.json")
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已保存: {out_path}")
