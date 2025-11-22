import os, json

PROJECT_ROOT = "/home/mzk/machine_unlearning"
WAV_DIR = "/home/mzk/Mic_8000"
META_DIR = os.path.join(PROJECT_ROOT, "data")

labels = {"Normal": 0, "OSA": 1}  # 先占两个类名，实际只用到 Normal

meta_data = []
for i, fname in enumerate(sorted(os.listdir(WAV_DIR))):
    if not fname.lower().endswith(".wav"):
        continue
    item = {
        "path": fname,  # 必须和 WAV 文件名完全一致
        "idx": str(i),
        "start": 0,
        "duration": 40,  # 随便先设 40 秒一段
        "end": 40,
        "label": "Normal",
    }
    meta_data.append(item)

data = {"labels": labels, "meta_data": meta_data}

for split in ["train", "dev", "test"]:
    out_path = os.path.join(META_DIR, f"{split}_meta_dummy.json")
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("写入:", out_path, "样本数:", len(meta_data))
