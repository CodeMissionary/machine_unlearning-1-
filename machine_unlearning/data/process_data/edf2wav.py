# edf2wav.py
# 把 APNEA-EDF 中所有 EDF 文件的 Mic 通道导出为同名 wav
import os
from pathlib import Path

import mne
import torch
import torchaudio
import numpy as np
import pyedflib

EDF_ROOT = Path("/home/mzk/psg_audio/APNEA_EDF/APNEA_EDF")
WAV_ROOT = Path("/home/mzk/Mic_8000")
WAV_ROOT.mkdir(parents=True, exist_ok=True)

edf_files = sorted(EDF_ROOT.rglob("*.edf"))
print(f"共发现 {len(edf_files)} 个 edf 文件")

if not edf_files:
    raise RuntimeError("没有找到 .edf 文件，请检查 EDF_ROOT 路径。")

# 看一下通道名，用第一个文件做样例
raw_tmp = mne.io.read_raw_edf(edf_files[0], preload=False, verbose="ERROR")
print("通道列表:", raw_tmp.ch_names)
CH_NAME = "Mic"  # 这里固定用 Mic 通道

if CH_NAME not in raw_tmp.ch_names:
    raise ValueError(
        f"通道 {CH_NAME!r} 不在 EDF 通道列表中，请根据上面打印的通道列表修改 CH_NAME。"
    )


def load_channel_with_mne(edf_path, ch_name):
    """优先用 mne 读单通道"""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    raw.pick_channels([ch_name])
    data = raw.get_data()[0]  # (n_samples,)
    sfreq = raw.info["sfreq"]
    return data, sfreq


def load_channel_with_pyedflib(edf_path, ch_name):
    """mne 失败时，用 pyedflib 读单通道"""
    f = pyedflib.EdfReader(str(edf_path))
    labels = f.getSignalLabels()
    if ch_name not in labels:
        f.close()
        raise RuntimeError(f"pyedflib 也找不到通道 {ch_name}，labels={labels}")
    ch_idx = labels.index(ch_name)
    n_samples = f.getNSamples()[ch_idx]
    signal = f.readSignal(ch_idx)  # numpy array
    sfreq = f.getSampleFrequency(ch_idx)
    f.close()
    data = np.asarray(signal, dtype=np.float32)
    if data.shape[0] != n_samples:
        print(f"  [pyedflib] 样本数不一致: header={n_samples}, data={data.shape[0]}")
    return data, sfreq


for i, edf_path in enumerate(edf_files, start=1):
    wav_name = edf_path.name.replace(".edf", ".wav")
    out_path = WAV_ROOT / wav_name

    print(f"\n[{i}/{len(edf_files)}] Processing {edf_path.name}")

    if out_path.exists():
        print("  已存在，跳过 ->", out_path)
        continue

    # 先尝试 mne
    try:
        data, sfreq = load_channel_with_mne(edf_path, CH_NAME)
        print("  使用 mne 读取成功")
    except Exception as e_mne:
        print(f"  mne 读取失败：{repr(e_mne)}")
        print("  尝试使用 pyedflib 读取...")
        try:
            data, sfreq = load_channel_with_pyedflib(edf_path, CH_NAME)
            print("  使用 pyedflib 读取成功")
        except Exception as e_py:
            print(f"  pyedflib 也读取失败，跳过该文件: {edf_path.name}")
            print("  错误信息:", repr(e_py))
            continue  # 实在不行就跳过这个 EDF

    waveform = torch.from_numpy(data).unsqueeze(0).to(torch.float32)
    print(f"  writing {out_path}  sr={int(sfreq)}")
    torchaudio.save(str(out_path), waveform, int(sfreq))

print("\n全部 EDF -> WAV 转换完成。")
