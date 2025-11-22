import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import mne

# ========= 路径配置 =========
# EDF 所在目录
EDF_ROOT = "/home/mzk/machine_unlearning/data/raw_edf"

# 转出来的 wav 要放到这里
WAV_ROOT = "/home/mzk/Mic_8000"

# 要用的通道名（根据你的 EDF 里的实际通道改）
TARGET_CHANNEL = "Mic"  # 例如 "Mic"、"Snore"、"C3-A2" 等

# 目标采样率
TARGET_SR = 8000
# ===========================

os.makedirs(WAV_ROOT, exist_ok=True)


def edf_to_wav(edf_path: str):
    print(f"Processing {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    if TARGET_CHANNEL in raw.ch_names:
        picks = mne.pick_channels(raw.ch_names, [TARGET_CHANNEL])
    else:
        picks = [0]
        print(f"[WARN] channel {TARGET_CHANNEL!r} not found in {edf_path}, "
              f"use {raw.ch_names[0]!r} instead.")

    data, _ = raw[picks, :]
    data = data.squeeze()
    sr = int(raw.info["sfreq"])

    if sr != TARGET_SR:
        data = resample_poly(data, TARGET_SR, sr)
        sr = TARGET_SR

    data = data / (np.max(np.abs(data)) + 1e-8)
    data_int16 = (data * 32767).astype(np.int16)

    base = os.path.splitext(os.path.basename(edf_path))[0]
    wav_path = os.path.join(WAV_ROOT, base + ".wav")
    wavfile.write(wav_path, sr, data_int16)
    print(f"  Saved to {wav_path}")


if __name__ == "__main__":
    for name in sorted(os.listdir(EDF_ROOT)):
        if not name.lower().endswith(".edf"):
            continue
        edf_path = os.path.join(EDF_ROOT, name)
        edf_to_wav(edf_path)
