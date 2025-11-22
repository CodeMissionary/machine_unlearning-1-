import os
import torch
import torchaudio
import h5py
from tqdm import tqdm

try:
    # 从 train.py 等模块方式导入时走这里
    from data.process_data.dataset import PSGAudioDataset, collate_fn
except ModuleNotFoundError:
    # 如果你以后在 data/process_data 目录里直接 python extract_fbank_feature.py，
    # 当前目录有 dataset.py，也能走到这里
    from dataset import PSGAudioDataset, collate_fn
from torch.utils.data import DataLoader
from s3prl import hub
from s3prl.upstream.baseline.hubconf import *
from s3prl.upstream.interfaces import Featurizer

torchaudio.set_audio_backend('sox_io')
from torch.utils.data import Dataset


class PSGDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = os.path.join(self.root_dir, self.file_list[idx])
        with h5py.File(name, 'r') as file:
            group_name = self.file_list[idx]
            group = file[group_name]
            return group['data'][:], group['label'][()], group.attrs['idx'], group.attrs['filename']


# 按照患者划分,不包含指定患者的数据
class TrainDataset(Dataset):
    def __init__(self, root_dir, partial_folder, include=False):
        self.root_dir = root_dir
        self.partial_folder = partial_folder
        self.include = include
        self.filepaths = self._get_filepaths()

    def __len__(self):
        return len(self.filepaths)

    def _get_filepaths(self):
        filepaths = []
        for folder in os.listdir(self.root_dir):
            if folder == self.partial_folder and self.include:
                folder_path = os.path.join(self.root_dir, folder)
                if os.path.isdir(folder_path):
                    files = os.listdir(folder_path)
                    for file in files:
                        filepaths.append(os.path.join(folder_path, file))
            elif folder != self.partial_folder and not self.include:
                folder_path = os.path.join(self.root_dir, folder)
                if os.path.isdir(folder_path):
                    files = os.listdir(folder_path)
                    for file in files:
                        filepaths.append(os.path.join(folder_path, file))
        return filepaths

    def __getitem__(self, idx):
        name = self.filepaths[idx]
        with h5py.File(name, 'r') as file:
            group_name = os.path.basename(self.filepaths[idx])
            group = file[group_name]
            return group['data'][:], group['label'][()], group.attrs['idx'], group.attrs['filename']


class TrainAllDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filepaths = self._get_filepaths()

    def __len__(self):
        return len(self.filepaths)

    def _get_filepaths(self):
        filepaths = []
        # 遍历当前目录下的所有文件和文件夹
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                # 拼接文件的完整路径
                file_path = os.path.join(root, file)
                # 将文件路径添加到列表中
                filepaths.append(file_path)
        return filepaths

    def __getitem__(self, idx):
        name = self.filepaths[idx]
        with h5py.File(name, 'r') as file:
            group_name = os.path.basename(self.filepaths[idx])
            group = file[group_name]
            return group['data'][:], group['label'][()], group.attrs['idx'], group.attrs['filename']


def extract(upstream, featurizer, dataloader, folder_path, device):
    for batch_id, (wavs, labels, filenames, idx) in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        wavs = [torch.FloatTensor(wav).to(device) for wav in wavs]
        with torch.no_grad():
            features = upstream(wavs)
        features = featurizer(wavs, features)
        for i in range(len(features)):
            filename = '{}_{}.h5'.format(filenames[i], idx[i])
            name = os.path.join(folder_path, filename)
            if os.path.exists(name):
                continue
            with h5py.File(name, "w") as hdf_file:
                group = hdf_file.create_group(filename)
                # 将张量数据保存为数据集
                group.create_dataset("data", data=features[i].cpu())
                group.create_dataset("label", data=labels[i])
                # 将字符串数据保存为属性
                group.attrs["idx"] = idx[i]
                group.attrs["filename"] = filenames[i]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    import os

    # BASE_DIR 还是 process_data 这一层
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../data/process_data
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # .../machine_unlearning

    # 新增：json 所在的目录是 data 这一层
    META_DIR = os.path.dirname(BASE_DIR)  # .../data

    # 原始 wav 目录
    DATA_ROOT = "/home/mzk/Mic_8000"

    # meta json 路径 ✅ 改成从 META_DIR 里取
    train_path = os.path.join(META_DIR, "train_meta_dummy.json")
    val_path = os.path.join(META_DIR, "dev_meta_dummy.json")
    test_path = os.path.join(META_DIR, "test_meta_dummy.json")

    # 特征输出目录：data/all_data/{train,validation,test}
    FEAT_ROOT = os.path.join(PROJECT_ROOT, "data", "all_data")
    train_out = os.path.join(FEAT_ROOT, "train")
    val_out = os.path.join(FEAT_ROOT, "validation")
    test_out = os.path.join(FEAT_ROOT, "test")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    class_num = 2
    batch_size = 256
    sampler = None
    num_workers = 6

    train_dataset = PSGAudioDataset(DATA_ROOT, train_path, False, class_num)
    val_dataset = PSGAudioDataset(DATA_ROOT, val_path, False, class_num)
    test_dataset = PSGAudioDataset(DATA_ROOT, test_path, False, class_num)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=num_workers, drop_last=False,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        sampler=sampler, num_workers=num_workers, drop_last=False,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        sampler=sampler, num_workers=num_workers, drop_last=False,
        collate_fn=collate_fn
    )

    upstream = getattr(hub, "fbank")().to(device)
    featurizer = Featurizer(
        upstream=upstream, feature_selection="hidden_states", layer_selection=None,
        upstream_device=device, normalize=False
    ).to(device)

    folder_path = val_out
    extract(upstream, featurizer, val_dataloader, folder_path, device)

    folder_path = train_out
    extract(upstream, featurizer, train_dataloader, folder_path, device)

    folder_path = test_out
    extract(upstream, featurizer, test_dataloader, folder_path, device)
