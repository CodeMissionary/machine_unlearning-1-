# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
import random
import torch
from pathlib import Path
from os.path import join as path_join

import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
import datetime
SAMPLE_RATE = 8000 #16000
SEGMENT_SEC = 40
torchaudio.set_audio_backend('sox_io')
class PSGAudioDataset(Dataset):
    # data_dir: /home/osa/Mic_8000
    # meta_path: train_meta_data_streaming4train_step20sec.json
    def __init__(self, data_dir, meta_path, pre_load=False, class_num=3):
        self.data_dir = data_dir
        self.meta_path = meta_path
        self.pre_load = pre_load
        self.class_num = class_num
        if class_num == 2:
            self.class_dict = {'Normal': 0, 'Hypopnea': 1, 'ObstructiveApnea': 1, 'CentralApnea': 1, 'MixedApnea': 1}
        elif class_num == 3:
            self.class_dict = {'Normal': 0, 'Hypopnea': 1, 'ObstructiveApnea': 2, 'CentralApnea': 2, 'MixedApnea': 2}
        elif class_num == 5:
            self.class_dict = {'Normal': 0, 'Hypopnea': 1, 'ObstructiveApnea': 2, 'CentralApnea': 3, 'MixedApnea': 4}
        else:
            exit("Error: class_num is out of definition!")
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.meta_data = self.data['meta_data']
        _, self.origin_sr = torchaudio.load(path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(self.origin_sr, SAMPLE_RATE)
        self.num_pos_samples = len(self.meta_data)
        print(">>>>>>><<<<<<<<<<<<<<<<<<<MMMMMMM: ", self.num_pos_samples)


    def pad_or_truncate(self, x, audio_length):
        """Pad all audio to specific length."""
        if x.size(1) < audio_length:
            padding = torch.zeros((x.size(0), audio_length - x.size(1)))
            return torch.cat((x, padding), dim=1)
        elif x.size(1) > audio_length:
            return x[:, 0: audio_length]
        else:
            return x
            
    def _load_audio_segment(self, idx, start_s, duration_s):
        audio_file = path_join(self.data_dir, self.meta_data[idx]['path'])
        waveform, sample_rate = torchaudio.load(audio_file)
        start_point = int(start_s * sample_rate)
        end_point = int((start_s + duration_s) * sample_rate)

        # ends by the wav length
        if start_point >= 0 and end_point > len(waveform[0]) and idx == self.num_pos_samples - 1:
            # end_point = len(waveform[0])
            # segment = waveform[:, start_point:end_point]
            segment = waveform[:, start_point:]

        # borrow from next wav
        elif start_point >= 0 and end_point > len(waveform[0]) and idx < self.num_pos_samples - 1:
            # end_point = len(waveform[0])
            # segment_pre = waveform[:, start_point:end_point]
            segment_pre = waveform[:, start_point:]

            audio_file_post = path_join(self.data_dir, self.meta_data[idx+1]['path'])
            waveform_post, _ = torchaudio.load(audio_file_post)
            # duration_post = end_point - len(waveform[0])
            # segment_post = waveform_post[:, 0:duration_post]
            end_point = end_point - len(waveform[0])
            segment_post = waveform_post[:, :end_point]

            segment = torch.cat((segment_pre, segment_post), 1)

        # borrow from previous wav
        elif start_point < 0 and idx > 0:
            audio_file_pre = path_join(self.data_dir, self.meta_data[idx-1]['path'])
            waveform_pre, _ = torchaudio.load(audio_file_pre)
            segment_pre = waveform_pre[:, start_point:]

            segment_post = waveform[:, :end_point]

            segment = torch.cat((segment_pre, segment_post), 1)

        # start from wav beginning
        elif start_point < 0 and idx == 0:
            segment = waveform[:, :end_point]

        # segment in normal case
        else:
            segment = waveform[:, start_point:end_point]

        # print("+++++ final: ", self.num_pos_samples, idx, audio_file, sample_rate, start_s, duration_s, len(waveform[0]), end_point, len(segment[0]), datetime.datetime.now())
        return segment, sample_rate

    def _load_wav(self, idx, start_s, duration_s):
        wav, _ = self._load_audio_segment(idx, start_s, duration_s) # num_frames=int(duration_s * self.origin_sr), offset=int(start_s * self.origin_sr))
        wav = self.pad_or_truncate(wav, SAMPLE_RATE * SEGMENT_SEC)
        wav = self.resampler(wav).squeeze(0)
        return wav

    def __getitem__(self, idx):
        '''
        # 如果 idx 是一个切片对象，就返回切片范围内的样本
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self)
            step = idx.step if idx.step is not None else 1

            samples = []
            for i in range(start, stop, step):
                label = self.meta_data[i]['label']
                label = self.class_dict[label]

                if label == 0:
                    post_s = 0
                    start_s = self.meta_data[i]['start'] + post_s
                else:
                    post_s = random.choice([2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7,
                                            8]) if "train_meta" in self.meta_path else 0 if "streaming4infer" in self.meta_path else 5
                    start_s = self.meta_data[i]['start'] + self.meta_data[i]['duration'] + post_s - SEGMENT_SEC

                wav = self._load_wav(i, start_s, SEGMENT_SEC)
                samples.append((wav.numpy(), label, Path(self.meta_data[i]['path']).stem, i))
            return samples

        else:
        '''
        # print("+++++ idx: ", idx)
        if self.pre_load:  # ignore pre_load process; do use non-pre_load process
            wav = self.wavs[idx]
            label = self.meta_data[idx]['label'] if idx < self.num_pos_samples else self.class_dict['Normal']
            label = self.class_dict[label]
            return wav.numpy(), label, Path(
                self.meta_data[idx % self.num_pos_samples]['path']).stem  # assume that the path doesn't much matter

        else:
            # segment_s = 40.0  # fixed segment length
            label = self.meta_data[idx]['label']
            label = self.class_dict[label]

            # merge split audio segments into a long one to avoid event-broken !!!! problematic: very slow to load wav file
            if label == 0:
                # post_s = random.choice([0, 1, 2, 3, 4, 5, 6]) if "train_" in self.meta_path else 0 # delay 0s for normal samples in eval & test
                post_s = 0
                start_s = self.meta_data[idx]['start'] + post_s
            else:
                post_s = random.choice([2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7,
                                        8]) if "train_meta" in self.meta_path else 0 if "streaming4infer" in self.meta_path else 5
                ##post_s = random.choice([2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8]) if "train_" in self.meta_path else 5 # delay 5s for apnea samples in eval & test. if streaming infer, change 5s to 0s
                start_s = self.meta_data[idx]['start'] + self.meta_data[idx]['duration'] + post_s - SEGMENT_SEC

            wav = self._load_wav(idx, start_s, SEGMENT_SEC)
            return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem, idx

    def __len__(self):
        return len(self.meta_data)
        # return self.total_num_samples

def collate_fn(samples):
    return zip(*samples)



