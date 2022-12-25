import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import time
import os
# MAX_WAV_VALUE = 32768.0
import config


# from utils import process_text, pad_1D, pad_2D
# from utils import pad_1D_tensor, pad_2D_tensor

from tqdm import tqdm
from scipy.io.wavfile import read, write
from librosa.util import normalize
from melspec import MelSpectrogram, MelSpectrogramConfig
mel_config = MelSpectrogramConfig()
mel_converter = MelSpectrogram(mel_config)
hparams = config.Config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_to_buffer(hparams):
    buffer = list()
    # text = process_text(os.path.join("data", "train.txt"))
    wav_path = hparams.wav_path
    mel_dir_path = hparams.mels_path

    start = time.perf_counter()
    filenames = sorted([fn for fn in os.listdir(wav_path) if os.path.isfile(os.path.join(wav_path, fn))])

    for i in tqdm(range(len(filenames))):
        audio_gt_name = os.path.join(
            wav_path, filenames[i])
        sr, audio = read(audio_gt_name)
        audio = torch.from_numpy(audio).float()


        mel_path = os.path.join(mel_dir_path, "mel" + filenames[i][2:-4])
        if not os.path.isfile(mel_path):
            mel_gt_target = mel_converter(audio.unsqueeze(0))
            np.save(mel_path, mel_gt_target)
        else:
            mel_gt_target = torch.from_numpy(np.load(mel_path))
       
        buffer.append({"audio": audio, "mel_target": mel_gt_target})
#         print("Len audio mel", audio.shape, mel_gt_target.shape)
#         if len(buffer) >= 1:
#             break

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]

def collate_fn_tensor(batch):
    new_batch = {
        "audio":[],
        "mel":[]
    }
    max_len = 8192
    mel_lens = np.array([d['mel_target'].size(-1) for d in batch])
    audio_lens = np.array([d['audio'].size(-1) for d in batch])

    max_mel_len = np.max(mel_lens)
    max_audio_len = np.max(audio_lens)

    for i in range(len(batch)):
        current_audio = batch[i]['audio']
        current_mel = batch[i]['mel_target']
#         print("Current mel shape: ", current_mel.shape)
        new_batch['audio'].append(torch.nn.functional.pad(current_audio, (0, max_audio_len - len(current_audio))))
        new_batch['mel'].append(torch.nn.functional.pad(current_mel, (0, max_mel_len - current_mel.shape[-1])))
    
    new_batch['audio'] = torch.stack(new_batch['audio'])
    new_batch['mel'] = torch.stack(new_batch['mel'])

    return new_batch




if __name__ == "__main__":
    # TEST
    get_data_to_buffer()
