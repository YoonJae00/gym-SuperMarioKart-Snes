# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MarioDataset(Dataset):
    def __init__(self, path, seq_len=10):
        arr    = np.load(path)
        frames = arr['frames']        # (N,84,84)
        actions= arr['actions']       # (N,)
        # 정규화 & 채널 차원 추가
        frames = frames.astype(np.float32) / 255.0
        frames = np.expand_dims(frames, 1)  # (N,1,84,84)
        self.X, self.y, self.seq_len = frames, actions, seq_len

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]     # (seq_len,1,84,84)
        y = self.y[idx + self.seq_len]         # scalar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def get_loader(path, batch_size=32, seq_len=10):
    ds = MarioDataset(path, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
