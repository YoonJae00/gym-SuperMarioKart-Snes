import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob

class MarioMultiDataset(Dataset):
    def __init__(self, npz_dir, seq_len=10):
        self.files = sorted(glob.glob(f"{npz_dir}/*.npz"))
        self.seq_len = seq_len
        # 미리 전체 데이터 길이 계산
        self.lens = []
        for f in self.files:
            arr = np.load(f)
            n = arr['actions'].shape[0]
            self.lens.append(n - seq_len)
        self.cum_lens = np.cumsum(self.lens)

    def __len__(self):
        return int(self.cum_lens[-1])

    def __getitem__(self, idx):
        # idx가 어느 파일인지 찾아내기
        file_idx = np.searchsorted(self.cum_lens, idx, side='right')
        start = idx - (self.cum_lens[file_idx-1] if file_idx>0 else 0)
        arr = np.load(self.files[file_idx])
        frames  = arr['frames'].astype(np.float32) / 255.0  # (N,84,84)
        actions = arr['actions']                           # (N,)
        # 시퀀스와 타깃
        x = frames[start:start+self.seq_len]               # (seq_len,84,84)
        y = actions[start+self.seq_len]                    # scalar
        x = np.expand_dims(x, 1)                           # (seq_len,1,84,84)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def get_multi_loader(npz_dir, batch_size=32, seq_len=10):
    ds = MarioMultiDataset(npz_dir, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
