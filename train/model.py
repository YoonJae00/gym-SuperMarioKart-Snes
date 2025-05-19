# model.py
import torch.nn as nn

class MarioRNN(nn.Module):
    def __init__(self, hidden_size=128, n_layers=1, n_actions=3):
        super().__init__()
        # CNN 특징 추출기
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> (16,42,42)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> (32,21,21)
            nn.ReLU(),
            nn.Flatten()                                # -> 32*21*21
        )
        feat_size = 32 * 21 * 21
        # LSTM
        self.lstm = nn.LSTM(input_size=feat_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True)
        # 출력층
        self.fc = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        # x: (B, seq_len,1,84,84)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feat = self.cnn(x)               # (B*S, feat_size)
        feat = feat.view(B, S, -1)       # (B, S, feat_size)
        out, _ = self.lstm(feat)         # (B, S, hidden)
        return self.fc(out[:, -1, :])    # (B, n_actions)
