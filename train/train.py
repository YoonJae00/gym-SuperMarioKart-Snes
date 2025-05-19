# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loader
from dataset_multi import get_multi_loader
from model import MarioRNN


def train(epochs=10, lr=1e-3, data_path='train/data/episodes'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'▶ 학습 장치: {device}')
    # loader = get_loader(data_path, batch_size=16, seq_len=10) # 단일 파일 로더
    
    loader = get_multi_loader(data_path, batch_size=16, seq_len=10) # 다중 파일 로더
    
    model = MarioRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 정확도 계산
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        print(f'Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  accuracy={accuracy:.2f}%')

    torch.save(model.state_dict(), 'mario_rnn.pth')
    print('▶ 모델 저장됨: mario_rnn.pth')


if __name__ == '__main__':
    train()