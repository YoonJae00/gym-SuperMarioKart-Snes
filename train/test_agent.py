import retro
import torch
import cv2
import numpy as np
from model import MarioRNN
from collections import deque

# 모델 로딩 및 환경 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MarioRNN().to(device)
model.load_state_dict(torch.load('mario_rnn.pth', map_location=device))
model.eval()

env = retro.make('SuperMarioKart-Snes')

# 버튼 인덱스 매핑 (동적)
buttons   = env.buttons
idx_B     = buttons.index('B')
idx_LEFT  = buttons.index('LEFT')
idx_RIGHT = buttons.index('RIGHT')

# 행동 단순화 함수
def to_raw_action(pred):
    a = np.zeros(len(buttons), dtype=np.uint8)
    a[idx_B] = 1
    if pred == 1:
        a[idx_LEFT] = 1
    elif pred == 2:
        a[idx_RIGHT] = 1
    return a

# 시퀀스 길이 및 상태 저장용 deque 설정
depth = 10

def reset_state_deque():
    obs = env.reset()
    state_deque = deque(maxlen=depth)
    for _ in range(depth):
        gray  = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (84,84)) / 255.0
        state_deque.append(small)
    return obs, state_deque

# 초기 상태
obs, state_deque = reset_state_deque()

print("에이전트 실행 중... Ctrl+C로 종료합니다.")
try:
    while True:
        # 모델 입력 생성: (1, depth, 1, 84, 84)
        seq = np.stack(state_deque)
        seq = np.expand_dims(seq, axis=0)
        seq = np.expand_dims(seq, axis=2)
        x = torch.tensor(seq, dtype=torch.float32).to(device)

        # 행동 예측 및 실행
        with torch.no_grad():
            logits = model(x)
            pred   = logits.argmax(dim=1).item()
        action = to_raw_action(pred)

        obs, _, done, _ = env.step(action)
        env.render()

        # 환경이 끝나면 재설정
        if done:
            print("환경 완료, 재설정 중...")
            obs, state_deque = reset_state_deque()
            continue

        # 다음 프레임 업데이트
        gray  = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (84,84)) / 255.0
        state_deque.append(small)

except KeyboardInterrupt:
    print("종료 요청 수신, 에이전트 중지")
finally:
    env.close()
