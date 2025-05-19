import retro
import cv2
import numpy as np
import time
from pyglet.window import key
import os

def simplify_action(raw):
    if raw[idx_B] and raw[idx_LEFT]:   return 1  # 가속 + 좌
    if raw[idx_B] and raw[idx_RIGHT]:  return 2  # 가속 + 우
    if raw[idx_B]:                     return 0  # 가속만
    return 0

def collect(output_dir='train/data/episodes'):
    os.makedirs(output_dir, exist_ok=True)
    env = retro.make('SuperMarioKart-Snes')

    # 버튼 인덱스
    buttons = env.buttons
    global idx_B, idx_LEFT, idx_RIGHT
    idx_B     = buttons.index('B')
    idx_LEFT  = buttons.index('LEFT')
    idx_RIGHT = buttons.index('RIGHT')

    # 렌더러 & 키 콜백 설정 (이전과 동일)
    env.render()
    viewer = env.unwrapped.viewer
    window = viewer.window

    raw = np.zeros(env.action_space.shape, dtype=np.uint8)
    def on_key_press(symbol, modifiers):
        if symbol == key.Z:     raw[idx_B]     = 1
        if symbol == key.LEFT:  raw[idx_LEFT]  = 1
        if symbol == key.RIGHT: raw[idx_RIGHT] = 1
    def on_key_release(symbol, modifiers):
        if symbol == key.Z:     raw[idx_B]     = 0
        if symbol == key.LEFT:  raw[idx_LEFT]  = 0
        if symbol == key.RIGHT: raw[idx_RIGHT] = 0
    window.push_handlers(on_key_press, on_key_release)

    episode = 0
    try:
        while True:
            obs = env.reset()
            data = []

            print(f"\n▶ 에피소드 {episode} 시작 — 플레이 후 Ctrl+C 누르세요.")
            done = False
            while not done:
                obs, _, done, _ = env.step(raw)
                env.render()

                gray  = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (84, 84))
                a     = simplify_action(raw)
                data.append((small, a))

                time.sleep(0.01)

            # 에피소드 끝나면 저장
            filename = f'episode_{episode:03d}.npz'
            path = os.path.join(output_dir, filename)
            frames  = np.stack([f for f, _ in data])
            actions = np.array([a for _, a in data])
            np.savez_compressed(path, frames=frames, actions=actions)
            print(f"✔ 에피소드 {episode} 저장됨: {path}")
            episode += 1

    except KeyboardInterrupt:
        print("\n수집 중단 — 종료합니다.")
    finally:
        env.close()

if __name__ == '__main__':
    collect()
