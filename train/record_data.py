import retro
import cv2
import numpy as np
import time
from pyglet.window import key

def simplify_action(raw):
    if raw[idx_B] and raw[idx_LEFT]:   return 1  # 가속 + 좌
    if raw[idx_B] and raw[idx_RIGHT]:  return 2  # 가속 + 우
    if raw[idx_B]:                     return 0  # 가속만
    return 0

def collect():
    env = retro.make('SuperMarioKart-Snes')
    obs = env.reset()

    # 1) 버튼 순서 가져오기
    buttons = env.buttons
    global idx_B, idx_LEFT, idx_RIGHT
    idx_B     = buttons.index('B')
    idx_LEFT  = buttons.index('LEFT')
    idx_RIGHT = buttons.index('RIGHT')
    print(f"Mapping → B:{idx_B}, LEFT:{idx_LEFT}, RIGHT:{idx_RIGHT}")

    # 2) 첫 렌더링으로 viewer 생성
    env.render()
    viewer = env.unwrapped.viewer
    window = viewer.window

    # 3) raw 액션 버퍼
    raw = np.zeros(env.action_space.shape, dtype=np.uint8)

    # 4) 키 이벤트 연결
    def on_key_press(symbol, modifiers):
        if symbol == key.Z:     raw[idx_B]     = 1
        if symbol == key.LEFT:  raw[idx_LEFT]  = 1
        if symbol == key.RIGHT: raw[idx_RIGHT] = 1

    def on_key_release(symbol, modifiers):
        if symbol == key.Z:     raw[idx_B]     = 0
        if symbol == key.LEFT:  raw[idx_LEFT]  = 0
        if symbol == key.RIGHT: raw[idx_RIGHT] = 0

    window.push_handlers(on_key_press, on_key_release)

    data = []
    print("데이터 수집 시작! 렌더 창을 클릭해 포커스를 준 뒤 Z·←·→ 키로 플레이하세요. Ctrl+C로 중단하고 저장합니다.")

    try:
        while True:
            obs, _, done, _ = env.step(raw)
            env.render()

            gray  = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (84, 84))
            a     = simplify_action(raw)
            data.append((small, a))

            if done:
                obs = env.reset()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print(f"\n수집 중단 → 총 {len(data)} 샘플 저장 중…")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'train\data\mario_data_{timestamp}.npz'
        np.savez_compressed(
            filename,
            frames=np.stack([f for f, _ in data]),
            actions=np.array([a for _, a in data])
        )
        print(f"저장 완료: {filename}")
    finally:
        env.close()

if __name__ == '__main__':
    collect()
