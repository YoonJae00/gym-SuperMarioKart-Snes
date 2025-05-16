## BizHawk + Lua + Python 연동
Python에서 서버를 실행하고 BizHawk 쪽에서 Lua 클라이언트 스크립트를 구동하여 서로 통신
Lua 클라이언트는 매 프레임 게임 화면(스크린샷 등)을 서버로 보내고 서버(Python)로부터 다음 조작 버튼을 받아와 에뮬레이터에 적용함

## RetroArch/Libretro + OpenAI Gym Retro 연동
 OpenAI의 Gym Retro 라이브러리는 인기 게임들을 강화학습 환경으로 사용할 수 있게 해주는데, Super Mario Kart (SNES)도 사용자 정의 통합을 통해 다룰 수 있음
 예를 들어, retro.make('SuperMarioKart-Snes')로 환경을 만들고, env.reset()으로 게임을 시작한 뒤 env.step(action)으로 한 프레임씩 진행하며 동작(action)을 입력할 수 있습니다

 
## Gym Retro 방식선택

### 10초 동안 우회전 예시 코드

```python
import retro, time
env = retro.make(game='SuperMarioKart-Snes', state='MarioCircuit1')  # 환경 생성
obs = env.reset()  # 게임 초기화
# 가속 + 우회전 액션 (예시로 [B버튼, Right버튼]만 True인 배열)
accelerate_right = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  
start_time = time.time()
while time.time() - start_time < 10:  # 10초간 실행
    obs, reward, done, info = env.step(accelerate_right)
    env.render()  # (선택) 그래픽 화면 렌더링
    if done:
        break
env.close()
```
