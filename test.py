import torch
import sys
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"현재 PyTorch 버전: {torch.__version__}")
print(f"CUDA 버전: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
else:
    print("\n문제 해결을 위한 정보:")
    print(f"Python 경로: {sys.executable}")
    print(f"PyTorch 경로: {torch.__file__}")