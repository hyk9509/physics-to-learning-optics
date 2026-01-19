# Week 2 — Differentiable Angular Spectrum Method (PyTorch)

## 개요
2주차에서는 1주차에 구현한 Angular Spectrum Method(ASM) 전파 코드를 기반으로,  
이를 **PyTorch의 자동 미분(autograd)을 지원하는 미분 가능한 forward model**로 확장했다.

딥러닝 모델을 학습시키는 것이 목적이 아니라,  
**물리 기반 전파 연산이 PyTorch 연산 그래프에 포함되고 gradient가 실제로 계산되는지**를 검증하는 데 초점을 두었다.

---

## 학습 목표
- ASM 전파 연산을 `nn.Module` 형태로 모듈화
- 복소수 FFT 기반 물리 연산에서 **gradient 흐름 확인**
- Apple Silicon(M3 Pro) 환경에서 **MPS 백엔드 GPU 가속** 사용
- 이후 위상 최적화(GS, gradient descent)로 확장 가능한 구조 확보

---

## 구현 요약

### 1. Differentiable ASM Forward Model
- ASM 전파 연산을 PyTorch `nn.Module`로 구현
- 전파 전달 함수(transfer function)는 `buffer`로 등록하여 재사용
- 입력은 복소 전기장 \( U(x,y) \), 출력은 전파된 복소 전기장 \( U(x,y,z) \)

이를 통해 ASM을 단순 시뮬레이션 코드가 아닌,  
**최적화·학습 파이프라인에 포함 가능한 연산 블록**으로 만들었다.

---

### 2. Gradient 검증 실험
- 학습 가능한 파라미터: 위상(phase)
- 복소 전기장: \( U_0 = \exp(i \cdot \text{phase}) \)
- Loss: 전파 후 강도(intensity)와 타깃 강도의 MSE
- `loss.backward()` 호출 후 `phase.grad` 확인

실행 결과:
```text
device: mps
torch: 2.9.1
mps: True
loss: 3.340844e-02
phase.grad | mean: 1.268439e-05, max: 8.193334e-05