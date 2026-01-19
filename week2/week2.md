# Week 2 — Differentiable ASM Forward Model (PyTorch)

## 개요
2주차에서는 1주차에 구현한 Angular Spectrum Method(ASM) 전파 코드를 기반으로,  
이를 **미분 가능한(differentiable) forward model**로 확장하는 데 집중했다.

PyTorch의 자동 미분(`autograd`)을 활용하여 물리 기반 전파 연산을 **최적화·학습 파이프라인에 포함할 수 있는 구조**로 만드는 것이 핵심 목표였다.

---

## 학습 목표
- ASM 전파 연산을 **PyTorch `nn.Module` 구조로 래핑**
- 복소수 텐서에 대한 **gradient 흐름 확인**
- 물리 연산이 **loss.backward()**를 통해 미분 가능함을 검증
- 이후 위상 최적화 및 학습 단계로 확장 가능한 코드 구조 설계

---

## 핵심 구현 내용

### 1. ASM Forward Model 모듈화
- 기존 함수 기반 ASM 코드를 `nn.Module` 형태로 재구성
- 입력: 복소 전기장 \( U(x,y) \)
- 출력: 전파된 복소 전기장 \( U(x,y,z) \)

이를 통해 ASM을 단순 시뮬레이션 코드가 아닌, **재사용 가능한 연산 블록**으로 만들었다.

---

### 2. Gradient 흐름 검증
- 입력 전기장 또는 위상에 `requires_grad=True` 설정
- 전파 후 강도 기반 loss 정의
- `loss.backward()` 호출 후 gradient 값 확인

이를 통해 다음을 확인했다.
- FFT 기반 연산에서도 PyTorch autograd가 정상 동작함
- 물리 전파 연산이 **딥러닝 최적화 과정에 직접 포함 가능**함

---

### 3. PyTorch 중심 프로그래밍 학습
2주차에서는 딥러닝 모델을 사용하지 않았지만,  
다음과 같은 **딥러닝 프레임워크 핵심 요소**를 실제 코드에서 경험했다.

- `nn.Module` 기반 설계
- `nn.Parameter` 개념 이해
- 자동 미분(`autograd`)의 실제 동작 방식
- GPU(MPS) ↔ CPU 디바이스 관리

---

## 소프트웨어 학습 포인트
- 물리 수식 → PyTorch 연산 그래프 변환
- 복소수 연산과 gradient 계산의 공존
- 시뮬레이션 코드와 학습 코드의 경계 이해
- 이후 최적화(GS, gradient descent)로 자연스럽게 이어지는 구조 설계

---

## 학습한 점
- PyTorch는 신경망 학습뿐 아니라 **미분 가능한 물리 시뮬레이터**로 활용 가능하다.
- FFT, 복소수 연산, 물리 모델도 자동 미분 그래프에 포함될 수 있다.
- 딥러닝 이전 단계에서 **forward model을 정확히 설계하는 것이 매우 중요**함을 이해했다.

---

## 다음 단계 (3주차)
- ASM forward model을 사용한 **Gerchberg–Saxton(GS) 알고리즘 구현**
- 반복 최적화와 gradient 기반 최적화의 차이 비교
- 위상 복원 문제를 본격적인 최적화 문제로 해석

---

## 한 줄 요약
> **2주차에서는 ASM 전파 연산을 미분 가능한 PyTorch forward model로 확장하고, 물리 연산이 학습 파이프라인에 포함될 수 있음을 확인했다.**