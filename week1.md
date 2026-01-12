# Week 1 — PyTorch 기반 Angular Spectrum Method 시뮬레이터

## 개요
본 프로젝트는 **PyTorch**를 사용하여 **Angular Spectrum Method(ASM)** 기반의 자유공간 광학 전파 시뮬레이터를 구현한다.  
1주차의 주요 목표는 딥러닝 모델을 사용하는 것이 아니라, **PyTorch 텐서 연산, 복소수 FFT, 그리고 Apple Silicon(M3 Pro) 환경에서의 GPU 가속(MPS)**에 익숙해지는 것이다.

이 단계에서 PyTorch는 신경망 프레임워크가 아니라, **수치 계산과 미분 가능한 계산을 위한 범용 계산 프레임워크**로 사용된다. 이는 이후 최적화 및 학습 단계로 확장하기 위한 기반을 마련하기 위함이다.

---

## 학습 목표
- NumPy 없이 **PyTorch만 사용하여 ASM 구현**
- **복소수 텐서** 및 FFT 기반 연산 처리
- Apple Silicon 환경에서 **MPS 백엔드 GPU 가속 실행**
- **단일 슬릿 회절 패턴** 생성 및 시각화
- 이후 최적화에 사용할 수 있는 **forward propagation 모듈** 구축

---

## 물리 모델
단색파(monochromatic wave)에 대해 복소 전기장 \( U(x,y,z) \)는 다음의 Helmholtz 방정식을 만족한다.

\[
\nabla^2 U + k^2 U = 0
\]

횡방향(x, y)에 대해 2차원 푸리에 변환을 적용하면, 파면은 여러 평면파 성분의 합인 **Angular Spectrum**으로 표현된다.  
각 평면파 성분은 z 방향으로 독립적으로 전파되며, 다음과 같은 위상 인자를 갖는다.

\[
H(f_x, f_y, z) = \exp(j k_z z), \quad
k_z = \sqrt{k^2 - k_x^2 - k_y^2}
\]

전파 과정은 다음 순서로 구현된다.

1. FFT를 통해 각도 성분(angular spectrum) 계산  
2. 전파 전달 함수(transfer function)와 곱셈  
3. IFFT를 통해 전파된 공간 파면 복원  

---

## 구현 세부 사항
- **프레임워크**: PyTorch  
- **디바이스**: Apple Silicon GPU (`mps`), 불가능할 경우 CPU  
- **데이터 타입**: `torch.complex64`  
- **사용 연산자**:
  - `torch.fft.fft2 / ifft2`
  - `torch.exp`, `torch.abs`
- **입력장(Field)**:
  - 평면파(plane wave)
  - x 방향으로만 제한된 **1차원 단일 슬릿 개구**

전파 연산은 하나의 재사용 가능한 함수로 구현되며, 이후 주차에서 최적화 및 학습 파이프라인에 포함될 예정이다.

---

## 실행 방법
```bash
python main.py
