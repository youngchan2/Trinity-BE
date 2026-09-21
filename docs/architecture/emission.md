# Emission

Emission은 [`PhysicalPlan`](planning.md)의 `Operation` 및 제어 구조를 바탕으로 구현을 선택하고 대상 하드웨어의 컴파일 입력을 구성합니다.

## 구성

| 영역                  | 책임                                        | 문서                                                    |
| --------------------- | ------------------------------------------- | ------------------------------------------------------- |
| Kernel Provider       | Operation 구현 명세와 선택 조건             | [Kernel Providers](emission/kernel-providers/README.md) |
| Hardware architecture | 결합 본문의 실행 배치, 자원 확정, 코드 출력 | [Hardware Architecture](emission/hardware/README.md)    |

향후 Provider 별 문서는 `emission/kernel-providers`에, 하드웨어 별 문서는 `emission/hardware`에 추가해주세요.

## 처리 경로

![](../images/trinity-lowering-emission.svg)

1. `prepare()`는 `PhysicalPlan`의 Buffer binding, dtype, shape, 저장 위치 등을 준비합니다.
2. 정의된 하드웨어와 Topology 등에 알맞는 Provider 구현을 선택합니다.
3. `combine()`은 구현을 결합하고 `CombinedPlan`을 반환합니다.
4. 각 아키텍쳐별 Renderer가 코드와 컴파일 입력을 작성합니다.

## 지원 범위

현재 [`emit()`](../../src/emit/mod.rs)의 지원 범위는 아래와 같습니다.

### 실행 경로

- Native Streamed CUDA

### 대상 하드웨어

- Hopper
- sm_89
- sm_120

### 연산

- Pointwise
- Reduction
- GEMM

## 확장

생성한 `CudaSource`는 [`compile`](../../src/compile/mod.rs) 단계에 전달합니다.
