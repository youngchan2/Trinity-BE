# Kernel Providers

`KernelProvider`는 `PhysicalPlan`의 `Operation`과 `Statement`를 받아, 구현을 반환하고 코드를 출력합니다.

## Kernel

각 `KernelProvider`는 `Kernel`을 반환합니다. 한 `Kernel`은 `Operation`, 입력, 출력, 자원 및 동기화 요구 사항 등으로 구성됩니다. Emit 파이프라인은 반환된 `Kernel`을 결합하고 하드웨어의 실행 단위에 알맞게 배치합니다.

`KernelProvider`는 생성된 `Kernel`의 특성에 따라 아래 두 종류의 `Kernel` 중 하나로 표현됩니다.

| 종류   | 제공하는 정보                                          | 결합과 실행                            |
| ------ | ------------------------------------------------------ | -------------------------------------- |
| Native | 조합 가능한 본문, 입출력·자원·동기화 요구사항          | 다른 Native 본문과 결합할 수 있습니다. |
| Opaque | 완성된 artifact, 인자·실행 계약, 외부 메모리 접근 정보 | 내부 본문과 결합하지 않고 실행합니다.  |

## 선택 정책

Emit은 각 `Operation` 마다 등록된 우선순위로 `KernelProvider`를 시도하고, 첫 지원 구현 하나를 선택합니다. 하나의 `PhysicalPlan`은 서로 다른 `KernelProvider`의 구현을 포함할 수 있습니다. 특정 연산을 모든 `KernelProvider`가 지원하지 않는다면, 오류를 출력합니다.

이 선택 정책은 Native collect 경로의 계약입니다. 현재 `emit()`에는 CuTe만 등록되어 있습니다.
`kernel_candidates`의 CuTe/Triton/Quack 후보 열거와 `emit_python`의 제한된 후보 비교는
별도 경로이며, Native/Opaque 통합 실행은 후속 작업입니다.
자세한 계약은 [Emission](../../emission.md), 전체 plan을 받는 Triton fallback은
[Triton provider](../../triton-provider.md)를 참고합니다.
