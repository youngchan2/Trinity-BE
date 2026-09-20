# Emission

[`PhysicalPlan`](planning.md)에 표현된 연산과 Loop 구조에 맞는 구현을 선택하고,
선택된 구현을 결합하여 실행 계획과 코드를 생성한다.

이 문서는 Emit 재구성을 위한 구조 초안이다. 각 구성 요소의 역할과 경계를 먼저 정리하며,
구체적인 내부 자료구조와 Provider 인터페이스는 후속 설계에서 확정한다.
현재 [`emit()`](../../src/emit/mod.rs)은 `EmitError::Unavailable`을 반환한다.

## Kernel

Kernel은 Provider가 Operation과 Loop 문맥에 맞춰 제공하는 구현 결과다.
원시 커널 하나가 독립적으로 launch 가능한 CUDA kernel 하나를 의미하지는 않는다.
전체 Operation의 원시 커널을 선택한 뒤, 별도 결합 단계에서 실제 실행할 본문을 구성한다.

### Native / Opaque

| 종류 | 제공하는 정보 | 결합과 실행 |
| --- | --- | --- |
| Native | 조합 가능한 본문, 입출력·자원·동기화 요구 | 다른 Native와 본문 결합 가능. Streamed와 Persistent에서 사용 |
| Opaque | 완성된 kernel의 artifact, 인자·launch 계약, 외부 메모리 접근 정보 | 내부 본문 결합 없이 실행. 우선 Streamed만 지원 |

Native의 계산 본문은 일반 CUDA kernel의 block과 Persistent kernel의 Worker CTA에서
공통으로 사용할 수 있도록 실행 배치와 분리한다.

### 처리하는 Operation과 Loop 문맥

구현 결과는 자신이 처리하는 Operation 목록을 명시한다. 결합 후에도 이 정보를 유지해
원래 프로그램의 연산이 누락되거나 중복 실행되지 않는지 확인한다.

Loop 처리 범위는 Operand의 접근 범위와 전달된 Loop 문맥으로 결정한다. 별도의 Loop
소유 범위 선언을 추가하지 않는다. Loop를 Emit이 생성할지 구현 내부에서 처리할지는
Emit과 Provider 사이의 구현 계약에서 다룬다.

### 실행·자원 요구사항

구현 결과는 thread 구성, 구현 내부의 타일 크기·용량, pipeline, 임시 메모리,
alignment와 동기화 등 선택·결합·실행 배치에 필요한 요구사항을 제공한다.
요구사항의 구체적인 필드와 표현은 후속 설계에서 정한다.

논리적인 타일과 접근 범위는 Plan의 expression과 Loop를 따른다. Loop의 step은
반복 사이의 이동량이고 tile index의 width는 접근 폭이므로 두 값을 동일하게 취급하지 않는다.

### Thread 참여 계약

`KernelRequirements.thread_policy`는 구현 본문이 허용하는 thread 참여 방식을 명시한다.

| 계약 | 실행 제약 | 현재 구현 |
| --- | --- | --- |
| `FullCta { threads }` | 지정된 크기의 CTA 전체가 참여한다. CTA barrier를 사용할 수 있다. | Hopper/SM80 GEMM |
| `Flexible { supported }` | 지원 목록의 어느 크기로든 전체 CTA가 참여할 수 있다. 목록 순서는 선호도를 뜻하지 않는다. | 메모리 입력 pointwise |
| `Subgroup { threads }` | CTA의 앞쪽 N개 thread만 참여한다. N은 warp 크기의 배수이며 CTA barrier를 사용하지 않는다. | Warp 단위 reduction |
| `FollowInput` | 독립 thread 수를 요구하지 않고 Register 생산자의 thread·좌표·출력 scope를 따른다. | Register 입력 pointwise |

개별 `SpecifiedKernel`은 지원 제약만 보관한다. Fusion 중에는 같은 Provider의 연속 연산이
허용하는 CTA 크기의 교집합을 유지하고, 공통 크기가 있는 동안 하나의 묶음으로 구성한다.
공통 크기가 없거나 Provider가 바뀌면 본문을 분리한다. Register 전달을 위해 storage를
바꾸지는 않는다.

묶음이 확정되면 `choose_block_threads()`가 최종 CTA 크기를 한 번 선택해
`CombinedBody.requirements.block_threads`에 저장한다. 현재 정책은 128이 가능하면 128,
그렇지 않으면 가능한 최소 크기다. 성능 기반 선택은 이 단계에서 확장한다.
선택 과정은 개별 명세를 수정하지 않는다.

확정된 크기를 `SpecifiedKernel::interface(block_threads)`에 전달해 Register layout과
연결을 검증하고, `KernelBindings.block_threads`로 renderer에 전달한다. Provider는 이
크기로 layout과 본문을 생성해야 하며, 나머지 자원 요구사항은 지원하는 모든 크기에서
유효해야 한다. CTA 크기 결정 전의 Loop 처리는 크기와 무관한 `iteration()` 계약을 사용한다.

큰 CTA 안의 Subgroup 호출은 prologue·전체 누적 loop·epilogue와 Register continuation까지
하나의 참여 조건으로 감싼다. 본문 내 root 사이의 CTA barrier는 이 조건 바깥에 둔다.
GEMM의 CTA barrier를 부분 그룹 barrier로 자동 치환하거나 thread 사이의 Register 값을
자동 재분배하지 않는다. Streamed와 Persistent는 동일한 참여 계약을 사용한다.

## KernelProvider

![KernelProvider의 후보 명세 생성과 렌더링 구조](../images/trinity-lowering-kernel-provider.svg)

KernelProvider는 Operation의 의미와 실행 문맥을 받아, 지원 조건을 만족하는 구현 하나를 제공한다.
구현 선택과 소유는 Emit의 책임이며 IR Reader와 `PhysicalPlanBuilder`는 구현을 선택하지 않는다.
`specify()`는 단일 `SpecifiedKernel` 또는 미지원·오류를 반환하고, `render()`는 해당 명세를
커널 본문으로 렌더링한다. Provider 내부의 구현체도 같은 단일 명세 계약을 사용한다.

### 입력 문맥

Operation의 expression·Operand, 값의 dtype·shape·storage, 이를 감싸는 Loop와
실행 순서, target과 요청된 실행 방식을 전달한다.
Provider는 입력에 정해진 연산 의미와 논리적인 Loop·접근 범위를 보존해야 한다.

Single GPU용 텍스트 IR과 Multi-GPU용 Builder 입력은 같은 Emit 경로를 사용한다.
통신용 텍스트 IR 문법 확장은 후속 작업으로 남긴다. Builder의 `all_gather` expression은
통신 의미를 전달하며, peer push/pull이나 NVLS 같은 구현 선택은 Provider에서 다룬다.
표현과 rank별 배치는 [Planning](planning.md#operations)을 따른다.

### 선택 정책

각 Operation마다 Provider를 등록된 우선순위대로 시도하고, 처음 지원하는 Provider의 구현
하나를 선택한다. 하나의 Plan에서 서로 다른 Provider를 사용할 수 있다.
모두 미지원이면 해당 Operation과 Provider별 미지원 사유를 `EmitError::NoProvider`로
반환한다. Provider 내부 오류는 `EmitError::Provider`로 즉시 전달한다.
구현 선택 중에 주변 연산과의 결합까지 확정하지 않는다.

요청된 실행 방식에 대한 지원 여부도 선택 조건에 포함한다. Persistent 요청에서는
Opaque를 선택 대상에서 제외한다. Opaque를 사용하기 위해 실행 방식을 바꾸거나
Persistent 프로그램을 분할하는 정책은 이번 범위에서 제외한다.

## 커널 결합

전체 Operation의 원시 커널 선택이 완료되면, Loop 구조와 실행 순서에 따라 선택된 구현들을
조합한다. IR에서 이미 fusion된 프로그램이어도 실제 구현들의 결합 가능성을 검사한다.
하나의 결합 본문에는 같은 Provider의 구현만 포함한다. Provider가 바뀌는 지점에서 본문을
나누며 Inter-KernelProvider combine은 지원하지 않는다. Provider의 이름이 같아도
등록 항목이 다르면 별도 Provider로 취급한다. 메모리로 연결할 수 있는 본문은 분리하고,
Register처럼 같은 본문이 필요한 전달이 Provider 경계를 넘으면 결합 실패로 처리한다.
`combine()`은 선택된 명세들을 한 번 결합해 단일 `CombinedPlan`을 반환한다.
연산별 후보의 Cartesian product를 만들거나 여러 결합 결과를 열거하지 않는다.

### 결합 조건

- 연산 순서, 의존성과 누적값의 초기화·갱신·최종 저장 시점을 보존한다.
- 중간값의 dtype·layout·storage와 전달 방식이 호환된다.
- thread 구성과 필요한 동기화를 함께 제공할 수 있다.
- 결합한 구현의 자원 요구가 target과 실행 방식의 제약을 만족한다.

초기 구현에서는 결합 실패를 이유로 원시 커널 선택을 다시 탐색하지 않는다. 분리 실행이
유효하면 선택된 구현들을 별도로 사용한다. 같은 kernel에서 실행되어야 하는 프로그램은
결합할 수 없으면 미지원으로 처리한다. 이 과정에서 storage를 임의로 바꾸지 않는다.

### Prologue / Mainloop / Epilogue

결합 단계는 반복 전에 한 번 수행할 처리, 반복 본문, 반복 후 처리를 함께 구성한다.
예를 들어 GEMM의 누적값 초기화, K 반복, 최종 저장과 뒤따르는 pointwise 처리를
지원 조건에 따라 조합한다.

Bias나 activation은 Plan에 있는 Operation이고, 누적값 초기화나 pipeline 준비는
구현체가 생성하는 처리다. 결합 결과는 두 종류를 구분하고 처리한 Operation을 추적한다.

Prologue와 epilogue는 해당 Loop를 기준으로 배치한다. K Loop의 초기화는 출력
타일마다 수행되어야 하므로, 중첩 Loop 전체를 하나의 전역 3단계로 평탄화하지 않는다.

### 누적값 초기화

[Planning의 reduction 규약](planning.md#operations)에 따라 인식된 누적은 각 출력 타일에서
암묵적으로 0부터 시작한다. 별도의 초기화 Operation을 요구하지 않으며, Emit이 해당 순차
Loop에 진입하기 전에 초기화를 생성한다.

일반적인 `load(dst) + rhs`를 모두 0에서 시작하는 누적으로 해석하지 않는다.
기존 목적지 값에서 시작하는 누적은 필요할 때 별도의 의미 표현을 추가한다.

### Shared / Register의 범위와 수명

실제 저장 범위와 수명은 결합 단계에서 검사한다. 생산자와 소비자가 같은 유효 범위에
있는지, 필요한 전달과 동기화가 가능한지, 마지막 사용까지 값이 유지되는지 확인한다.
결합하지 않은 원시 커널에도 동일한 검증을 적용한다.

같은 kernel 안에 있다는 사실만으로 전달이 보장되지는 않는다. Shared 값은 해당 block의
범위를 따라야 하고, Register 값의 thread 간 전달에는 이를 지원하는 구현이 필요하다.
Statement 자체는 kernel 경계가 아니다. 기존 Plan의 `CrossStatementStorage` 검사는
이 Emit 검증이 연결되면 교체한다.

## 실행 계획

결합된 본문을 실제 실행 단위에 배치하고, 임시 메모리·동기화·kernel 간 의존성과
launch 구성을 확정한다. 구현에 종속적인 자원 배치는 선택과 결합 결과를 반영한다.

### Streamed

일반 CUDA kernel의 grid에 작업을 배치하고, kernel 간 실행 순서로 의존성을 보장한다.
Native 본문과 Opaque의 launch 계약을 실행 계획에 연결한다.

### Persistent

Native 본문을 Worker CTA에서 실행하도록 작업과 인자, 의존성을 구성한다.
작업 dispatch와 완료 처리는 Persistent 실행 방식의 책임으로 둔다.

## CudaSource

[`CudaSource`](../../src/compile/source.rs)는 생성된 CUDA 코드와 실행에 필요한 자원
요구사항을 담는 현재 컴파일 입력이다.

```rust
pub struct CudaSource {
    code: String,
    requirements: CudaRequirements,
}
```

### `code`

확정된 본문과 실행 계획에서 생성한 CUDA 소스다.
코드 생성 단계에서 구현을 다시 선택하거나 결합 가능성을 다시 판단하지 않는다.

### `requirements`

Target과 rank 수, 버퍼 binding, workspace, shared memory, thread 구성과
통신 기능 등 컴파일과 실행에 필요한 조건을 전달한다.

현재 `CudaSource`는 CUDA 소스와 자원 요구사항을 담는다. Opaque artifact까지 전달하는
반환·compile·runtime 계약은 실제 외부 kernel 연동 단계에서 별도로 확정한다.

## 처리 경로

![여러 KernelProvider를 통한 원시 커널 선택과 본문 결합, CudaStreamed·CudaPersistent 실행 배치 및 다른 플랫폼의 확장 지점을 보여주는 Emit 처리 경로](../images/trinity-lowering-emission.svg)

공통 정보 준비에서는 BufferBinding과 값의 dtype·shape·storage, Loop 문맥과
실행 순서를 구성한다. 구현에 종속적인 임시 메모리와 최종 자원 배치는 이 시점에 고정하지 않는다.

각 단계의 구체적인 함수와 입출력 타입은 구현 시 정한다. 실행 배치 이후의 코드 생성과
출력은 플랫폼별로 분리한다. CUDA 경로는 CUDA render를 거쳐 `CudaSource`를 생성한다.
점선은 AMD 등 다른 플랫폼의 실행 배치·render·출력을 추가할 확장 경로이며,
해당 플랫폼의 구체적인 코드 생성과 출력 계약은 후속 설계에서 정한다.

## 검증 책임

| 구분 | 검증하는 내용 |
| --- | --- |
| 상위 입력 생성 단계 | Loop 범위와 메모리 접근의 유효성 |
| Plan | 참조, binding과 실행 순서 등 내부 일관성 |
| 원시 커널 선택 | 구현별 dtype·정렬·타일·실행 방식 지원 조건 |
| 커널 결합 | Operation 누락·중복, 값의 전달·수명, 초기화·갱신·최종 저장, 결합 가능성 |
| 실행 계획 확정 | 실제 배치에서의 자원 제약, 동기화, kernel 간 의존성과 launch 구성 |

## 다음 단계

구현은 Native pointwise 하나의 Streamed 실행, 연속 pointwise 결합,
GEMM의 순차 Loop와 전후 처리, 동일 Native 본문의 Persistent 실행 순서로 확장한다.
Opaque는 반환 형태와 선택 조건을 먼저 정의하고 실제 외부 kernel 연동은 이후에 진행한다.
세부 구현 순서는 [Emit 재구성 계획](../../../../docs/draft/PLAN.md#emit-재구성-단계별-계획--2026-09-16)을 참고한다.

생성한 `CudaSource`는 [`compile`](../../src/compile/mod.rs) 단계에 전달한다.

[기존 lowering 설명](trinity-lowering.md)은 이전 emitter의 구현을 포함한다.
[프로젝트 아키텍처](../../../../docs/draft/ARCHITECTURE.md)의 Plan 내 구현 선택·소유 설명은
이 초안과 차이가 있어 Emit으로 옮기는 개정을 제안한다. 상위 아키텍처 문서는 이 초안에서 수정하지 않는다.
