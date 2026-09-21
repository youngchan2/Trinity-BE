# Planning

Text IR 또는 `PhysicalPlanBuilder` 입력을 받아, Tensor Binding, 연산, Loop와 실행 순서를 `PhysicalPlan`으로 구성합니다.

## PhysicalPlan

[`PhysicalPlan`](../../src/plan/mod.rs)은 입력으로 전달된 Tensor, 연산, Loop 구조와
실행 순서를 표현하는 자료구조입니다.

입력의 연산과 실행 구조를 유지하며, Lowering에 필요한 정보를 수집하여 보관합니다.

```rust
pub struct PhysicalPlan {
    /// `PhysicalPlan`의 하드웨어 사양을 표현
    target: TargetCapability,
    /// 프로그램 실행에 참여하는 rank 수를 표현
    world_size: usize,

    /// 외부 입력 tensor 이름과 Plan 내부 값의 binding
    inputs: Box<[TensorBinding]>,
    /// 외부 출력 tensor 이름과 Plan 내부 값의 binding
    output: TensorBinding,

    value_instances: IdVec<ValueInstanceId, ValueInstance>,
    operations: IdVec<OperationId, Operation>,
    statements: Vec<Statement>,

    /// 컴파일 cache key로 사용
    hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorBinding {
    tensor: String,
    value: ValueInstanceId,
}
```

![](../images/trinity-lowering-physical-plan-model.svg)

### `inputs`, `output`

프로그램 실행에 필요한 외부 입력 Tensor의 목록입니다.

예를 들어 `Y = A + B`의 프로그램을 작성할 때, `inputs`에 `A`와 `B`, `output`에 `Y` Tensor가 할당됩니다.

각 요소는 `TensorBinding`으로 구성되어 입력 Tensor의 이름(`tensor`)과 내부 참조 ID(`value`)를 보관합니다.

### `value_instances`

프로그램의 Tensor 정보를 `ValueInstance` 구조로 정리해 보관합니다. 해당 값은 `ValueInstanceId` key를 이용해 조회 및 수정, 추가합니다.

```rust
pub struct ValueInstance {
    dtype: DType,
    shape: Box<[usize]>,
    /// Tensor 저장 위치
    storage: Storage,
    name: Option<String>,
}
```

`storage`는 Tensor의 물리적 저장 위치를 다음 값 중 하나로 표현합니다.

| 값         | 의미                                 |
| ---------- | ------------------------------------ |
| `External` | ABI binding으로 연결된 입출력 Tensor |
| `Global`   | Device global memory                 |
| `Shared`   | CTA body 내 shared-memory tile/panel |
| `Register` | CTA body 내 register 값              |

- 다른 스케줄에 위치한 Task 사이 Tensor는 `External` 또는 `Global`에 위치해야 합니다.
- `Shared/Register` 값의 `shape`는 기존 rank-local shape를 유지해야 합니다.

### `operations`

`Operation`은 `PhysicalPlan` 내 메모리 상태 변경을 표현하는 최소 단위로, 해당 변경에 필요한 읽기 및 쓰기, 연산이 이에 속합니다.

```rust
pub struct Operation {
    // Tensor Read
    inflows: Vec<ValueInstanceId>,
    // Tensor Write
    outflows: Vec<ValueInstanceId>,
    /// 계산, 통신 또는 메모리 접근 표현식
    expression: Expression,
}
```

### `statements`

`statements`는 프로그램의 제어 구조를 표현합니다. 각 [`Statement`](../../src/plan/statement.rs)는 하나의 `Operation`을 표현하거나 `Loop`의 `body`에 중첩된 `statement`를 포함합니다.

`Loop`의 `kind`는 병렬 또는 순차 반복을 구분합니다. `Loop` 내 `domain`은 반복 변수와 `start`, `stop`, `step` 값을 나타냅니다. `body`는 Loop 내 중첩된 `Statement`를 보관합니다.

Emit은 하드웨어와 `Operation`·Loop을 고려해 구현을 선택하고, 이 제어 구조를 바탕으로 실행 배치를 정합니다.

## 입력 경로

![](../images/trinity-lowering-physical-plan.svg)

[`lower_ir()`](../../src/plan/ir.rs)는 텍스트를 파싱하고 tensor·symbol을 연결한 뒤,
명시된 Loop와 expression을 [`PhysicalPlanBuilder`](../../src/plan/builder.rs)에 삽입합니다.

또한, 직접 `PhysicalPlanBuilder`에 값을 직접 전달하여 `PhysicalPlan`을 구성할 수 있습니다.

## 다음 단계

구성된 Plan은 [Emit 파이프라인](emission.md#처리-경로)에 전달합니다.
