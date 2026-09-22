# Planning: 공통 IR 정보와 PhysicalPlan

2026-09-22 기준 공통 plan과 storage 분석의 구현을 설명한다.
계약 검증은 `tests/common_storage.rs`, `tests/plan_access.rs`, `tests/logical_dtype.rs`에서 확인할 수 있다.
이 문서는 공통 정보의 원본이며, Triton 구현 경로는 [provider](triton-provider.md),
수치 구현은 [precision](triton-precision.md), 후보·조합은 [emission](emission.md)이 원본이다.

[Architecture 안내](README.md)에서 전체 단계와 관련 소스를 찾을 수 있다.

## 분석 파일의 배치

- `analysis/ir/`: 공통 parser와 ScheduledIr 모델, source 수집 및 PhysicalPlan projection.
- `analysis/facts/`: 기존 접근·loop·scalar·metadata·dependency·flow·dtype 분석과 `ProgramFacts`.
- `analysis/storage/`: 공통 facts에 근거한 storage/초기화/publication 계약.
- `analysis/plan/`: 기존 PhysicalPlan 표현·builder·reader·normalization.

`facts/`는 기존 분석 파일을 함께 묶은 것이다. 새로운 분석 단계나 정책을 추가하지 않았다.
기존 공개 `analysis::*` API는 `analysis/mod.rs`의 re-export를 통해 유지한다.

## 입력 경로

| 진입점 | 입력과 결과 | 현재 차이 |
| --- | --- | --- |
| `PhysicalPlanBuilder::from_scheduled` | `ScheduledIr` + `ScheduledConfig` → `PhysicalPlan` | 선택된 source kernel을 `Region`으로 유지. view/symbol, 복수 출력, 입력 갱신, 초기 recurrence seed를 보존. single GPU |
| `lower_ir(text, &IrConfig)` | explicit 텍스트 → `Vec<PhysicalPlan>` (현재 한 plan) | 공통 storage 추론을 사용하지만 별도 Reader/Builder 경로. 한 출력 경로이며 모든 extended-source 기능과 동등하다고 보장하지 않음 |
| 직접 `PhysicalPlanBuilder` | dtype/shape/storage와 operation/statement 직접 등록 → plan | storage와 명시 dtype은 호출자가 지정. 공개 `build`는 출력 하나를 받음; 내부 `build_program`은 공개 다중 출력 builder API가 아님 |

구현: [scheduled](../../src/analysis/plan/scheduled.rs), [explicit reader](../../src/analysis/plan/ir.rs),
[builder](../../src/analysis/plan/builder.rs). `lower_ir`도 `analysis::storage::infer_source`를
사용한다. 예전의 모든 intermediate를 Global로 두는 분류는 현재 경로가 아니다.
`from_scheduled`는 제공된 shape/symbol binding으로 공통 facts를 확정한다.
Triton의 `lower_source` 편의 진입점은 그 전에 provider의 기본 tile 값을 정해 전달한다.
공통 builder 자체가 backend 성능 후보나 최적 tile을 선택하는 것은 아니다.

텍스트 syntax는 [analysis/ir/syntax.rs](../../src/analysis/ir/syntax.rs)의 `IrNode::parse` 한 곳에서 처리한다.
explicit `lower_ir`는 한 번 파싱한 AST를 공통 수집과 Reader가 함께 사용한다.
Python Builder의 expression/index 문자열도 같은 parser를 사용한다. 주석, source span,
중첩 제한과 syntax 오류 위치가 공유되지만, 각 Reader의 지원 연산·arity·의미 해석은 유지한다.
Parser 통합이 explicit Reader의 extended IR 지원 범위를 넓힌다는 뜻은 아니다.

### Logical dtype 확정

[analysis/facts/dtype.rs](../../src/analysis/facts/dtype.rs)의 `analysis::dtype::resolve`가
scheduled source의 logical/storage dtype을 확정하고 `from_scheduled`가 각
`ValueInstance::dtype()`에 기록한다. 입력/출력 기본값과 명시 override가 전파의 기준이며,
미명시 중간값은 operand type을 따라 fixed point로 계산한다. scalar-only producer는
기준 operand가 없으면 default를 받은 뒤 소비자까지 다시 전파한다. recurrence를 처음부터
FP16으로 가정해 BF16과 잘못 섞지 않는다. exp/reduction의 FP32 계산은 storage 승격 근거가 아니다.

`dtype_is_explicit() == false`는 **추론된 값이라는 provenance**다. 미해결 dtype 표식이 아니다.
모든 provider는 이미 확정된 `dtype()`를 받는다. 직접 Builder/explicit Reader의 명시 계약도
계속 유지한다. FP32 opmath, GEMM operand 변환과 accumulator, cast 위치는 provider의
구현 책임이며 [precision 경계](triton-precision.md#provider-integration-boundary)를 따른다.

## PhysicalPlan이 소유하는 정보

[정확한 타입과 getter](../../src/analysis/plan/mod.rs)가 API 원본이다.

| 정보 | 타입/API | 의미 |
| --- | --- | --- |
| 환경 | `target()`, `world_size()` | 대상 capability, rank 수. provider 선택 결과는 없음 |
| 경계 값 | `inputs()`, `outputs()`, `mutable_inputs()` | ordered ABI bindings, 복수 출력, 제자리 갱신되는 입력 |
| Tensor identity | `ValueInstanceId`, `value_instance(s)` | 한 allocation/value에 대한 식별자. 각 store의 SSA version을 따로 만드는 모델은 아님 |
| 값의 계약 | `ValueInstance::{dtype, dtype_is_explicit, shape, dimensions, storage, name}` | logical/storage dtype, 명시 여부, base shape 및 symbolic dimensions, backing storage, 이름 |
| 계산·효과 | `Operation::{expression, inflows, outflows, zero_init}` | typed expression, 읽기/쓰기 값, source에서 보존한 초기 zero recurrence seed |
| 실행 구조 | `statements()` | `Region`, `Loop`, `Operation`의 순서 있는 트리 |
| Binding | `bindings()`, `symbols()`, `bind_symbols()` | 구성 symbol과 allocation/access/loop의 일관된 specialization |
| 동등성 | `same_body()`, `hash()` | canonical body 비교와 process-local cache key; hash는 영구 artifact identity가 아님 |

`output()`은 첫 출력을 반환하는 호환 API이며 출력이 없으면 panic한다.
새 provider는 전체 프로그램에서 `outputs()`를 사용하고, 단일 출력 제약이 필요하면
지원 판정에서 명시한다. 이름은 binding/debug 용도이며 연결 판단은 ID·접근·scope로 한다.
Builder가 ID와 loop 이름을 canonicalize하므로 build 이전 ID를 최종 plan에 재사용하지 않는다.

### TensorAccess: 값의 shape와 접근 view를 분리

[TensorAccess](../../src/analysis/plan/expression.rs)는 다음을 가진다.

- `value`: `ValueInstanceId`.
- `view_shape`: 이 접근에서 바라보는 shape. 없으면 base `ValueInstance::shape()`.
- `view_dimensions`: specialization 전 parametric view expression.
- `indices`: view의 축 순서에 맞춘 `AccessIndex` 목록.

예를 들어 base `[B, H*D]`를 한 접근에서는 `[B, H, D]`로 볼 수 있다.
view는 동일 element 수의 논리적 row-major reinterpretation이며 새 allocation이나
자동 transpose/copy를 뜻하지 않는다. `indices`는 그 view에서 읽고 쓰는 부분을 고른다.
둘 중 하나로 다른 정보를 대체할 수 없다. 임의의 물리 stride/layout 정보가 이 구조체에
모두 들어 있는 것은 아니다. Triton은 runtime tensor stride와 view 좌표를 조합해 offset을
생성하고, Native/외부 provider는 자신이 지원하는 stride/layout을 별도로 검증한다.

`AccessIndex`는 `FullTile`, `Tile`, `ClippedTile`, `Elem`, `Slice`, `Element`를 구분한다.
`Elem`은 loop 좌표를 step으로 나눈 ordinal 접근이다. `Element`는 scalar index 식,
`Slice`는 산술 시작점과 접근 폭을 가진다.

`TileWidth`는 `TensorAccess`의 필드 자체가 아니라 일부 `AccessIndex` 안에 있는 폭 타입이다.
현재 `Constant(usize)` 또는 `Symbol(String)`이고, symbol은 configuration parameter다.
폭에 임의의 문자열 수식을 넣는 계약은 없다. `IndexExpr`의 산술 loop/index 식과 구분한다.
Loop step(이동량)과 tile width(접근 폭)도 독립적이다.

### Statement와 연산

[Statement/Loop](../../src/analysis/plan/statement.rs)는 아래 구조를 유지한다.

- `Region(body)`: scheduled source의 kernel 영역. Triton program provider에 전달할 때
  내부 store와 sequential loop를 독립 host-call 후보로 임의 분해하지 않는다.
- `Loop { kind, domain, body }`: `Parallel`, `Sequential`, 정규화된 mloop의 `Split`;
  domain은 변수와 `start/stop/step`의 `IndexExpr`를 가진다.
- `Operation(id)`: plan이 소유하는 expression의 효과 실행.

`Expression::Store`의 계산 자식은 pure expression이다. 산술·matmul·reduction·broadcast와
확장 `ValueOp`가 원래 순서대로 보존된다. 공통 plan이 명령이나 template을 선택하지 않는다.
Builder의 `AllGather`는 통신 의미를 표현할 수 있지만, 이것이 현재 Triton/native emit의
통신 실행 지원을 의미하지 않는다. Native 새 emit의 Persistent/통신 연결은 미구현이다.

## 공통 분석과 storage

전체 facts를 PhysicalPlan 필드에 복제해 저장하지 않는다. Plan은 의미를 소유하고,
[analysis::from_physical](../../src/analysis/ir/physical.rs)이 그것을 occurrence/scope 표로
투영한다. source 문자열을 직렬화·재파싱하지 않으며, 이 `ScheduledIr::ir()`은 `None`이다.
분석을 위해 원본 텍스트를 provider에 별도로 넘길 필요는 없다.

| 공통 API | 제공 정보 |
| --- | --- |
| `analysis::analyze_text` / `ScheduledIr` | 원본 scope, statement, tensor, 각 read/write의 `AccessId`, loop/index 식 |
| `TensorMetadata::collect`, `ProgramFacts::analyze/resolve` | shape·symbol metadata, resolved logical accesses, kernel별 dataflow |
| `TensorDataflow` / `KernelDataflow` | ordered reads/writes, `live_out`, `entry_value`, common scope, additive updates, 이전 write 목록, loop dependencies |
| `analysis::storage::infer` / `infer_source` | source의 backing storage, local read binding, 초기화·publication 위치 |
| `analysis::storage::for_values` | 이미 선택된 PhysicalPlan storage 계약을 유지·검증하며 per-region access plan 생성 |

각 분석의 ID는 해당 `ScheduledIr`에 속한다. 서로 다른 projection/plan의 ID를 숫자가
같다는 이유로 섞지 않는다. 입력 binding과 metadata도 같은 projection에서 만들어야 한다.

### Backing storage와 kernel 내부 사용

| `ValueInstance.storage` | 의미 |
| --- | --- |
| `External` | 입력/출력 ABI 값. 제자리 갱신되는 입력도 포함. 메모리가 register라는 뜻은 아님 |
| `Global` | materialization이 필요한 내부 device buffer. 영역 간 전달, 접근 방식 불일치 등 |
| `Register` | 공통 분석에서 local tile로 전달할 수 있는 내부 값. 실제 GPU register spill 여부와는 별개 |
| `Shared` | 타입에는 존재하지만 명시 Shared transport는 현재 공통 storage 소비/Triton 경로에서 거절. layout·동기화 계약의 후속 연결 필요 |

`StoragePlan.values`는 프로그램의 backing storage다. 별도로
`KernelStoragePlan::{tensors, register_accesses, local_reads}`가 영역 내부 사용을 표현한다.
`TensorStoragePlan.storage`의 `AccessMode::{Register, Global, Materialized}`는 위 enum과 다르다.
External/Global 출력도 한 kernel 안에서는 register accumulator로 누적한 뒤 publish할 수 있다.
`Materialized`는 local accumulator 접근과 memory 접근이 공존하는 경우이지 새 allocation 종류가 아니다.

`TensorStoragePlan`에는 representative access, initialization(scope/access/Zero 또는 Global),
export scope, publish 여부, additive-update statement 집합이 있다.
`LocalRead`는 producer access와 대응 축을 기록한다. 이미 common에서 결정된 local 연결을
provider가 tensor 이름이나 동일 shape만으로 다시 추측하지 않는다.
명시 Global은 local 계산이 가능해도 publication을 유지하고, 명시 Register가 materialization을
요구하면 거절한다. 실패를 피하기 위한 자동 spill은 하지 않는다.

### 초기화·수명 분석의 범위

Scheduled 경로의 `EntryValue`는 input, 영역 안의 이전 정의, 이전 kernel, 최초 zero recurrence,
producer 누락을 구분한다. `additive_updates`는 중첩 덧셈·scaled self term도 탐색하지만
그 자체가 zero 초기화의 근거는 아니다. `from_scheduled`는 최초 zero recurrence만
`Operation::zero_init`에 기록하고 기존 producer를 보존한다.

Region 밖의 직접 Builder/explicit reader에는 단일 본문의 sequential reduction을
zero-start로 해석하는 기존 규약도 남아 있다. Scheduled seed 보존과 이 규약이 모든 경우에
동일하다고 가정하지 않는다. [normalize](../../src/analysis/plan/normalize.rs)와
[recurrence 테스트](../../tests/triton_recurrence.rs)를 함께 확인한다.

현재 dependency 분석은 ordered access와 제한된 coverage/scope 증명이다.
완전한 SSA value-version graph, arbitrary alias analysis, 전체 buffer lifetime 재사용 allocator,
모든 symbolic 부등식 증명을 제공하지 않는다. 증명할 수 없는 local/global 접근은 거절할 수 있다.
Triton이 loop 밖 SSA 사용을 위해 추가하는 incoming-value 초기화나 lane 간 barrier 위치는
별도의 backend lowering 결정이며 common의 의미상 초기화와 구분한다.

## Provider가 추가해야 하는 정보

| 공통에서 받는 것 | Provider가 결정·검증할 것 |
| --- | --- |
| 원래 expression·region·loop·접근 view/indices | 맡을 operation/region 범위, 지원 여부, 실행 순서 보존 |
| 확정된 logical/storage dtype와 backing storage | register 계산 타입, cast 위치, 실제 operand/accumulator type; [수치 경계](triton-precision.md#provider-integration-boundary) |
| Local producer/read 및 publication | thread-to-element mapping, layout 호환, barrier/pipeline |
| Symbolic tile와 target | 허용 tile·instruction·resource 조건, binding/config/launch |
| ABI 입출력과 mutation | library 호출 인자, stride/alignment, workspace, output alias 제한 |

Quack은 [KernelRequest](../../src/emit/request.rs)의 제한된 whole-tensor 경로를 사용하며,
CuTe는 내부 `KernelContext`와 Native 명세/조합을 사용한다. 이들은 전체 scheduled fallback과
동일한 지원 범위가 아니다. [provider의 현재 경로와 제한](triton-provider.md#selection-boundary-and-current-limits)을 확인한다.

## 아직 확정되지 않은 확장

- 여러 operation/loop/region을 하나의 GEMM+epilogue candidate로 묶는 coverage 계약.
- Native와 Triton의 register 경계 rounding 차이에 대한 공통 수치 비교 기준.
- Shared 전달, 일반 alias/value-version 분석, Native/opaque의 통합 compile·benchmark·선택.

위 항목은 기존 공통 API의 제공 기능으로 간주하지 않는다. 새 provider 작업에서 필요한
범위를 producer와 consumer 양쪽 테스트와 함께 확정한다.
