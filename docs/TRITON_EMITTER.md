# Rust Triton fallback emitter

이 문서는 현재 `feat/triton`에 구현된 Rust emitter를 설명한다.
입력은 optimizer가 선택한 scheduled Trinity IR이다. Backend는 입력의 계산 순서와
loop 구조, kernel fusion boundary를 유지하면서 Triton Python 소스를 생성한다.
현재 유지하는 통합 테스트 입력은 postprocessed batched MLA stage14·16·20이다.

Rust emitter는 라이브러리 API로 호출한다. Optimizer의 native IR·metadata를 직접
전달하는 adapter와 CUDA/외부 라이브러리 구현 선택 단계는 아직 연결되어 있지 않다.
저장소의 `src/physical/`, `src/implementation/`은 현재 Triton 경로와 별도이다.

## 1. 입력에서 Python 소스까지

```text
S-expression text
    ↓ analysis::analyze_text
ProgramAnalysis
  원본 구문, tensor/access/statement ID, lexical scope, view/index 정보
    + Options { shapes, symbols, managed, tuning }
    ↓ triton::lower
ProgramPlan
  공통 metadata, 접근별 TileAccess, 계산식 Expr
  kernel별 KernelPlan, tensor별 TensorPlan, LocalRead
    ↓ ProgramPlan::emit
Python source String
  kernel_0, kernel_1, ... + autotune + forward(...)
```

공개 API는 [analysis/mod.rs](../src/analysis/mod.rs)와
[triton/mod.rs](../src/triton/mod.rs)에 있다.

| API | 역할 |
|---|---|
| `analysis::analyze_text(text)` | 문자열을 파싱하고 `ProgramAnalysis` 생성 |
| `analysis::analyze(node)` | 이미 파싱된 `IrNode`의 소유권을 받아 분석 |
| `triton::lower(analysis, options)` | 분석 결과를 `ProgramPlan`으로 lowering |
| `ProgramPlan::emit()` | 계획에서 Python 소스 문자열 생성 |
| `triton::compile(text, options)` | 파싱·분석·lowering·emit을 한 번에 수행 |
| `triton::compile_analysis(analysis, options)` | 기존 분석 결과에서 lowering·emit 수행 |

`compile`은 소스를 반환하며 파일 저장이나 Triton JIT 실행을 수행하지 않는다.
파일 입출력은 현재 [MLA emit 테스트](../tests/batched_mla_emit.rs)가 담당한다.

## 2. 파일별 책임

```text
src/
  analysis/
    mod.rs                  공개 분석 API
    ir.rs                   IR 구문, parser, source span
    collect.rs              tensor/access/statement/scope 수집
    model.rs                ProgramAnalysis, ID와 분석 결과 타입
    dependencies.rs         IR 기반 scope·loop 의존성과 영역 비교
  triton/
    mod.rs                  공개 emit API, Options, Error
    plan.rs                 ProgramPlan / KernelPlan / TensorPlan / Expr
    shape.rs                shape, tile, index와 정수식 계산
    lowering/
      mod.rs                metadata → access/expr → kernel 계획 조립
      dependencies.rs       ownership, coverage, materialized read 검사
      storage.rs            storage, 초기화, export, recurrence, register binding
      expression.rs         연산별 shape 검증과 Expr 생성
      loops.rs              loop bound와 lexical binding 검증
      access.rs             view 축 대응, local sub-tile, producer coverage
      metadata.rs           shape symbol, split alias, 후보, Python 이름 결정
    codegen/
      mod.rs                전체 Python 모듈 조립
      context.rs            CodegenContext / EmittedValue
      indexing.rs           좌표, offset, mask와 global load/store
      ops.rs                scalar, pointwise, reduction, dot, transform
      kernel.rs             signature, loop, 초기화와 export 배치
      wrapper.rs            launcher 모드 분기와 기존 benchmark ABI
      scalar.rs             shape/bound/grid의 공통 parameter 표기
      local.rs              local definition의 reshape/gather와 validity
      launch.rs             managed autotune, 할당과 kernel 호출
```

`analysis`는 IR의 사실을 보관한다. `triton/lowering`은 구체적인 shape와 실행
조건을 사용해 저장·초기화·scope 등을 결정한다. `triton/codegen`은 이 정보를
사용해 주소식, 계산식과 Python 소스를 작성한다. 모든 주소가 미리 문자열로
저장되는 구조는 아니며, offset과 mask 표현은 codegen에서 조립한다.

## 3. 분석 결과와 정보 보관

[ProgramAnalysis](../src/analysis/model.rs)는 tensor 이름으로 모든 접근을
한꺼번에 분류하지 않고 `TensorId`, `ScopeId`, `StatementId`, `AccessId`를
사용한다. 동일 tensor의 서로 다른 load/store를 구별하고, 같은 이름의 loop 변수도
가장 가까운 lexical binding으로 해석한다.

분석은 store의 RHS read를 먼저 수집한 뒤 해당 write를 기록한다. 접근마다 index,
view의 축 이름·shape, statement와 scope를 보관한다.
`analysis/dependencies.rs`의 tensor별 loop 의존 집합은 공통 조회 정보이며,
전체 프로그램의 일반적인 memory SSA나 alias 증명을 제공하는 것은 아니다.

[ProgramPlan](../src/triton/plan.rs)은 한 IR 후보 전체의 분석 결과와 options,
kernel 목록을 소유한다.

| 정보 | 저장하는 내용 |
|---|---|
| `TileAccess / AxisAccess` | 접근별 tile shape, 시작점, 논리 너비, 축 크기, stride 계산 정보, loop end |
| `TensorPlan` | storage 분류, 대표 접근, 초기화 값·scope, export scope, publish 여부, 누적 statement |
| `KernelPlan` | root scope, parallel loops, sample grid, 실제 grid 식, tensor 계획 |
| `register_accesses` | global load/store 대신 지역 변수로 처리할 접근 |
| `local_reads` | read가 참조하는 앞선 write의 AccessId와 view 축 대응 |
| `ProgramMetadata` | Python 이름, tensor shape 식, 입력 shape와 symbol 연결, alias, split controller, 튜닝 후보 |

`Initialization`은 zero 초기화와 global load 초기화를 구분한다. 누적값의 reset과
최종 저장은 `scope` 및 `export_scope`에 따라 배치한다. 같은 tensor라도 누적
scope 내부에서는 register를 사용하고, 뒤의 소비자는 global에서 읽을 수 있다.

`KernelPlan.grid`는 입력한 sample shape·symbol 값으로 계산한 값이고, 실제
launcher는 `grid_extents`를 shape 및 튜닝 parameter로 평가한다.

`CodegenContext`는 indentation, 임시 변수 번호와 load cache 같은 출력 상태를
소유한다. `EmittedValue`는 생성된 식, tile shape, 축별 validity와 zero-fill 정보를
전달한다. 완성된 `ProgramPlan` 자체는 emit 도중 수정하지 않는다.

## 4. View, slot과 주소 계산

[collect.rs](../src/analysis/collect.rs)는 `view`의 base tensor를 storage identity로
사용하고, `layout`에 나열된 축 순서로 `keyed_index`를 정렬한다.
`slot`은 축 이름으로 연결하므로 slot의 소스 순서가 주소의 축 순서를 바꾸지 않는다.

예를 들어 layout이 `[b, p, m]`이고 slot 순서가 `[b, m, p]`여도 주소는 다음과 같다.

```text
offset = coordinate_b * stride_b
       + coordinate_p * stride_p
       + coordinate_m * stride_m
```

[indexing.rs](../src/triton/codegen/indexing.rs)의 load와 store는 같은 주소 계산
함수를 사용한다. 축 좌표는 다음 규칙으로 작성한다.

- `tile(start, width)`: `start + arange(width)`.
- `elem(loopvar)`: IR의 의미에 따라 `loopvar // loop_step`, 길이 1의 축 유지.
- `fulltile`: view 축 전체의 좌표.
- `const_tile(start, width)`: 지정된 시작점과 너비의 좌표.

Triton tile 크기에 맞춘 padding은 논리적인 접근 너비와 구분한다.
Mask에는 축의 유효 범위, 필요한 tile padding 조건과 loop/chunk end를 반영한다.
Launch에서는 kernel 인자의 실제 tensor stride를 전달한다.

Singleton 축을 삽입·제거하는 view는 대응 축의 stride를 사용한다.
같은 kernel에서 한 tensor를 서로 다른 shape로 접근하면 필요한 경우 logical
row-major offset을 대표 view의 축 좌표로 분해한 뒤 실제 stride에 연결한다.
Host-side `.view`는 PyTorch가 허용하는 storage-compatible 변환을 사용한다.

Register에 보관된 값은 [local.rs](../src/triton/codegen/local.rs)가 처리한다.
인식된 view 또는 부분 tile 관계에 따라 reshape/gather를 생성한다.
Stage20의 `__fission_t2` cell read는 정규화한 두 번째 store에 연결되고,
`cell - q_h_q`에 해당하는 local 좌표로 한 head를 읽는다.

## 5. Loop 처리

`ploop`는 grid의 program ID에 대응하며, `sloop`는 kernel 내부의 `range`로
출력된다. 예를 들어 다음 IR 범위는 외부 head tile의 시작점을 사용한다.

```text
sloop q_h_q (+ q_h_q tile_q_h_q) 1 cell
    ↓
range(q_h_q, q_h_q + META_tile_q_h_q, 1)
```

`mloop(start, stop, step, n, m, ns, body)`는 분석 단계에서 split 변수 `m`을
바인딩하는 `SplitLoop`와 serial 변수 `n`을 바인딩하는 자식 scope로 표현한다.
원본 구문은 별도로 보관한다. Split 축은 grid 한 축을 차지한다.

```text
chunk = (stop - start) // ns
n: start + m * chunk .. start + (m + 1) * chunk, step
```

Additive recurrence는 serial loop 전에 FP32 zero로 초기화하고, 누적 후
split scratch에 FP16으로 저장한다. 후속 reduction은 IR에 표현된 순서를 따른다.
유효한 profile 후보는 `ns == 1` 또는 `(stop-start) % (ns*step) == 0`을 만족한다.

현재 parallel 축은 split을 포함해 최대 3개이며 하나의 enclosing chain을 따라야 한다.
Parallel grid의 start/end는 프로그램 전체에서 평가할 scalar parameter여야 한다.
Serial bound에는 외부 loop 변수를 사용할 수 있다. Scalar 범위식은
`+`, `-`, `*`, 정수 나눗셈을 지원하며 divisor는 검증 가능한 양수여야 한다.
Tensor 데이터에 의존하는 loop bound는 지원하지 않는다.

## 6. Managed launcher, autotune과 precision

[Options](../src/triton/mod.rs)는 다음 정보를 받는다.

| 필드 | 역할 |
|---|---|
| `shapes` | 분석·검증에 사용할 구체적인 tensor shape |
| `symbols` | tile 등 IR scalar의 sample 값 |
| `managed` | 내부 tensor 할당과 output 반환을 포함한 launcher 사용 |
| `tuning` | IR symbol별 명시적인 튜닝 후보 |

MLA 테스트는 `managed = true`를 사용하며, `mloop`가 있으면 이 모드는 자동으로
활성화된다. 입력 view의 symbolic dimension을 실제 입력의 shape에 연결하고,
내부 tensor shape는 view에서 추론한다. `forward`는 필요한 global intermediate를
할당하고 output을 반환한다. Input/output의 FP16 dtype과 shape 등을 검사하고
실제 stride를 kernel에 전달한다.

일반 tile symbol은 명시한 값을 기본 후보로 사용한다.
지정하지 않은 split count는 1/2/4/8 후보를 사용한다.
같은 scratch에서 연결되는 split symbol은 alias로 묶고, 충돌하는 명시적 값은 거부한다.
Launcher는 legal 후보의 최대 split count를 기준으로 scratch를 할당한 뒤,
선택된 count를 후속 reduction에 전달한다. 할당 capacity와 유효 reduction 범위는
별도이다. 각 kernel의 autotune key에는 필요한 shape/split parameter와 stride가 들어간다.

현재 dtype 정책은 global FP16, 일반 계산·reduction·누적 FP32이다.
Managed 모드의 일반 tensor store는 register에 남아도 FP16 rounding을 적용한다.
Additive recurrence는 FP32로 유지하다가 export 시 FP16으로 변환한다.
Tensor-core dot 입력은 FP16이고, 작은 dot의 multiply/reduce 경로는 managed 모드에서
FP32를 사용한다. 명시적인 `cast`는 IR 위치에 반영한다.

Validity는 축별 predicate로 전달한다. `tl.load(..., other=0)`의 zero-fill을
재사용하며, `exp` 이후 reduction처럼 무효 lane이 결과를 바꿀 때는 필요한
`tl.where`를 생성한다. `concat`은 padding을 제외한 논리 길이로 이어 붙인다.

`managed = false`인 기존 caller-owned tensor launcher도 코드에는 남아 있다.
현재 MLA 통합 테스트는 managed 경로를 대상으로 한다.

## 7. 현재 유지하는 테스트

저장소 루트에서 실행한다.

```shell
cargo test --locked --test batched_mla_emit -- --nocapture
```

[batched_mla_emit.rs](../tests/batched_mla_emit.rs)는
[fixtures/batched_mla](../tests/fixtures/batched_mla)의 IR과 `shapes.txt`를 읽어
`triton::compile`을 호출하고 다음 파일을 생성한다.

| 입력 | 출력 | 기대 kernel 수 |
|---|---|---|
| `batched_mla_postprocessed_stage14.txt` | `target/tests/batched_mla/stage14.py` | 3 |
| `batched_mla_postprocessed_stage16.txt` | `target/tests/batched_mla/stage16.py` | 2 |
| `batched_mla_postprocessed_stage20.txt` | `target/tests/batched_mla/stage20.py` | 3 |

입력 sample shape는 Q=[2,128,1,128], CKV_cache=[2,64,512],
W_DK=W_DV=[512,16384]이다. Tile 값은 `tile_b=1`, `tile_m=32`,
`tile_q_h_q=16`, `tile_p=16`이다.

각 테스트는 생성 파일의 Python 구문, kernel 함수 이름·개수, `forward` 함수
존재를 검사한다. Python 3만 사용하며 PyTorch/Triton을 import하거나 GPU를 실행하지
않는다. 따라서 이 테스트의 통과는 Triton JIT 컴파일, 주소·수치 정확성 또는
성능 검증을 뜻하지 않는다. 모든 extracted IR 조합의 지원을 보장하는 테스트도 아니다.

`generated_kernels/`와 `reference_kernels/`는 Git에서 제외된 산출물이며
이 테스트에서 변경하지 않는다. 일반 개발 검사는 다음과 같다.

```shell
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```
