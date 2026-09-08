# Trinity Triton backend architecture plan

이 문서는 구현 전의 초기 제안 기록이다. 아래의 범용 ABI/KernelIR/PhysicalPlan 연동과
bf16/fp32 계획은 현재 initial emitter의 구현 범위를 뜻하지 않는다. 현재 구현은
fp16 기반이며, 실제 API·출력 규약·검증 결과는 [TRITON_FALLBACK.md](TRITON_FALLBACK.md)를 따른다.

작성일: 2026-09-05. 분석 기준: `Trinity` commit `e9b4468`, `trinity-lowering` commit `815ba98`와 현재 working tree. 구현 변경 없이 작성한 설계안이다. 요구사항은 [BACKEND_REFACTOR.md](/home/chani227/Project-Trinity/Trinity/BACKEND_REFACTOR.md)를 따른다. 아래에서 **현재**는 확인한 코드, **제안**은 앞으로 구현할 계약과 동작을 뜻한다.

## 1. 핵심 결론

**`trinity-lowering/`에 Rust backend를 구축하는 방향을 권장한다. optimizer와 공유할 것은 선택된 프로그램과 명시적인 metadata 계약이며, e-graph 자체나 `LoopData`를 backend 내부 상태로 공유하지 않는다.**

```text
Rust optimizer / extraction
  → SelectedProgram + ProgramAbi + OptimizationMetadata
  → Rust verification / normalization / KernelIR
  → dependency / fusion / scheduling / implementation selection
  → PhysicalPlan + owned kernel bodies
  → Rust Triton AST / source printer 또는 검증된 template renderer
  → generated Python module + launch manifest + source map
  → Python Triton compile / cache / launch
```

핵심 결정은 다음과 같다.

1. **별도 KernelIR을 둔다.** 현재 TileLang은 이미 tile·loop 수준이다. high-level tensor graph로 되돌리거나 Inductor 전체를 복제하지 않고, 이름 해석·타입·메모리 효과·loop-carried value가 명시된 작은 IR로 정규화한다.
2. **기존 PhysicalPlan을 실행 그래프 계층으로 확장한다.** KernelIR은 커널 내부 계산, PhysicalPlan은 action 사이의 값·물리 저장소·실행 순서를 담당한다. 두 표현은 책임이 다르다.
3. **optimizer의 선택과 backend의 구현 결정을 구분한다.** 선택된 loop 순서, tile 의미, split-K는 보존한다. backend는 그 프로그램의 구현 가능성, materialization, GPU program mapping, register tile, launch 설정을 결정한다.
4. **metadata는 semantic contract / analysis / tuning hint / provenance로 구분한다.** dtype·실제 stride·초기값처럼 correctness를 결정하는 정보를 비용 추정치와 같은 신뢰 수준으로 취급하지 않는다.
5. **범용 연산은 Rust programmatic emitter, 정형 GEMM 등은 typed template 경로로 생성한다.** 두 경로가 같은 signature·index·mask·precision·launch 계약을 사용한다.
6. **Python은 실행 경계에 남긴다.** Rust가 Triton Python 파일까지 생성하고 Python이 compile·allocation·launch·benchmark를 수행한다. Rust로 Triton compiler 자체를 다시 구현하지 않는다.

초기 범위는 single-GPU, 명시적 fp32/bf16 계약, 검증된 strided access, pointwise부터 시작한다. fp16은 기존 benchmark 이관 시 명시적으로 추가한다. 기존 AllGather/NVLS 구현은 유지하되 이번 Triton 경로의 지원 대상으로 간주하지 않는다.

## 2. 현재 코드 분석

### 2.1 실제 데이터 흐름과 파일별 책임

| 위치 / symbol | 현재 책임 | 설계에 미치는 영향 |
|---|---|---|
| [language.rs:120](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:120), `TileLang` | `egg::Language`로 전체 S-expression 연산 정의 | e-graph용 표현과 실행용 표현의 경계가 필요 |
| [language.rs:266](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:266), `LoopData`, `Access`, `LoopAnalysis` | shape, read/write access, rewrite 관련 cache | `analysis.rs`는 비어 있고 실제 analysis는 이 파일에 있음 |
| [shape.rs:6](/home/chani227/Project-Trinity/Trinity/optimizer/src/shape.rs:6), `Dimension`, `TensorShape`, `ShapeTracker` | concrete/wildcard shape, 이름별 shape table, 기본 tile 크기 | symbolic shape identity와 dtype/stride를 제공하지 않음 |
| [dependency.rs](/home/chani227/Project-Trinity/Trinity/optimizer/src/dependency.rs), `no_raw_dependency` 등 | rewrite 허용 여부를 위한 access 기반 dependency 판단 | 선택된 실행 프로그램의 scheduler 분석으로 그대로 대체할 수 없음 |
| [extract_preprocess/mod.rs:107](/home/chani227/Project-Trinity/Trinity/optimizer/src/extract_preprocess/mod.rs:107), `PreprocessOutput` | kind, iteration count, FLOPs/bytes, invalid-store, value 선택, reuse pair, tile assignment | export할 metadata의 주요 원천 |
| [extract_preprocess/emit.rs:48](/home/chani227/Project-Trinity/Trinity/optimizer/src/extract_preprocess/emit.rs:48), `write_all`, `StructuralView` | structural graph 재번호화, JSON/NPZ, `val_exprs.json`, `tile_sizes.json` | original/structural/selected ID를 명시적으로 연결해야 함 |
| [extract_preprocess/reconstruct.rs:76](/home/chani227/Project-Trinity/Trinity/optimizer/src/extract_preprocess/reconstruct.rs:76), `reconstruct` | 선택된 e-node를 `RecExpr`로 복원, cycle 거부 | 반환값이 `RecExpr`뿐이므로 provenance export API 추가 필요 |
| [python/reconstruct.py:119](/home/chani227/Project-Trinity/Trinity/optimizer/python/reconstruct.py:119), `reconstruct` | structural selection에 사전 선택된 `val_exprs`를 삽입 | Rust export도 이 값 선택을 보존해야 하며 다시 extractor를 돌리면 안 됨 |
| [postprocess.rs:77](/home/chani227/Project-Trinity/Trinity/optimizer/src/postprocess.rs:77), `postprocess`, `postprocess_v2` | value forwarding, DCE, loop classification; v2는 문자열 coherence 보정 | metadata를 붙인 뒤 문자열 재작성하면 ID 연결이 깨짐 |
| [convert_module.py:9](/home/chani227/Project-Trinity/Trinity/backend/codegen/convert_module.py:9), `convert_ir_to_triton` | text parse → view shape 수집 → desugar → source | dtype, alias, initialization, selected-node metadata를 받는 ABI가 없음 |
| [ViewDesugar.py:57](/home/chani227/Project-Trinity/Trinity/backend/codegen/ViewDesugar.py:57), `desugar` | named-axis→positional, variadic permute, mloop 변환 | 아이디어와 regression case는 재사용; 정보 손실 방식은 교체 |
| [triton_generator/ARCHITECTURE.md](/home/chani227/Project-Trinity/Trinity/backend/codegen/triton_generator/ARCHITECTURE.md) | 현재 구조가 AST 직접 문자열 생성임을 명시 | 파일 분리만으로 lowering/scheduling 경계가 생긴 상태는 아님 |
| [trinity-lowering/src/lib.rs:1](/home/chani227/Project-Trinity/trinity-lowering/src/lib.rs:1), `PhysicalPlanBuilder` export | compiler source graph를 소유하지 않는 물리 계획 라이브러리 | 이 독립성은 유지할 가치가 있음 |

현재 backend는 완전히 과거 operator 집합에 머물러 있지는 않다. [NodeType.py:10](/home/chani227/Project-Trinity/Trinity/backend/codegen/NodeType.py:10)에는 `mloop`, `view`, `keyed_index`, variadic `permute`가 있고 desugar도 존재한다. 문제의 중심은 **operator enum 누락만이 아니라 실행 의미와 metadata를 다시 추측하는 구조**다.

실제 operator 불일치도 있다. Rust의 `Const`와 달리 [Python parser의 operator map](/home/chani227/Project-Trinity/Trinity/backend/codegen/IrParser.py:99)에는 `const`가 없다. 따라서 전체 operator coverage test는 여전히 필요하다. 또한 [SHAPE_TRACKER](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:7)는 thread-local 상태이므로 이를 backend가 직접 조회하는 구조 대신 export 시 필요한 정보를 owned ABI/metadata로 복사한다.

### 2.2 TileLang의 표현력

| 검토 항목 | 현재 확인한 사실 |
|---|---|
| 전체 계층 | e-graph의 e-class 안에 여러 동등 e-node가 존재. extraction 후 child ID를 가진 `RecExpr` DAG. 별도 module/function/type/block 계층은 없음 |
| operation/value/type | 같은 `TileLang` enum에 statement, expression, tensor name, slice, axis, literal이 공존. `Num(i32)`, `Var(Symbol)`, `Cast([dtype, value])`; 모든 값에 붙는 dtype 없음 |
| tensor/buffer | `Input`, `Output`, `Tensor`는 이름 wrapper. allocation 크기, storage offset, alignment, ownership/lifetime을 나타내는 buffer descriptor 없음 |
| shape/stride/layout | `view(T, layout(axis(name,size)...))`는 named-axis 주소 해석. 물리 stride·GPU thread layout은 표현하지 않음. `TensorShape`는 `Concrete`/`Wildcard`뿐 |
| SSA/use-def | expression child 연결은 있으나 일반적인 SSA 프로그램은 아님. `Load/Store`로 같은 storage를 여러 번 갱신. memory version·dominance·phi/iter_arg 없음 |
| region/block/control flow | `Seq([a,b])`와 body child를 가진 `Loop/DLoop/PLoop/SLoop/MLoop`. arbitrary CFG, branch region, terminator 없음 |
| pointwise | `+ - * / <= max min exp sqr sqrt sigmoid erf abs cast` |
| reduction | `rsum/rmin/rmax(value, axis)`와 별개로 `store T (+ (load T idx) rhs) idx` 형태의 누적. reducer identity/init/precision/order는 별도 명시가 부족 |
| broadcast | `bcast`는 wildcard axis 삽입, `unsqueeze`는 크기 1 axis 삽입. 일반적인 NumPy식 implicit broadcasting을 전제하면 안 됨 |
| transforms | `transpose`, `permute3/4/variadic`, `squeeze`, `unsqueeze`, `concat`. `reshape` 전용 variant는 없으며 재인덱싱 loop 등으로 표현 |
| memory/address space | `Load/Store`는 존재. TileLang에 global/shared/register/address-space·barrier·atomic 표현 없음 |
| symbolic/dynamic shape | bounds, tile, axis size에 symbol을 쓸 수 있으나 typed `DimExpr`, symbol binding/guard 체계가 없음. wildcard는 runtime symbol과 다름 |
| side effect/alias/mutation | Store/Seq가 effect를 표현하고 Access에 base name/index/view가 있음. 서로 다른 외부 인자의 alias·in-place ABI·effect token은 없음 |
| verifier/canonicalization | `LoopAnalysis::modify`의 keyed slot 정렬·Seq 정리·forwarding, postprocess의 DCE·scope 검사 존재. 전체 타입/shape/메모리 안전 verifier는 아님 |

근거: [TileLang 정의](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:120), [shape 표현](/home/chani227/Project-Trinity/Trinity/optimizer/src/shape.rs:6), [modify](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:1545), [validate_scope](/home/chani227/Project-Trinity/Trinity/optimizer/src/postprocess.rs:396).

### 2.3 correctness를 위해 먼저 확정할 의미

| 문제와 근거 | 제안하는 계약 / 초기 처리 |
|---|---|
| 명세는 input write 금지이지만 [prenorm source:92](/home/chani227/Project-Trinity/Trinity/optimizer/playground/prenorm_all_v3_source.txt:92)는 `(store (view (input K_cache,V_cache) ...) ...)` 사용. [명세:44](/home/chani227/Project-Trinity/Trinity/optimizer/docs/IR_SPEC.md:44)와 충돌 | ABI에 `ReadOnly / WriteOnly / ReadWrite`를 둔다. KV cache mutation은 명시적인 ReadWrite binding으로만 허용. 처음에는 ReadOnly 경로부터 구현 |
| [prenorm source:2](/home/chani227/Project-Trinity/Trinity/optimizer/playground/prenorm_all_v3_source.txt:2)는 X2의 초기 Store 없이 자기 자신을 load해 누적. 기존 [accumulator 초기화](/home/chani227/Project-Trinity/Trinity/backend/codegen/triton_generator/analysis/accumulators.py:235)는 zeros 생성 | `InitPolicy = Uninitialized / Fill(value) / ReadExisting`. sum=0을 이름·첫 load 패턴만으로 추측하지 않는다. legacy adapter에 명시한 fixture 계약을 통해 초기값 보충 |
| `elem(lv)`의 주소는 `lv // enclosing_step`; `(tile lv 1)`과 다름. [language.rs:136](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:136). 기존 backend는 rank를 유지하지만 [stage2_load_shape](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:1886)는 axis를 제거하는 helper | 초기 계약은 backend·split-K unsqueeze와 맞춰 길이 1 axis 유지, 제거는 `squeeze`로 명시. 해당 helper는 이번 `src` 검색에서 호출부가 확인되지 않으므로 활성 경로의 버그로 단정하지 않는다. nonzero start에서도 `lv // step`을 보존; `(lv-start)//step` 변경 금지 |
| `layout` 순서를 presentation으로 설명한 [주석](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:223)과 실제 storage-axis mapping에 사용하는 [desugar](/home/chani227/Project-Trinity/Trinity/backend/codegen/ViewDesugar.py:392) | source view의 axis→storage axis map을 importer에서 한 번 확정하고 보존. slot 정렬과 storage 순서는 별개. 크기가 같은 축을 size만 보고 동일시하지 않음 |
| [collect_view_shapes](/home/chani227/Project-Trinity/Trinity/backend/codegen/ViewDesugar.py:363)는 conflict 시 첫 shape 사용. [LoopAnalysis::merge](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:349)는 Concrete 충돌을 검증하지 않음 | physical storage shape와 각 access/view shape를 분리. 같은 view 계약의 충돌은 오류; 합법적인 다른 view라면 명시적 index map 필요 |
| [bare keyed fallback](/home/chani227/Project-Trinity/Trinity/backend/codegen/ViewDesugar.py:217)은 slot 순서 사용, [keyed conversion](/home/chani227/Project-Trinity/Trinity/backend/codegen/ViewDesugar.py:404)은 중복 slot을 dictionary에 덮어씀 | bare keyed access는 명시적 ABI axis map 없으면 거부. 중복/알 수 없는 축은 오류. 누락 slot의 `fulltile` 기본값은 source dialect 계약으로 선언하고 normalization에서 채움 |
| `bcast` wildcard와 unresolved symbolic axis가 둘 다 Wildcard가 될 수 있음 | `BroadcastExtent`와 `DimExpr::Symbol`을 분리. broadcast extent는 consumer에서 결정; 미해결이면 오류/명시적 shape guard |
| [mloop 명세:196](/home/chani227/Project-Trinity/Trinity/optimizer/docs/IR_SPEC.md:196)는 `stop-start=nsplit*step`을 말하지만 [language.rs:197](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:197), [desugar:95](/home/chani227/Project-Trinity/Trinity/backend/codegen/ViewDesugar.py:95)는 chunk 내부 serial iteration을 허용 | 기존 실행 해석에 따라 `chunk=(stop-start)/nsplit`, `chunk % step == 0`, `nsplit>0`를 초기 조건으로 사용. 한 chunk가 반드시 한 step이라는 제약은 채택하지 않음. 비균등 split은 후속 설계 |
| [MLoopSplitK](/home/chani227/Project-Trinity/Trinity/optimizer/src/applier.rs:1257)는 scratch+형제 final reduce 생성. `SplitKOp`는 현재 Add만 존재 | mloop를 sum split-K로 우선 지원. sibling reduce를 놓치거나 최종 reduce를 중복 생성하지 않음. initial accumulator를 split마다 복제하지 않음 |
| `(tensor Q1,K1,V1)` 등 comma name: Rust는 하나의 Symbol, [Python parser](/home/chani227/Project-Trinity/Trinity/backend/codegen/IrParser.py:62)는 여러 children | kernel correctness baseline에서는 단일 tensor name 사용. 실제 bundling 의미는 adapter의 explicit grouped binding으로 표현하거나 거부; 문자열 split만으로 tensor 개수를 결정하지 않음 |
| `/`가 수치 연산과 [index 정수 나눗셈](/home/chani227/Project-Trinity/Trinity/backend/codegen/triton_generator/codegen/indexing.py:25) 양쪽에 쓰임; dtype/precision 전체 규칙 없음 | `IndexExpr::FloorDiv`와 typed numeric division 분리. scalar literal의 dtype, promotion, cast, dot input/accumulator dtype, NaN·signed-zero·reassociation 정책을 PrecisionPolicy로 고정 |

이 항목들은 현재 프로그램의 의미를 정하는 계약 변경이다. 잘못된 입력을 Triton compile에 넘겨 발견하게 하지 않는다. `force_scope_fix` 같은 자동 변수 치환은 production verifier의 대체 수단으로 사용하지 않는다.

### 2.4 기존 backend에서 재사용할 것과 교체할 것

재사용할 것은 benchmark 입력, PyTorch reference, loop/split-K 사례, dot/GEMV의 지원 조건에 대한 경험이다. [dot_lowering.py](/home/chani227/Project-Trinity/Trinity/backend/codegen/triton_generator/codegen/dot_lowering.py)의 작은 matmul 분기도 새 capability test의 재료로 삼는다. 기존 출력은 compatibility 비교 대상이고 수치적 정답의 유일한 oracle은 아니다.

교체할 것은 shared `CodeGenState`에 shape·temp·precision·scope를 섞는 방식과 AST mutation이다. [ScalarOps.generate_binary_op](/home/chani227/Project-Trinity/Trinity/backend/codegen/triton_generator/codegen/math_ops.py:40)는 shape 추론과 indentation/source 생성을 동시에 수행한다. [promote_dot_operands](/home/chani227/Project-Trinity/Trinity/backend/codegen/triton_generator/state.py:64)는 문자열 suffix로 dtype을 판단하고 fp16으로 내린다. [generate_header](/home/chani227/Project-Trinity/Trinity/backend/codegen/triton_generator/kernel.py:107)는 shape runtime parameter 없이 stride와 constants를 `tl.constexpr`로 만든다.

기존 backend에 mask와 stride 처리가 없다고 볼 수는 없다. [MaskGenerator](/home/chani227/Project-Trinity/Trinity/backend/codegen/triton_generator/codegen/masking.py:21)와 [Indexer](/home/chani227/Project-Trinity/Trinity/backend/codegen/triton_generator/codegen/indexing.py:46)가 존재한다. 다만 추론·문자열·scope cache에 흩어져 있고 semantic verifier가 그 결과를 보장하지 않는다. `analysis/accumulators.py`, `analysis/allocations.py`의 source emission도 새 구조에서는 제거한다.

## 3. End-to-end pipeline와 optimizer 경계

### 3.1 단계별 입력과 출력

```text
EGraph + structural selection + selected value bodies + ABI declarations
  ↓ ExportSelection: 원본/structural ID 매핑과 occurrence 생성
SelectedProgram + OptimizationMetadata + ProgramAbi
  ↓ IRVerifier: syntax/scope/axis/binding/init/precision 검증
VerifiedProgram
  ↓ Canonicalizer: explicit axes/broadcast/cast/index-domain/loop bindings
KernelIR (normalized structured regions + typed SSA values + memory effects)
  ↓ DependencyAnalyzer: 선택된 프로그램의 RAW/WAR/WAW/alias/order 분석
KernelIR + DependencyInfo
  ↓ FusionPlanner / KernelPartitioner: 합법적인 커널 그룹과 materialization 후보
KernelGroups
  ↓ AlgorithmSelector / TilingPlanner / LayoutPlanner: 후보별 공동 결정
ScheduledKernel + buffer layouts + guard set
  ↓ IndexLowering / ReductionLowering / MemoryPlanner + scheduled verifier
IndexedKernel + launch plan + scratch plan
  ↓ PhysicalPlanBuilder::finalize: action/value DAG와 storage 경계 검증
PhysicalPlan (kernel body·ABI·정확성 metadata를 소유)
  ↓ generic TritonKernelBuilder 또는 typed template renderer
Triton AST → Python module + manifest + source map
  ↓ Python RuntimeCompiler: guards / allocation / compile / launch
CompiledProgram
```

`Normalized/Scheduled/Indexed`는 같은 작은 KernelIR의 상태별 wrapper와 분석 결과다. 세 종류의 범용 IR framework를 각각 만들지 않는다. 물리 실행 그래프의 node가 커널 전체를 나타내고, 커널 내부 loop의 수천 iteration을 PhysicalPlan operation으로 전개하지 않는다.

### 3.2 Rust 의존성 방향

제안하는 작은 공유 crate는 `trinity-lowering/crates/trinity-ir-contract`에 둔다. `egg`, Z3, Triton/Python에 의존하지 않는 owned Rust 타입만 포함한다.

```text
Trinity/optimizer (trinity crate) ───→ trinity-ir-contract
trinity-lowering (backend crate) ──→ trinity-ir-contract
integration CLI ──────────────────→ optimizer + trinity-lowering
Python runtime ───────────────────→ generated artifact/manifest
```

optimizer adapter는 `Trinity/optimizer/src/backend_export.rs`에 추가한다. `TileLang`에 대한 exhaustive match로 `SelectedProgram`을 만들고 지원하지 않는 variant에 명확한 오류를 반환한다. backend core가 optimizer crate에 의존하면 egg fork·solver·extraction lifetime까지 끌려오므로 피한다. optimizer와 backend를 하나의 거대한 crate로 합칠 필요는 없다.

단일 프로세스에서는 Rust 함수로 owned 구조체를 전달한다. 현재 Python extraction workflow를 지원하려면 **동일한 구조체의 versioned JSON**도 제공한다. JSON과 in-memory 경로를 별도의 의미 해석기로 만들지 않는다. PyO3는 반복 CLI 비용이 실제 문제가 될 때 추가한다.

초기에는 raw TileLang enum을 공유 crate로 옮기지 않는다. rewrite 전용 form과 `egg::Id`가 backend 계약에 섞이는 비용이 더 크다. 대신 exhaustive exporter + operator coverage test로 두 표현의 drift를 잡는다.

### 3.3 어떤 metadata를 넘길 것인가

| 구분 | 넘길 정보 | backend에서의 신뢰와 수명 |
|---|---|---|
| 필수 의미 계약 | tensor identity, dtype, storage/view shape, 축 매핑, runtime stride binding, alias/mutation, init, output observability, precision policy | 필수 입력. optimizer에 현재 없는 정보는 frontend/호출자가 ABI로 보충. 모르면 추측하지 않음 |
| 선택된 결정 | selected loop kind/range/tile, `nsplit`, 실제 선택된 value DAG, specialization assumptions | v0에서 고정. 다른 선택은 별도 candidate 및 재검증 대상으로 취급 |
| 재검증 가능한 분석 | selected-node shape, access summaries, scope/loop ownership, explicit constraints | 선택된 occurrence와 연결 후 검증. shape/access를 정확하게 재계산할 수 있으면 비교 |
| 비용/성능 힌트 | `df_flops`, `df_read_bytes`, `df_write_bytes`, `iter_count`, tile assignment, reuse pairs, resource estimate | 해당 graph/config/target에서만 유효. legality 증명이 아님 |
| provenance | snapshot digest, original/structural enode ID, selected occurrence, value-body origin, pass history | debug/설명용. egraph lifetime 종료 후에도 owned 값으로 유지 |

`LoopData.read_set/write_set`는 merge에서 union한다. 이는 **e-class의 여러 대안에 대한 요약**이며 최종 커널의 정확한 access set이 아니다. [merge:315](/home/chani227/Project-Trinity/Trinity/optimizer/src/language.rs:315)를 그대로 serialize하여 scheduler의 유일한 근거로 사용하지 않는다. [PairTable](/home/chani227/Project-Trinity/Trinity/optimizer/src/extract_preprocess/reuse_pairs.rs:42)의 조건부 reuse도 조건·scope를 다시 확인해야 한다.

비용은 dtype와 target 가정까지 함께 기록한다. 예를 들어 [dataflow_ilp.rs:480](/home/chani227/Project-Trinity/Trinity/optimizer/src/extract_preprocess/dataflow_ilp.rs:480)의 byte 추정은 기본 2-byte dtype을 가정한다. 새 fp32 kernel에 같은 byte 수를 사실처럼 적용하면 안 된다.

### 3.4 ID·metadata 유효성 계약

- `EnodeIndex`의 안정성은 같은 snapshot 기준이다. 서로 다른 extraction/재구성 사이의 영구 ID로 사용하지 않는다.
- provenance는 `(snapshot_digest, original_enode_id, structural_enode_id, occurrence_path)`로 연결한다. 여러 실행 위치에 등장하는 같은 e-class는 여러 occurrence를 만든다.
- effectful Store/Load와 loop binding을 expression DAG memo 하나로 공유하지 않는다. `StmtId`는 실행 위치, `ValueId`는 region 안의 계산 결과, `BufferVersionId`는 메모리 시점을 식별한다.
- `val_exprs`로 대체된 body는 선택된 body 자체와 그 origin을 보존한다. 과거 비용만 붙이고 원래 value child를 다시 선택하는 경로는 금지한다.
- 각 pass는 `old_id → new_ids` provenance mapping과 invalidated analyses를 반환한다. 현 text postprocess를 사용하는 전환기에는 postprocess 이후 정확한 분석을 다시 하고, 매핑 불가능한 옛 통계는 program 수준의 참고치로만 남긴다.
- tile/loop/fusion/layout/precision 변경 시 `graph_digest + assumptions_digest + target + cost_model_version`이 달라진다. 해당 비용·reuse·resource estimate를 무효화한다.
- backend가 tile 값을 독자 변경하고 원래 optimizer cost를 계속 보고하는 상황을 막는다. 후보 비교는 같은 end-to-end 실행 계획을 대상으로 한다.

## 4. 컴포넌트와 디렉토리 구조

아래는 제안 구조다. 현재 root package를 유지하며 작은 contract crate만 workspace member로 추가한다. 파일은 해당 phase에서 필요할 때 만든다.

```text
/home/chani227/Project-Trinity/trinity-lowering/
  Cargo.toml                         # 기존 backend package + workspace
  crates/trinity-ir-contract/src/
    lib.rs                           # public interchange types
    program.rs                       # SelectedProgram, SourceOp, StmtId, Origin
    abi.rs                           # ProgramAbi, BufferDesc, TensorView, InitPolicy
    metadata.rs                      # OptimizationMetadata, AnalysisStamp, CostHint
    types.rs                         # DType, DimExpr, SymbolRole, PrecisionPolicy
  src/
    lib.rs                           # lower(), emit_triton()
    config.rs                        # TargetSpec, LoweringOptions, capability query
    import.rs                        # contract validation entry, legacy adapters
    ir/
      mod.rs                         # KernelIR, ValueId, Region, buffer versions
      expr.rs                        # ExprOp, ValueType, typed expression DAG
      index.rs                       # IndexExpr, IndexMap, Predicate, Access
      control.rs                     # ParallelFor, SerialFor, iter_args, Reduce
    passes/
      verify.rs                      # IRVerifier, verify_scheduled()
      canonicalize.rs                # Canonicalizer, normalize_views/broadcasts/casts
      dependencies.rs                # DependencyAnalyzer, AccessSummary
      reductions.rs                  # accumulator recognition, ReductionLowering
    schedule/
      mod.rs                         # plan_kernel_groups(), plan_candidate()
      fusion.rs                      # FusionPlanner, FusionDecision/rejection reason
      tiling.rs                      # TilingPlanner, ProgramMapping, ReductionStrategy
      layout.rs                      # LayoutPlanner, freeze_candidate_layouts()
      memory.rs                      # MemoryPlanner, lifetimes/scratch/materialization
      indexing.rs                    # IndexLowering, range proof, mask generation
    physical/                        # 기존 plan/builder/finalize/canonical/error 확장
    implementation/                  # 기존 Hopper/NVLS registry 유지
      triton.rs                      # generic/template capability descriptors
    codegen/triton/
      ast.rs                         # TritonExpr, TritonStmt, Function, Module
      expressions.rs                 # TritonExpressionEmitter, typed CSE
      builder.rs                     # TritonKernelBuilder, signature/structured body
      printer.rs                     # precedence/indentation/source-map output
      manifest.rs                    # GeneratedProgram, LaunchSpec, AllocationSpec
      compat.rs                      # tested Triton version / target feature profiles
    templates/
      registry.rs                    # TemplateRegistry, can_implement(), prepare()
      context.rs                     # GemmRenderContext, FusionHook, checked bindings
      renderer.rs                    # restricted Jinja-compatible rendering
      gemm.py.jinja                  # algorithm skeleton (Phase 5)
    selection.rs                     # AlgorithmSelector, static ranking then autotune
    diagnostics.rs                   # stage/node/constraint-rich CompileDiagnostic
  python/trinity_runtime/
    compiler.py                      # RuntimeCompiler, load/compile generated module
    launch.py                        # guards, allocations, ordered launch, lifetimes
    cache.py                         # KernelCache, durable artifact manifest
    autotune.py                      # isolated candidate benchmark (Phase 6)
  tests/
    contract.rs                      # export/roundtrip/metadata identity
    semantics.rs                     # dtype/axis/elem/effect/init rejection
    indexing.rs                      # address and mask coverage against scalar oracle
    physical_plan.rs                 # 기존 invariant tests 확장
    gpu/                             # generated kernels vs independent reference
    fixtures/                        # selected IR + explicit ABI + expected semantics
  docs/TRITON_BACKEND_ARCHITECTURE.md
```

### 4.1 기존 PhysicalPlan의 확장 범위

현재 [ValueInstance](/home/chani227/Project-Trinity/trinity-lowering/src/physical/plan.rs:99)는 `DType + Box<[usize]> + Storage`를 가진다. [OperationPayload](/home/chani227/Project-Trinity/trinity-lowering/src/physical/plan.rs:198)는 concrete implementation 기반 Compute/Communication이고 계산 body 자체는 없다. [PhysicalPlan](/home/chani227/Project-Trinity/trinity-lowering/src/physical/plan.rs:261)은 단일 output이다.

현재 [finalize](/home/chani227/Project-Trinity/trinity-lowering/src/physical/finalize.rs:11)의 producer uniqueness, operation/action DAG, action membership, cross-action storage 검증은 재사용한다. **GEMM의 type/shape 의미 검증까지 수행하는 것은 아니다.** [GemmImplementation::enumerate](/home/chani227/Project-Trinity/trinity-lowering/src/implementation/mod.rs:20)는 shape가 사전 검증되었다고 가정한다.

| 기존 요소 | 필요한 변경 |
|---|---|
| `ValueInstance.shape: [usize]` | Phase 1은 concrete 유지. Phase 2에서 symbolic physical layout 표현 도입; concrete helper 제공. 기존 Hopper enumerator에는 모든 required extent가 concrete일 때만 전달 |
| `DType::{Bf16,Fp32}` | tensor fp16 필요 시 추가. `<=`의 Bool, index I32/I64는 ValueType/IndexType에 표현. contract가 dtype 정의의 단일 소유자가 되고 기존 public path는 re-export |
| `ComputeOperation.implementation` | generic Triton kernel instance와 typed body를 표현하도록 확장. plan이 kernel table을 소유하고 operation이 KernelId로 참조; naked implementation string에 body를 숨기지 않음 |
| `output: TensorBinding` | ordered `outputs: Vec<TensorBinding>`. 기존 `finalize(name,value)`는 single-output convenience API로 보존 가능 |
| immutable producer DAG | mutable buffer는 동일 storage를 참조하는 서로 다른 logical versions로 표현. 필요한 RAW/WAR/WAW ordering은 명시적 action dependency. in-place cycle을 그대로 그래프에 넣지 않음 |
| `Storage::External/Global/Shared/Register` | external binding과 allocation ownership 명확화. Triton tensor temporary의 register/shared 실현은 Triton compiler가 결정하며 임의의 shared-memory layout 강제를 지원한다고 가정하지 않음 |
| `Action` | Triton v0에서는 하나의 검증된 fused kernel launch가 하나의 compute action. 기존 communication action은 별도 dispatch, unsupported Triton 통신은 거부 |
| canonicalization/hash | kernel body·layout·precision·guards·effect dependencies까지 canonical equality/hash에 포함. debug provenance와 cost hints는 executable identity에서 제외 |

`PhysicalPlanBuilder::finalize` 후 ID가 재번호화될 수 있다. 현재 [builder 주석](/home/chani227/Project-Trinity/trinity-lowering/src/physical/builder.rs:65)을 유지하고 source map에도 finalization remap을 적용한다. 정당한 view/alias relation은 plan이 소유하는 semantic 데이터이며 hash 밖 sidecar에만 보관하지 않는다.

기존 [HopperWgmmaBf16](/home/chani227/Project-Trinity/trinity-lowering/src/implementation/hopper_wgmma.rs:11)는 BF16 2D tile candidate `[128,128,64]`를 열거하는 구현이다. 이것을 생성된 `tl.dot`이 특정 WGMMA 명령을 반드시 사용한다는 보장으로 바꾸지 않는다. Triton 경로는 별도 implementation ID/capability로 등록한다.

## 5. 핵심 인터페이스와 정책

다음은 책임을 설명하는 Rust 유사 pseudocode이며 현재 public API가 아니다.

```rust
struct ExportedProgram {
    schema_version: u32,
    semantics_version: u32,
    program: SelectedProgram,          // owned; no egg::Id references
    abi: ProgramAbi,                   // required declarations, not cost hints
    metadata: OptimizationMetadata,
}

enum SymbolRole { RuntimeExtent, RuntimeStride, SemanticConstant, ScheduleParameter }
enum DimExpr { Const(u64), Symbol(SymbolId), Add(Box<Self>, Box<Self>),
               Mul(Box<Self>, Box<Self>), FloorDiv(Box<Self>, Box<Self>) }
enum InitPolicy { Uninitialized, Fill(TypedScalar), ReadExisting }
enum AccessMode { ReadOnly, WriteOnly, ReadWrite }

struct BufferDesc {
    id: BufferId,
    dtype: DType,
    storage_shape: Vec<DimExpr>,
    strides: Vec<DimExpr>,              // element strides, not byte strides
    storage_offset: DimExpr,            // relative to declared storage base
    alias_group: AliasGroupId,
    access: AccessMode,
    init: InitPolicy,
}
struct TensorView {
    buffer: BufferId,
    axes: Vec<AxisId>,
    extents: Vec<DimExpr>,
    logical_to_storage: IndexMap,
}
struct AnalysisStamp {
    selected_program_digest: Digest,
    assumptions_digest: Digest,         // tile/split/shape/precision
    target: TargetSpec,
    producer_version: String,
}
struct CostHint { stamp: AnalysisStamp, flops: u64, read_bytes: u64,
                  write_bytes: u64, source: Origin }

fn export_selected(selection: &Selection, snapshot: &OptimizerSnapshot,
                   abi: ProgramAbi) -> Result<ExportedProgram, ExportError>;
fn verify(input: ExportedProgram) -> Result<VerifiedProgram, Diagnostics>;
fn normalize(input: VerifiedProgram) -> Result<KernelIR, Diagnostics>;
fn lower(input: ExportedProgram, options: &LoweringOptions)
    -> Result<PhysicalPlan, Diagnostics>;
fn emit_triton(plan: &PhysicalPlan, profile: &TritonProfile)
    -> Result<GeneratedProgram, Diagnostics>;
```

storage pointer 기준을 하나로 고정한다. 위 내부 `storage_offset`은 storage base 기준이다. Python tensor의 `data_ptr()`처럼 view 시작점을 전달한다면 runtime binding이 offset을 이미 반영한 것으로 표시하여 두 번 더하지 않는다.

```rust
struct KernelIR {
    values: Arena<ValueId, TypedValue>,
    regions: Arena<RegionId, Region>,
    buffers: BufferTable,
    origins: OriginMap,
}
enum Statement {
    Let { result: ValueId, expr: ExprOp },
    Store { target: Access, value: ValueId, before: BufferVersionId,
            after: BufferVersionId },
    SerialFor { binding: LoopId, range: LoopRange, iter_args: Vec<IterArg>,
                body: RegionId, results: Vec<ValueId> },
    ParallelFor { binding: LoopId, range: LoopRange, body: RegionId },
    Yield(Vec<ValueId>),
}
struct Reduce {
    input: ValueId, axes: Vec<AxisId>, combiner: ReductionOp,
    identity: TypedScalar, accumulator_type: DType, result_type: DType,
    order_policy: ReductionOrder,
}
struct ScheduledKernel {
    body: KernelIR,
    program_mapping: ProgramMapping,
    tile_config: TileConfig,
    reduction_strategy: ReductionStrategy,
    layouts: ResolvedLayouts,
    guards: Vec<Constraint>,
}
struct IndexedAccess {
    buffer: BufferId,
    element_offset: IndexExpr,
    validity: Predicate,
    index_type: IndexType,
}
```

`Load`는 해당 memory version과 predicate를 읽는 expression이다. CSE key에는 region/dominance, memory version, address, mask, dtype가 들어간다. 다른 Store를 사이에 둔 같은 주소 load를 동일 Value로 합치지 않는다.

`InitPolicy`만으로 초기화 시점을 결정하지 않는다. normalization 결과에 `InitSpec(buffer_or_iter_arg, value, owning_region, before_statement)`를 만들어 program 시작, kernel 시작, outer/inner loop 시작 중 어느 범위에서 초기화되는지 명시한다. register accumulator는 해당 `SerialFor`의 초기 `iter_arg`로, global scratch fill은 필요할 경우 별도 action으로 내린다.

### 5.1 컴포넌트별 입력·출력·소유 책임

| 컴포넌트 | 입력 → 출력 | 핵심 책임 |
|---|---|---|
| IRVerifier | ExportedProgram → VerifiedProgram | arity, statement/value 구분, symbol binding, axis unique/coverage, rank/dtype, reduction axes, definite initialization, write legality, alias 계약 |
| Canonicalizer | VerifiedProgram → KernelIR | named/positional index 통일, broadcast extent/cast 명시, permute/squeeze 정규화. 없는 reshape 의미를 발명하지 않음 |
| DependencyAnalyzer | KernelIR → DependencyInfo | 선택된 access에 대해 RAW/WAR/WAW, alias, loop-carried effects. 미해결 alias는 보수적 dependence |
| FusionPlanner | KernelIR + dependencies + limits → KernelGroups | 동일 program에서 만족 가능한 producer-consumer 관계만 fusion; 거절 이유도 반환 |
| TilingPlanner | group + constraints + target → candidate schedule | source loop/tile와 GPU lane block 구분, PID flattening, X/Y/R block, serial/reduction 전략 |
| LayoutPlanner / MemoryPlanner | candidate + ABI → physical layouts / allocation plan | external layout 보존, internal materialization·lifetimes·scratch·output stride 결정 |
| IndexLowering | scheduled view access → IndexedAccess | logical→storage→pointer offset, bounds와 tile padding predicate, int width 증명 |
| TritonExpressionEmitter | typed ExprOp + value map → TritonExpr | arithmetic/comparison/cast/intrinsic/broadcast 및 scope 안전 CSE |
| TritonKernelBuilder | IndexedKernel → TritonFunction | signature, structured loops, loads/reduces/stores, temporary binding. scheduling 결정은 받기만 함 |
| TemplateRegistry | validated operation/candidate → eligible templates/context | dtype/rank/layout/target 지원 검사; raw source IR을 template에 노출하지 않음 |
| AlgorithmSelector | eligible candidates + hints/measurements → chosen plan | 초기 deterministic ranking; 이후 실제 end-to-end benchmark. 후보 없음과 compilation failure 구별 |
| RuntimeCompiler | GeneratedProgram + bound args → CompiledProgram | guards, compiler invocation, diagnostic, launch order. graph rewrite 없음 |
| KernelCache | artifact/config/environment identity → verified cached entry | 소스 cache, compiled artifact, tuning result의 수명·키 구분 |

### 5.2 scheduling, fusion, layout

source `ploop`는 독립 iteration이라는 의미이며 무조건 `program_id(axis)` 한 개와 1:1 대응하지 않는다. 선두 parallel loop chain을 grid로 옮기되, GPU grid 차원보다 많은 축은 검증된 flatten/unflatten mapping으로 표현한다. kernel 내부 parallel loop는 vector lane 또는 serial subloop로 구현할 수 있고 GPU grid를 내부에서 새로 만들 수는 없다. `sloop`의 순서와 loop-carried value는 보존한다.

source tile width, loop step, padded lane block은 서로 다른 값이다. 예를 들어 step=96, access width=80, lane block=128이면 다음 iteration은 96만큼 진행하고 valid lane은 80개다. block을 128로 padding했다고 source step을 128로 바꾸지 않는다.

fusion은 normalized KernelIR에서 수행한다. optimizer가 탐색한 fusion 구조를 우선 존중하고 backend에서는 kernel boundary의 합법성과 구현을 확정한다. Phase 3 이후 추가 fusion은 아래 조건을 모두 만족하는 별도 candidate로 만든다.

- producer 결과가 consumer와 같은 GPU program에서 사용 가능하고 inter-program synchronization이 필요하지 않음.
- alias를 포함한 RAW/WAR/WAW와 observable mutation 순서 보존.
- iteration/index map의 정합성이 증명되고 reduction 결과가 준비된 뒤 consumer 실행.
- multi-consumer materialization 제거에 따른 duplicated compute가 설정된 budget 이내.
- liveness 기반 peak value footprint와 target resource limit 이내. optimizer의 byte 추정만으로 register fit을 보장하지 않음.
- FP reassociation 또는 reduction tree 변경은 해당 precision/order policy가 허용할 때만 수행.

layout은 두 시점으로 나눈다. **입출력 ABI의 실제 stride·alias·필수 output layout은 scheduling 전에 고정**한다. **새 internal buffer의 physical layout은 schedule/algorithm 후보별로 선택하고 IndexLowering 전에 고정**한다. output layout 요구가 없으면 초기에는 contiguous 출력을 선택하고 manifest가 이를 선언한다. 외부에서 output buffer를 주면 그 stride를 따른다.

### 5.3 dynamic shape, index와 mask

`DimExpr::Symbol(N)`은 runtime argument, `TILE_K/XBLOCK/RBLOCK/num_warps`는 schedule parameter다. `tl.arange` lane shape 등 compile-time 값이 필요한 곳은 constexpr를 사용한다. 모든 크기·stride를 constexpr로 만들지는 않는다. runtime `N`도 Triton 자체 specialization 정책의 영향을 받을 수 있으므로 wrapper signature/guard/cache policy를 함께 검증한다.

**현재 static optimizer 결과에서 숫자 4096을 지우고 N으로 바꾸는 것은 dynamic-shape 지원이 아니다.** upstream이 symbolic semantics와 유효한 rewrite assumptions를 내보내거나 runtime guard로 고정 shape임을 확인해야 한다. Phase 2는 backend의 symbolic ABI/index 지원과 upstream export의 symbolic 보존을 함께 완료해야 한다.

IndexLowering은 `storage_offset + Σ(storage_index[d] * stride[d])`를 만든다. transpose/permute는 index map 변환이며 항상 data movement는 아니다. contiguous reshape가 필요한 경우 layout 조건을 증명하고, 실패하면 명시적 copy candidate 또는 unsupported 오류로 처리한다.

logical access validity는 KernelIR에 유지하고, **실제 padded block의 boundary mask는 IndexLowering에서 생성하여 IndexedAccess에 명시**한다. 이후 emitter/template가 이를 추론하지 않는다. shape 상한뿐 아니라 lower bound, source tile width, loop end, split chunk end도 검사한다.

masked load의 `other=0`만으로 reduction correctness가 보장되지 않는다. `exp(0)=1`처럼 계산 후 padding 값이 바뀔 수 있다. value validity를 전파하고 reduction 직전에 invalid lane을 reducer identity(sum=0, max=-inf, min=+inf)로 바꾼다. masked Store는 valid lane에만 실행한다. NaN·empty reduction·integer identity는 dtype/reducer 계약으로 별도 정의한다.

32-bit index는 pointer offset뿐 아니라 중간 multiply/add의 범위가 안전할 때만 사용한다. 모르면 64-bit로 먼저 계산한다. 음수 stride나 overlapping writable view는 초기에는 명시적으로 거부하고 진단한다. indirect indexing은 source에 전용 typed form이 없으므로 이번 첫 범위에서 제외; 나중에 load된 index dtype·bounds 정책을 명시한 Gather/Scatter 계약과 함께 추가한다.

### 5.4 reduction, multi-stage와 atomic

normalized IR에서는 `Reduce`와 `SerialFor(iter_args)`를 구분해 유지한다. arbitrary recurrence를 reduction이라고 가정하지 않는다. scheduler는 `BlockReduce / SerialTiledReduce / TwoPassReduce / SplitK` 중 합법적인 전략을 선택하고 reduction lowering이 구체화한다.

multi-stage reduction은 `partial kernel → global scratch → final kernel`이라는 PhysicalPlan action graph로 표현한다. launch 순서와 scratch lifetime이 manifest에 포함된다. `mloop`의 scratch와 sibling final reduction을 인식해 이 그래프로 연결한다. resource 때문에 kernel을 분리할 때도 global dependency를 만드는 비용까지 비교한다.

atomic은 v0/v1의 기본 fallback으로 사용하지 않는다. 후속 구현에서는 `AtomicReduce(op, ordering, initial_value)`를 명시하고 target/dtype capability, initialization action, nondeterminism 허용 여부를 검사한다. unsupported 또는 floating order policy 불일치이면 two-pass 후보를 선택하거나 거부한다.

### 5.5 source 생성과 runtime interface

```rust
impl TritonKernelBuilder {
    fn emit_index(&mut self, expr: &IndexExpr) -> ExprId;
    fn emit_mask(&mut self, predicate: &Predicate) -> ExprId;
    fn emit_load(&mut self, access: &IndexedAccess, fallback: TypedScalar) -> ValueId;
    fn emit_expression(&mut self, op: &ExprOp, inputs: &[ValueId]) -> ValueId;
    fn emit_reduction(&mut self, plan: &LoweredReduction) -> ValueId;
    fn emit_store(&mut self, access: &IndexedAccess, value: ValueId);
    fn build(self) -> Result<TritonFunction, EmitError>;
}
struct GeneratedProgram {
    modules: Vec<GeneratedModule>,      // Python source + source digest
    manifest: ProgramManifest,        // ABI, guards, allocations, launch DAG
    source_map: SourceMap,             // final source line → KernelIR/source origin
}
struct LaunchSpec {
    entry: KernelName,
    args: Vec<ArgumentBinding>,
    grid: GridExpr,
    constexprs: OrderedMap<String, TypedScalar>,
    num_warps: u32,
    num_stages: u32,
    predecessors: Vec<ActionId>,
}
```

printer만 Python precedence, indentation, import deduplication과 formatting을 담당한다. IR node에 `to_triton_string()`을 넣지 않는다. 최소 `Name/Literal/Unary/Binary/Call/Subscript`, `Assign/For/If/Return`, Function/Module AST면 시작할 수 있으며 Python 전체 grammar나 MLIR을 도입할 필요는 없다.

runtime은 `load(artifact) → bind(args) → check_guards() → allocate() → compile() → launch()`를 수행한다. 초기에는 동기 compilation, 같은 stream의 순차 launch면 충분하다. compile 오류에는 kernel 이름, config, target, source 파일, 원본 node와 검증 단계 정보를 붙인다. Triton의 내부 cache를 재구현하지 않고 generated source/manifest 및 tuning result의 관리 계층을 추가한다.

영구 cache key는 canonical selected/physical program digest, ABI dtype/layout contract, specialization과 guards, precision, schedule/template version, backend version, Triton version, target architecture/features, compilation options를 포함한다. compiled binary cache는 CUDA/toolchain/driver 호환성 fingerprint도 확인한다. runtime extent 값은 specialize한 경우에만 specialization key에 넣고 일반 runtime 값은 kernel signature/guards로 표현한다.

현재 [PhysicalPlan::hash](/home/chani227/Project-Trinity/trinity-lowering/src/physical/plan.rs:312)는 process-local이며 persistent identity가 아니라고 명시되어 있다. durable cache는 versioned canonical serialization의 안정적인 digest와 manifest 일치 검증을 사용한다.

## 6. PyTorch Inductor 대응표

2026-09-05에 접근 가능한 upstream `main` 파일과 symbol을 확인했다. 아래 링크는 움직이는 `main`이며 exact commit pin은 확보하지 못했다. Trinity가 이 private API를 import한다는 제안이 아니라, 책임 분리를 참고한다는 의미다.

| Custom IR / 제안 요소 | 실제 확인한 Inductor 파일·symbol | 그대로 적용 가능 여부 | 필요한 변경 / 이유 |
|---|---|---|---|
| export/import + compile orchestration | [graph.py, GraphLowering](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/graph.py) | 개념만 | FX interpreter 대신 selected TileLang importer와 명시적 ABI |
| operator normalization | [lowering.py, register_lowering](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/lowering.py) | 개념만 | 작은 Rust exhaustive match부터 시작. dynamic Python registry 불필요 |
| KernelIR expression/reduction/buffer | [ir.py, Loops / Pointwise / Reduction / ComputedBuffer](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py) | 일부 | 기존 tile loops를 보존하는 structured region과 memory version이 필요 |
| body 분석 및 emission 반복 방문 | [loop_body.py, LoopBody](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/loop_body.py) | 아니오 | body를 다시 FX로 capture하지 않고 owned typed DAG를 방문 |
| analysis/codegen context | [virtualized.py, Virtualized](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/virtualized.py) | 아니오 | hidden virtual globals 대신 pass input/context 명시 |
| target-neutral operation interface | [ops_handler.py, OpsHandler](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ops_handler.py) | 개념만 | enum visitor와 필요할 때만 trait. 작은 연산 집합에 handler framework 전체 도입 불필요 |
| read/write/effect 분석 | [dependencies.py, ReadWrites](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/dependencies.py) | 개념 적용 | e-class union이 아니라 선택된 region/versioned storage 분석 |
| fusion/ordering/kernel groups | [scheduler.py, Scheduler](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py) | 개념 적용 | optimizer가 이미 선택한 tile/loop 구조와 비용 전제 존중 |
| iteration/PID/tile planning | [codegen/simd.py, SIMDKernel / SIMDScheduling](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/simd.py) | 일부 | fulltile/elem/mloop의 source 의미를 명시적 ProgramMapping으로 변환 |
| generic emitter | [codegen/triton.py, TritonKernel](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/triton.py) | 개념 적용 | typed AST/printer와 별도 indexing 계획. 현재 규모에 맞는 작은 builder |
| template와 candidate 선택 | [select_algorithm.py, TritonTemplate / TritonTemplateKernel / AlgorithmSelectorCache](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/select_algorithm.py) | 개념 적용 | Rust registry/context + Python benchmark. 두 template symbol은 이 파일에 위치 |
| GEMM skeleton | [templates/triton_mm.py.jinja](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/templates/triton_mm.py.jinja), [mm.py의 mm_template](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py) | skeleton 원칙 | 실제 `def_kernel`, size/stride, `load_input`, `store_output` hook 확인. Trinity는 context의 의미 결정을 더 엄격히 제한 |
| launch wrapper | [codegen/wrapper.py, PythonWrapperCodegen](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/wrapper.py) | 개념 적용 | explicit LaunchSpec와 allocation manifest로 단순화 |
| async compile | [async_compile.py, AsyncCompile](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/async_compile.py) | 초기 제외 | compile latency가 병목으로 확인된 뒤 병렬화 |
| source cache | [codecache.py, PyCodeCache](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codecache.py) | 개념 적용 | Trinity stable digest/manifest와 Triton cache 조합 |
| launch heuristic / tuning | [runtime/triton_heuristics.py, CachingAutotuner](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/runtime/triton_heuristics.py) | 단계적 | static config 먼저, correctness 확인된 소수 후보 측정부터 확장 |

Inductor의 장점인 역할 분리를 참고하되 lazy realization 전체, FX body recapture, virtualized global handlers, 대규모 symbolic engine, 여러 target backend와 persistent autotuning framework를 첫 구현에 가져오지 않는다.

## 7. Programmatic emitter와 Jinja의 경계

| 연산 종류 | 생성 방식 | 이유 | fallback |
|---|---|---|---|
| pointwise, comparison, cast, intrinsic | programmatic | expression DAG·dtype·CSE 중심 | 지원 안 되는 dtype/intrinsic은 명시적 거부 |
| bcast/unsqueeze/squeeze/transpose/permute | normalized index/value map + programmatic | view와 tensor rank 의미를 먼저 결정해야 함 | copy kernel 또는 미지원 view 오류 |
| load/store, mask, multiple output | programmatic 공통 builder | ABI와 effect/mask 일관성 | 합법적인 multi-kernel 분할 |
| sum/min/max, serial recurrence | programmatic + reduction strategy | reduction axis/identity/order가 입력마다 다름 | serial tiled / two-pass, 정책 불일치 시 오류 |
| concat | verified piecewise index/copy 또는 value assembly | source axis에 따라 접근이 갈림 | 분리 copy kernel. fusion된 gather가 안전하지 않으면 분리 |
| GEMV/작은 matmul | 검증된 dot 또는 multiply/reduce emitter | template 고정 비용보다 간단할 수 있음 | 지원되는 tiled GEMM, 둘 다 안 되면 오류 |
| GEMM/BMM | typed Jinja skeleton + builder hooks | K loop, block pipeline 구조가 정형 | 일반 tiled dot kernel; legal candidate 없으면 오류 |
| convolution | 초기 unsupported; 후속 template/decomposition | 현재 전용 TileLang op가 없음 | semantic decomposition이 명시된 경우에만 구성 연산으로 lowering |
| attention | 초기에는 기존 연산으로 분해된 프로그램 | 현재 Attention variant 없음 | 검증된 softmax/matmul pipeline; pattern 전체 의미가 확인된 후 template |
| persistent / target-specific kernels | Phase 5 이후 guarded template | target feature와 algorithm 제약이 강함 | nonpersistent 검증 후보 |
| AllGather/NVLS | 기존 implementation 별도 경로 | 현재 TileLang의 일반 Triton 연산이 아님 | 등록된 실행 backend 없으면 UnsupportedTargetOperation |

Jinja는 Phase 5에만 추가한다. Rust source 생성과 통일하기 위해 [MiniJinja 공식 문서](https://docs.rs/minijinja/latest/minijinja/)의 Jinja 호환 렌더러를 후보로 사용하되, 채택 시 제한된 문법과 dependency version을 고정한다. full Python Jinja API 호환을 전제하지 않는다.

```rust
struct GemmRenderContext {
    signature: ValidatedSignature,
    inputs: [CheckedTensorBinding; 2],
    output: CheckedTensorBinding,
    tiles: GemmTiles,
    index_plan: GemmIndexPlan,
    precision: ResolvedDotPrecision,
    prologue: Option<CheckedFusionHook>,
    epilogue: Option<CheckedFusionHook>,
    launch: LaunchSpec,
}
trait KernelTemplate {
    fn can_implement(&self, op: &VerifiedGemm, target: &TargetSpec)
        -> Result<TemplateConstraints, UnsupportedReason>;
    fn prepare(&self, kernel: &ScheduledKernel) -> Result<GemmRenderContext, Diagnostics>;
    fn render(&self, context: &GemmRenderContext) -> Result<GeneratedModule, EmitError>;
}
```

template에 raw TileLang, egraph, unknown dtype, 미해결 layout을 전달하지 않는다. signature/index/mask는 checked context와 공통 builder가 준비한다. template의 분기는 `EVEN_K`, chosen algorithm처럼 이미 검증된 config만 사용한다. prologue/epilogue hook은 input/output ValueId, index domain, dtype, effect restrictions를 가지며 이름·들여쓰기를 맞추기 위한 ad-hoc 문자열 치환을 사용하지 않는다.

예상 DSL은 `define_kernel`, `declare_shape`, `load_input`, `render_prologue`, `render_epilogue`, `store_output`다. hook은 structured fragment를 생성하고 renderer가 지정된 block에 출력한다. Jinja 파일 자체의 고정 indentation은 허용하지만 생성 후 whitespace matching으로 코드를 끼워 넣지 않는다. 등록 시 syntax check, render 후 Python parse를 수행한다.

## 8. 실제 IR에서 생성까지의 예시

### 8.1 선택한 source subprogram

[prenorm_all_v3_source.txt:20](/home/chani227/Project-Trinity/Trinity/optimizer/playground/prenorm_all_v3_source.txt:20)의 X_norm 생성 단계다. 앞 단계가 만든 X2를 읽는 **단일 action**의 예시이며 전체 prenorm/attention을 구현했다는 뜻은 아니다. 아래는 whitespace만 축약했다.

```lisp
(ploop 0 4096 tile_k k
  (store
    (view (tensor X_norm) (layout (axis a_0 16) (axis a_1 4096)))
    (/
      (load (view (input X) (layout (axis a_0 16) (axis a_1 4096)))
            (keyed_index (slot a_0 (fulltile)) (slot a_1 (tile k tile_k))))
      (bcast
        (sqrt (/ (load (view (tensor X2) (layout (axis a_0 16)))
                       (keyed_index (slot a_0 (fulltile))))
                 4096))
        1))
    (keyed_index (slot a_0 (fulltile)) (slot a_1 (tile k tile_k)))))
```

예시의 추가 ABI 가정은 `X: fp32[16,4096]`, `X2: fp32[16]`, `X_norm: fp32[16,4096]`, 실제 element strides는 호출 시 전달, alias 없음, X2는 앞 action에서 초기화/계산 완료, selected `tile_k=128`이다. source에 dtype와 tile assignment가 없으므로 이 값은 **예시를 위한 명시적 계약**이다. epsilon 추가, rsqrt 재작성, implicit fp16 cast는 하지 않는다.

### 8.2 Normalized representation

```text
buffer X      : f32[16,4096], external readonly
buffer X2     : f32[16],     initialized producer result
buffer X_norm : f32[16,4096], fresh materialized result

parallel k in [0,4096) step 128:
  v0 : f32[16,128] = load X[a_0=0:16, a_1=k:k+128]
  v1 : f32[16]     = load X2[a_0=0:16]
  v2 : f32[16]     = div(v1, scalar_f32(4096))
  v3 : f32[16]     = sqrt(v2)
  v4 : f32[16,128] = broadcast(v3, inserted_axis=1, extent=128)
  v5 : f32[16,128] = div(v0, v4)
  store X_norm[a_0=0:16, a_1=k:k+128], v5
```

이 action은 X2를 읽으므로 PhysicalPlan에 X2 producer로부터의 dependency를 가진다. 독립 함수로 검증할 때만 X2를 해당 action의 외부 입력으로 바인딩한다.

### 8.3 Scheduled/indexed representation

```text
grid=(ceil_div(4096,128),)=(32,)
k = program_id(0) * TILE_K
r = arange(0, ROW_BLOCK=16)
c = k + arange(0, COL_BLOCK=128)

X offset      = r[:,None]*sx0 + c[None,:]*sx1
X2 offset     = r*ss0
X_norm offset = r[:,None]*so0 + c[None,:]*so1
row valid     = r < R
column valid  = (c < K) AND (lane_c < TILE_K)
store valid   = row_valid[:,None] AND column_valid[None,:]
```

index multiply 전에 i64로 변환한다. 이 예시에서 R/K는 guard된 runtime ABI 값이며 source의 16/4096 계약은 유지한다. 이것만으로 arbitrary dynamic shape를 지원한다고 주장하지 않는다. TILE_K는 source step, ROW_BLOCK/COL_BLOCK은 lane allocation 크기다.

### 8.4 생성할 Triton Python 코드

아래 코드는 설계 예시이며 새 backend의 실제 생성 결과가 아니다. 기본 `triton.jit`, masked load/store, constexpr block 형태는 [Triton vector-add 공식 예제](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)를 따른다. lane extent는 [tl.arange 계약](https://triton-lang.org/main/python-api/generated/triton.language.arange.html)에 맞춘 compile-time power-of-two 값으로 계획한다.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def trinity_x_norm(
    X, X2, Y, R, K, sx0, sx1, ss0, sy0, sy1,
    TILE_K: tl.constexpr, ROW_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr,
):
    r = tl.arange(0, ROW_BLOCK).to(tl.int64)
    lane_c = tl.arange(0, COL_BLOCK).to(tl.int64)
    k = tl.program_id(0).to(tl.int64) * TILE_K
    c = k + lane_c
    row_mask = r < R
    col_mask = (c < K) & (lane_c < TILE_K)
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(X + r[:, None] * sx0 + c[None, :] * sx1,
                mask=mask, other=0.0)
    x2 = tl.load(X2 + r * ss0, mask=row_mask, other=1.0)
    denominator = tl.sqrt(x2 / 4096.0)
    y = x / denominator[:, None]
    tl.store(Y + r[:, None] * sy0 + c[None, :] * sy1, y, mask=mask)


def run_x_norm(x, x2):
    if not x.is_cuda or not x2.is_cuda or x.device != x2.device:
        raise ValueError("X and X2 must be on the same CUDA device")
    if x.dtype != torch.float32 or x2.dtype != torch.float32:
        raise ValueError("This specialization requires fp32 inputs")
    if tuple(x.shape) != (16, 4096) or tuple(x2.shape) != (16,):
        raise ValueError("Expected X[16,4096], X2[16]")
    if any(s <= 0 for s in (*x.stride(), *x2.stride())):
        raise ValueError("This example requires positive input strides")
    with torch.cuda.device(x.device):
        y = torch.empty((16, 4096), dtype=torch.float32, device=x.device)
        trinity_x_norm[(triton.cdiv(4096, 128),)](
            x, x2, y, 16, 4096,
            x.stride(0), x.stride(1), x2.stride(0),
            y.stride(0), y.stride(1),
            TILE_K=128, ROW_BLOCK=16, COL_BLOCK=128,
            num_warps=4, num_stages=1,
        )
    return y
```

launch parameter 4 warps/1 stage는 초기 유효성 확인용 후보이며 최적 성능값으로 측정한 결과가 아니다. reference는 `x / torch.sqrt(x2 / 4096.0)[:, None]`이다. 전체 프로그램에서는 runtime이 X2→X_norm dependency와 device/stream을 관리하며 위 wrapper의 별도 X2 입력을 내부 allocation binding으로 치환한다.

필요한 검증은 contiguous/strided 입력의 수치 비교, 다른 값의 X2, Inf/NaN 정책, shape/dtype guard 거부, indexing coverage다. backend dynamic phase에서는 별도로 symbolic source fixture와 tail 크기를 사용한다.

## 9. 구현 단계와 완료 조건

Phase 0을 먼저 추가한다. interface가 없고 의미 충돌이 있는 상태에서 pointwise emitter부터 옮기면 다시 Python backend의 추측이 Rust로 이동하기 때문이다. 기본 source cache/diagnostics는 첫 실행 단계부터 필요하며 성능 autotuning만 마지막에 둔다.

| 단계 | 작업 / 범위 | 완료 조건과 필요한 검증 |
|---|---|---|
| Phase 0: 계약과 semantics baseline | shared contract, ABI/init/precision/axis 규칙, selected export + provenance, fixture 선정 | in-memory/JSON 동일 결과. source ID remap, `val_exprs` 선택 보존, metadata stamp 불일치·unknown dtype·unbound symbol·중복 axis·초기화 전 load 거부. input mutation/elem/mloop 결정을 versioned 계약으로 기록 |
| Phase 1: 최소 pointwise vertical slice | concrete shapes, 단일 output, verified non-mutating program; IRVerifier→KernelIR→PhysicalPlan→Rust AST→Python launch. generic kernel payload 추가 | Add/Mul/Div/Sqrt 최소 연산과 독립 reference 수치 일치. 생성 Python parse 및 실제 GPU compile/run. input stride 오류·output guard·source map 확인. GPU 없는 CI에서는 GPU job 미실행을 명시 |
| Phase 2: broadcasting와 dynamic shape | symbolic ABI, guard, explicit broadcast/cast, view/index map, padded mask, physical symbolic layout, fp16 필요 시 지원 | runtime N을 보존한 source로 N=0,1,block-1,block,block+1, odd size; same artifact의 다양한 runtime 크기; noncontiguous input, size-1 axes, permutation, overflow 경계 주소 oracle. static export를 임의 dynamic으로 바꾸지 않음 |
| Phase 3: fusion과 여러 output | selected loop 구조 보존, producer-consumer fusion, output Vec, internal materialization/liveness, alias/version/order 확장 | fused/unfused/reference 비교, multi-consumer 및 multi-output, unsafe cross-program dependency 거부, duplicate compute budget. mutation 지원 전후 WAR/WAW 및 ABI ReadWrite 테스트 |
| Phase 4: reduction와 현재 TileLang coverage | sum/min/max, serial recurrence/init, block/serial/two-pass, mloop sum split-K 및 sibling final reduce, concat/elem/transforms 미지원 항목 정리 | sum-of-squares→X_norm 실제 2-stage fixture; negative-only max/positive-only min·odd reduction length·exp padding·empty/NaN policy; nonzero loop start·step≠access width; uneven split 거부; scratch init/lifetime/launch order; dtype별 error tolerance |
| Phase 5: specialized template | GEMM/BMM template, small GEMV fallback, prologue/epilogue hook, TargetSpec/TritonProfile 분리 | generic/template/reference 비교, M/N/K tail·transpose/stride·dtype·batch, illegal dot shape 사전 거부, fused epilogue 결과. attention/persistent는 검증된 pattern별 추가; convolution을 지원된 것으로 표시하지 않음 |
| Phase 6: autotuning와 durable cache | correctness를 통과한 후보만 benchmark; static ranking 대체, persistent source/tuning identity, 필요 시 async compile | cold/warm cache, 환경/template/precision/config 변경 invalidation, corrupted artifact 복구, repeat launch determinism policy, 전체 multi-stage latency 비교. mutation 입력과 scratch를 후보마다 복원하여 tuning이 사용자 상태를 바꾸지 않음 |

첫 구현 PR은 Phase 0과 Phase 1의 작은 수직 경로로 묶는 편이 좋다. 계약 타입만 대량 추가하고 실제 kernel을 생성하지 못하는 상태가 오래 지속되지 않도록 한다. 그 뒤 실제 X_norm subprogram과 sum-of-squares stage를 연결하면서 reduction/metadata 전달의 설계를 검증한다.

테스트 계층은 다음 네 가지를 유지한다.

1. **semantic/index oracle**: 작은 tensor에서 독립 scalar interpreter로 주소·mask·store coverage와 순서를 확인한다. expected string을 복사하는 테스트로 대체하지 않는다.
2. **GPU differential correctness**: PyTorch/scalar reference와 생성 kernel 비교. 기존 backend와 비교는 compatibility 보조 수단.
3. **plan/contract invariants**: stable export, analysis invalidation, action dependency, alias/init/lifetime, finalization remap.
4. **성능 회귀**: correctness 통과 fixture의 실제 launch 수, scratch bytes, GPU latency 측정. architecture plan 단계에서 임의 성능 목표 수치를 만들지 않는다.

기존 backend는 migration 기간 fixture 비교 경로로 유지한다. 새 capability checker가 지원한 입력만 새 경로로 보내고, 미지원은 구체적 사유를 반환한다. 기존 경로 fallback은 명시적으로 선택하는 호환 옵션이며 semantic 오류를 자동으로 숨기는 수단이 아니다.

## 10. 위험, 대안과 최종 결정

### 10.1 우선순위별 위험

| 우선순위 | 위험 | 대응 / 대안 |
|---|---|---|
| P0 | dtype/init/mutable input/elem/view 의미의 불일치 | Phase 0 계약과 verifier. 애매한 source는 explicit ABI 없이 실행하지 않음 |
| P0 | e-class ID를 selected occurrence로 오인하거나 `val_exprs` 선택 손실 | snapshot-aware export + provenance mapping. whole egraph metadata를 backend state로 공유하지 않음 |
| P0 | shape·stride·view rank를 혼동하거나 unknown axis를 추측 | BufferDesc/TensorView/IndexMap 분리; physical address oracle |
| P0 | padding·reduction identity·split-K 초기값·cross-program ordering 오류 | explicit validity/reduction/init와 action DAG, two-pass 우선 |
| P1 | optimizer tile assignment와 backend autotune tile 변경으로 비용/의미 불일치 | semantic tile 고정, 다른 후보는 재검증/재비용화; feedback에 실제 schedule fingerprint 포함 |
| P1 | 현재 PhysicalPlan의 concrete shape·단일 output·DAG 제한에 mutable/dynamic 의미를 억지로 넣음 | 단계별 명시적 확장; loop-carried 값은 kernel region 안에서 표현 |
| P1 | generic Triton에 CUDA shared/register layout 보장을 과도하게 부여 | TargetSpec와 implementation capability 경계. Triton이 제어하지 못하는 concrete 요구는 지원 후보에서 제외 |
| P1 | template와 generic emitter의 dtype/index/mask/fusion 동작 분기 | checked context + 공통 builder + 공통 verifier/reference tests |
| P1 | 기존 heuristic output을 정답으로 고정 | 독립 수치 reference. legacy-only 성공을 correctness 보장으로 취급하지 않음 |
| P2 | Rust/Python 인터페이스·packaging 비용 | typed Rust core + versioned artifact + 얇은 Python runtime. PyO3/async는 측정 후 |
| P2 | source snapshot만 검사하거나 환경 누락 cache | GPU differential tests와 versioned environment-aware manifest |

### 10.2 구현 대안 비교

| 대안 | 장점 | 단점 | 판단 |
|---|---|---|---|
| 기존 Python AST emitter에 새 operator/metadata 추가 | 가장 빠른 단기 수정 | 의미 추론과 shared state 유지, Rust optimizer와 schema drift 지속 | 급한 compatibility 수정에만 사용 |
| Rust lowering + Python semantic codegen | 기존 코드를 더 많이 재사용 | lowering/index/precision 책임이 두 언어에 남을 가능성 | 전환기 adapter는 가능; 최종 구조는 source AST까지 Rust가 소유 |
| Rust core + Rust Triton source generation + Python runtime | 선택된 metadata와 타입을 유지, 디버깅·검증 경계 명확 | emitter/index/ABI를 새로 구현해야 함 | **권장** |
| optimizer와 backend 전체를 같은 crate/egraph에 결합 | 내부 데이터 직접 접근 | optimizer dependency/lifetime/rewrite state에 coupling, durable artifact 경계 부족 | 권장하지 않음 |
| MLIR/LLVM 또는 Inductor 전체 IR 도입 | 큰 확장 기반 | 현재 tile-aware IR에 비해 구축 비용·추상화 과다 | 첫 버전에서 제외 |

### 10.3 설계 질문에 대한 확정 답변

| 질문 | 결정 |
|---|---|
| Custom IR에서 직접 Triton 출력? | compatibility 경로만. 새 backend는 verified KernelIR 경유 |
| 별도 Loop/Kernel IR? | 작은 KernelIR 필요. normalized/scheduled/indexed 상태를 공유 |
| layout 확정 시점? | external ABI는 scheduling 전, internal candidate layout은 IndexLowering 전 |
| fusion level? | normalized KernelIR. optimizer의 semantic 선택 보존 및 candidate별 재검증 |
| symbolic runtime vs constexpr? | extent/stride는 runtime 기본; lane shape·algorithm config는 constexpr, source constants는 유지 |
| output layout/stride 소유자? | caller ABI 제약을 받아 LayoutPlanner가 결정, runtime은 manifest대로 allocation |
| mask 위치? | source validity는 KernelIR, concrete boundary/padding mask는 IndexLowering 결과에 명시 |
| reduction 표현? | normalized Reduce + ordered recurrence; scheduled 단계에서 strategy 구체화 |
| multi-stage/atomic? | action DAG+scratch 우선. atomic은 명시적 semantics/capability 추가 후 |
| Triton/GPU 차이? | TargetSpec / TritonProfile / template capability / runtime compiler adapter |
| 최초 autotuning? | deterministic config로 시작. cache identity는 먼저, benchmark tuning은 Phase 6 |
| Jinja 범위? | 검증된 정형 algorithm skeleton과 hook; semantic 판단 제외 |
| debugging? | source→selected occurrence→KernelIR→final source map, pass dump, rejected-candidate reasons |
| unsupported 거부 시점? | import/semantic verifier에서 형식·의미 오류; candidate selection에서 target 미지원; source emission에서 unexpected op면 backend invariant failure |

추가 답변이 없어도 Phase 0/1은 진행 가능한 계획이다. 성능 최적화를 시작하기 전에는 실제 배포 GPU와 Triton version, 주력 dtype, 허용 numerical tolerance/reassociation, KV cache mutation을 첫 지원 범위에 넣을지를 프로젝트 실행 계약으로 고정해야 한다. 현재는 이를 확정된 요구사항으로 가정하지 않는다.

이번 작업에서는 구현 및 GPU 실행을 수행하지 않았다. 문서의 Python 예시는 Python AST 문법 검증을 통과했고, 로컬 파일 링크의 존재와 참조 줄 번호를 확인했다. runtime correctness와 성능은 각 phase의 완료 조건에 포함된다.
