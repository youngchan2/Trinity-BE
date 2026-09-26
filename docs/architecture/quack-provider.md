# Quack whole-region provider

2026-09-26 작업 트리 기준. Quack 후보는 **하나의 ploop kernel region 전체**를
구현한다. 내부 GEMM만 추출해서 나머지를 없애거나 이웃한 region을 합치지 않는다.
중첩된 선두 ploop들은 같은 영역의 grid 축이다. 독립 operation용 기존 adapter도 유지한다.

## 책임과 경로

```text
PhysicalPlan
  → analysis::regions::RegionFacts
      원본 Statement/operations + 전체 boundary/def-use/storage
  → emit::region_candidates
      ├─ TritonPlan::emit_region → 원래 영역의 kernel + run(values)
      └─ quack::pattern::match_region
           선택적 의미 helper 호출 + 자체 API 지원 판정 → QuackRegionSpecification
                                       → 외부 API 호출 run(values)
  → emit_python → PythonProgram [region_candidates 모드]
      prepare(sample_inputs): reference → 후보 compile/실행 → 정확성 → 시간 비교
      executable(inputs): 선택된 호출들을 원본 영역 순서대로 실행
```

공통 분석은 원본 식·operation coverage·scope, recurrence 초기값, local producer 연결과
view/index 사실을 제공한다. [Planning](planning.md#연산-패턴-분류)이 공통 계약의 원본이다.
GEMM·SwiGLU·정규화·RoPE 계산 패턴 인식은 Quack provider 내부에서 수행한다.
`recognition.rs`와 `recognition/`은 원본 계산을 해석하고, `pattern.rs`는 그 결과를
Quack API·하드웨어·dtype/stride 조건과 대응시켜 호출 명세를 만든다.
두 단계 모두 Quack 책임이며 공통 `prepare`/PhysicalPlan 생성에는 실행되지 않는다.

[Quack matcher](../../src/emit/provider/quack/pattern.rs)는 원본 region/facts를 받아
지원 여부와 거절 이유를 만든다. `QuackRegionSpecification`은 원본 scope,
연산별 인자 view, output view, 고정 shape/dtype, target capability,
preparation/conditions와 선택적인 의미 분석 결과를 가진다.
공통 PhysicalPlan은 수정하지 않는다. 이웃 region이나 register accumulator의
수명을 새로 추론하지 않는다.

[영역 후보 구성](../../src/emit/region.rs)은 기존 operation/Native phase 계약과
별개다. `RegionKernelCandidate`의 coverage는 `RegionScope` 전체이고,
`kernel_candidates`의 operation 후보가 자동으로 영역 전체를 담당하게 되지는 않는다.
Native `emit`/CuTe 조합에는 이 변경으로 Quack 호출이 삽입되지 않는다.

`QuackPatternAnalysis`와 `QuackPatternKind`는 `emit`에서 공개하는 Quack 진단 API다.
기존 `analysis::pattern` API는 제거했다. `Other`여도 RoPE matcher를 실행하며,
Triton/CuTe의 후보를 이 결과로 제한하지 않는다. JSON의 `pattern`/`region_pattern` 필드는
호환을 위해 유지하되 Quack 분류라는 의미다. 별도 provider는 원본 region/facts를 직접 받는다.

Reference 평가기는 `emit/region.rs`와 `emit/wrapper/reference.py`에 남는다.
Quack recognition이 보존한 원본 계산식/ordered store를 직렬화해서 PyTorch로 계산하며,
Quack API 호출 명세나 후보의 출력을 reference로 사용하지 않는다.

## Quack 연산 패턴 인식

[emit/provider/quack/recognition.rs](../../src/emit/provider/quack/recognition.rs)의
`QuackPatternAnalysis::analyze(&PhysicalPlan)`은 **하나의 kernel region 전체**를 분류한다.
Scheduled 입력은 `Statement::Region`, 그 외 입력은 최상위 loop/독립 operation이 단위다.
선두 중첩 ploop chain은 같은 영역이다. 이웃 region을 합치거나 공통 plan을 재작성하지 않는다.

| 결과 | 의미 |
| --- | --- |
| `SingleGemm` | 완전한 GEMM 및 결과 복사 |
| `GemmEpilogue` | 하나의 GEMM 뒤 pointwise 계산; provider 지원과는 별개 |
| `GatedGemm` | 동일 입력의 두 GEMM과 SwiGLU gate 결합 |
| `RmsNorm` / `LayerNorm` | 축·평균 분모·epsilon 위치가 일치하는 정규화 |
| `Softmax` | 명시적인 max correction과 exp/sum 정규화 |
| `Other` | 현재 규칙으로 영역 전체를 설명하지 못함; IR 오류 또는 fallback 불가라는 뜻은 아님 |

`RegionAnalysis`는 `scope/kind/gemm/reason`에 더해 `computation/computation_reason`을
보관한다. `RegionScope`에는 원본 statement 경로와 영역의 모든 operation ID가 있다.
`operation()`은 위치 정보, `region_for_operation()`은 소속 영역 분석을 반환한다.

기존 `GemmPattern`은 원본 Matmul/누적 operation, K-loop, 초기화, accumulator 접근,
ordered epilogue와 최종 RHS를 **borrow**한다. `value()`는 최종 store의 원래 표현식이다.
Quack 진단 분류는 arbitrary pointwise/cast를 보존할 수 있지만 Quack 지원을 보장하지 않는다.

추가된 [region summary](../../src/emit/provider/quack/recognition/region.rs)는 ordered local 정의를
따라 펼친 읽기 전용 `RegionComputation { output, expression, operation }`을 만든다.
K-loop를 완전한 Matmul, additive sum recurrence를 전체 축 sum으로 설명할 때
0부터 전체 축까지 step=접근 폭으로 순회하는지와 zero seed를 확인한다.
`P=A@B; acc+=P`도 기존 recurrence 증명을 재사용한다. 외부/nonzero seed,
loop 안의 nonlinear epilogue는 이런 축약에 포함하지 않는다.
이 식은 IR rewrite가 아니라 외부 구현의 동등한 계산 범위를 검사하는 summary다.

연결은 이름이 아닌 ValueInstanceId, 접근 view/indices, store 순서를 기준으로 한다.
최종 결과에 기여하지 않는 store, 추가로 관측되는 global/live-out 중간값,
다른 view/subtile로 local producer를 읽는 경우는 이 단일식 summary가 거절한다. 선두 ploop는 출력 축을
독립적으로 완전히 순회해야 한다. nested/split/serial epilogue 등 증명하지 못한 구조는
`computation_reason`을 남긴다. 같은 accumulator의 반복 읽기와 원본의 여러 GEMM site를 구분한다.

[normalization](../../src/emit/provider/quack/recognition/normalization.rs)은 RMS/LN의 mean divisor,
variance 및 epsilon 위치와 softmax의 max correction을 확인한다. scalar constant가
다르거나 correction이 없다면 라이브러리에 맞게 식을 바꾸지 않는다.

Quack 내부의 `recognize_gemm`, `summarize`, `domain_stores`와
`recognition::rope::recognize`는 Quack matcher가 필요에 따라 호출한다. `domain_stores`는
완전한 선두 ploop 순회를 증명한 후에도 각 store와 원래 표현식을 따로 보관한다.
직선적인 store 본문을 다루며 sloop/mloop는 별도 의미 증명이 필요하다.

[RoPE helper](../../src/emit/provider/quack/recognition/rope.rs)는 paired view의
`x0*c-x1*s`, `x0*s+x1*c` 식과 conjugate 부호를 확인한다. 두 input view의
identity/stride/offset, pairing, 회전 축/크기, cos/sin view와 broadcast 축,
pass-through tail, GEMM producer 연결, 원본 operation/store를 반환한다.
인접 pair는 `[...,P,2]`, half pairing은 마지막 축의 연속된 두 slice를 인식한다.
IR에서 batch/head라는 이름을 추측하지 않는다. Table이 달라지는 축과 모든 접근
stride를 기록하고, provider가 API의 B/T/H 축으로 배치한다.
상세 지원/거절 범위는 [아래 RoPE 지원](#rope-의미-분석과-api-지원)를 따른다.

## 현재 인식과 연결

| Quack 진단 분류 | Quack 구현 | 현재 조건 |
| --- | --- | --- |
| `SingleGemm` | `gemm` | 완전한 2D GEMM, 동일 FP16/BF16 입력, K/N은 8의 배수 |
| `GemmEpilogue` | `gemm`, `gemm_add`, `gemm_act` | 하나의 bias 또는 residual, 선택적 ReLU/SiLU; arbitrary epilogue는 거절 |
| `GatedGemm` | `gemm_act(activation="swiglu", store_preact=False)` | 동일 입력의 gate/up GEMM과 `gate * sigmoid(gate) * up`, 같은 출력 shape |
| `RmsNorm` | `rmsnorm_fwd` | 전체 마지막 축의 mean-square, sqrt 내부 epsilon, 선택적 weight/bias |
| `LayerNorm` | `layernorm_fwd` | 명시된 mean subtraction과 population variance, 선택적 weight/bias |
| `Softmax` | `softmax_fwd` | IR에 max correction이 명시된 exp/sum 정규화 |
| `Other` 중 RoPE | `apply_rotary` / `rope_epi` | 아래의 view·dtype·전체 영역 계약을 통과한 경우 |
| 그 외 `Other` | 현재 일치 API 없음 | Triton fallback 유지; IR 자체의 오류라는 뜻은 아님 |

직접 GEMM 외에 완전한 K-loop와 여러 register store를 거치는 표현도 처리한다.
정규화 sum recurrence도 인식하지만 임의의 serial loop를 축약하지 않는다.
원본에 서로 다른 GEMM site가 두 번 쓰인 경우를 동일 식이라는 이유로 SingleGemm으로
합치지 않는다. 같은 accumulator를 SiLU에서 두 번 읽는 경우와 구분한다.

현재 adapter target은 single-GPU Hopper와 sm_120이다. Quack 라이브러리 전체의
하드웨어 지원표가 아니라 이 저장소 adapter의 범위다. 구현은 로컬 Quack 0.6.5의
공개 forward API signature를 기준으로 작성했다. import/JIT 실패는 후보 실패로 기록한다.

정규화/softmax는 2D input/output과 마지막 축 reduction으로 제한하고,
input/output dtype이 같아야 한다. weight/bias는 마지막 축 vector다.
임의 차원 flattening, batched GEMM, attention, backward, 두 GEMM의 직렬 연결,
reduction GEMM epilogue 및 임의 custom epilogue 생성은 아직 지원하지 않는다.
standalone SwiGLU activation 전용 kernel은 연결하지 않았다.

## RoPE 의미 분석과 API 지원

확인한 로컬 라이브러리는 **Quack 0.6.5**이며 소스 경로는
`target/quack-runtime-deps/quack`다. 최신 upstream 전체에 대한 지원 선언은 아니다.
검증 환경과 범위는 아래의 검증 기록에 정리한다. 개별 실행 로그는 로컬 작업 기록으로 보관한다.

| API | 확인한 라이브러리 계약 | Trinity adapter 범위 |
| --- | --- | --- |
| `quack.rotary.apply_rotary(x, cos, sin, seqlen_offsets=None, cu_seqlens=None, max_seqlen=None, interleaved=False, inplace=False, conjugate=False)` | 고정 `[B,T,H,D]` 또는 varlen 입력; cos/sin `[T_ro,R/2]`; FP16/BF16/FP32; `D<=512`, D/R은 8의 배수, `R<=D`; 두 pairing·부분 회전·conjugate 지원 | 고정 길이, 완전한 연속 logical view, 마지막 축 회전; table은 최대 한 token 축에서 변함. 명시된 table slice/broadcast만 사용. offsets/varlen/inplace는 미연결 |
| `quack.epilogue.library.rope_epi(A,B,*,out=None,tuned=True,**operands)` | `mode="acc_pair"`; `table`은 D와 같은 matrix에 cos/sin을 adjacent-N pair로 배치. B는 `[K,N]`. `rope_epi.gemm(..., epi_args=..., b_kn=...)` 저수준 API도 존재 | `rope_epi(A,B,out={"D":out},table=table,tuned=tuned)` 사용. 하나의 full-K 2D GEMM과 완전한 adjacent-pair forward rotation; FP16/BF16 동일 입력, K/N은 8의 배수. register producer만 흡수 |

실제 standalone API가 있으므로 GEMM을 만들어 우회하지 않는다. 지원 범위를 벗어난
RoPE도 의미 helper 결과와 API의 거절 사유를 구분한다. 예를 들어 half-pairing
projection은 RoPE로 인식되지만 이 adapter의 `rope_epi` 호출 후보가 되지 않는다.

인식하는 기본 형태는 다음과 같다. 이름은 의미 판정에 사용하지 않는다.

```text
half:     x0 = X[...,0:R/2], x1 = X[...,R/2:R]
adjacent: X의 view = [...,R/2,2], x0 = view[...,0:1], x1 = view[...,1:2]
Y = concat(x0*cos - x1*sin, x0*sin + x1*cos, component_axis)
partial half: concat(Y, X[...,R:D], last_axis)
projection: P = A @ B; 위 X의 identity가 P와 같고 view/producer 범위가 일치
```

- 선두 중첩 ploop를 유지하고 complete non-overlapping 축 순회를 증명한다.
- 원본 access는 plan에 유지한다. `RopePattern`의 first/second, input/output,
  cos/sin, rotation_axis, rotary_dimension, pairing, conjugate, table_axes,
  passthrough, projection, operations/stores가 의미 연결을 보존한다.
- Table이 변하는 축을 token으로 매핑한다. 나머지 두 broadcast 축을 B/H로 배치하는
  `order`와 원래 stride를 보관하고 역변환한다. IR 이름이 Q/K/rotary인지 검사하지 않는다.
- Standalone은 input permutation/contiguous copy, table 준비, output allocation/copy를
  `run` 안에서 수행한다. 원래 partial tail도 output copy에 포함한다.
- GEMM epilogue는 필요 시 operand alignment copy, cos/sin FP32 승격과 interleave를
  호출마다 수행한다. 승격은 저장된 값을 정확히 보존한다. 실행 비용 비교에 준비 비용도 들어간다.
- Standalone은 load/rotation FP32, input과 같은 storage dtype으로 최종 cast한다.
  GEMM+RoPE는 FP16/BF16 GEMM 입력 → FP32 accumulator/rotation → output dtype이다.
  중간 cast나 global/외부 projection store를 없애는 후보는 만들지 않는다.

현재 미지원: 서로 다른 store에 rotation 두 성분을 나눠 쓰는 식, partial interleaved
view, 마지막 축 이외의 pair 축, 증명하지 못한 비연속 view, computed cos/sin,
position/frequency 생성, varlen/position offset API, RoPE 앞의 sloop/mloop나 추가
epilogue, batched/head reshape projection. 현재 optimizer spec에는 sin/cos 전용 op가
없어 명시된 table 입력을 사용한다. 이런 제한은 지원 거절 사유를 남긴다.
Half/partial/conjugate projection의 Quack 연결, 별도 Q/K/V packing, cache 갱신 결합은
후속 matcher/API mapping 작업이다. 이웃 region fusion이나 기존 region 분할은 하지 않는다.

`Projection + RoPE + Attention`, 추가 외부 출력, 입력 갱신은 일부만 구현하는
RoPE API의 **전체 region 후보로 등록하지 않는다**. 모든 관측 store는 common facts에
남고, 현재 Quack adapter가 단일 출력/비변이 조건으로 거절한다.

## View, packing, dtype와 호출 비용

`TensorView { value, shape, strides, offset }`은 기존 contiguous backing에 대한
접근이다. 현재 whole-domain load, 상수 slice/element, squeeze, transpose/permute,
축 삽입을 표현한다. 계산된 GEMM operand나 증명되지 않은 local subtile은 거절한다.

SwiGLU는 IR이 두 GEMM이어도 후보로 만들 수 있다.

```text
Gate = X @ W_gate
Up   = X @ W_up
Y    = Gate * sigmoid(Gate) * Up
```

- gate/up이 같은 allocation의 interleaved view라는 증명이 있으면 `[K,2H]`로 재사용한다.
- 서로 다른 weight라면 `stack((gate,up), -1).flatten(-2)`를 호출마다 수행한다.
  training 중 weight 변경을 놓치는 영구 packing cache는 만들지 않는다.
- GEMM 입력은 정렬된 지원 stride를 유지한다. 필요하면 candidate 내부에서
  contiguous/alignment copy를 수행한다. 계산의 입출력 값과 IR view는 보존한다.
- 준비·packing·copy는 모두 `run` 안에 있으므로 후보 benchmark에 포함된다.
  packed GEMM이 항상 빠르다고 전제하지 않는다.

저장 dtype은 PhysicalPlan에서 받는다. GEMM 입력은 원래 FP16/BF16을 유지하며,
matcher가 임의로 FP32 입력을 낮춰서 Quack을 가능하게 만들지 않는다. public Quack
kernel의 수치 구현은 Quack이 담당하고, Triton과 bitwise 동일하다고 보장하지 않는다.
명시 cast가 들어간 epilogue는 현재 Quack matcher가 거절한다.
LayerNorm의 affine 인자는 API 요구에 따라 FP32로 승격한다. 저장값을 바꾸는 축소는 없다.

public norm/softmax API는 새 output을 반환한다. adapter는 이를 caller-owned output에
복사하므로 추가 allocation/copy가 있다. **한 region 후보라는 것은 한 번의 GPU launch라는
보장이 아니다.** GEMM wrapper처럼 직접 out을 쓰는 최적화는 이후 API 검토 대상이다.
max correction이나 epsilon을 IR에 새로 넣지 않는다.

## 선택과 남은 제한

`region_candidates(&plan)`은 source와 지원/거절 결과만 만든다. 라이브러리 설치,
GPU compile·실행·성능 확인을 뜻하지 않는다. `emit_python`은 다음 조건에서
영역 비교 프로그램을 만든다.

- program output 하나, 입력 mutation 없음, 고정된 shape/target;
- 적어도 한 region에 Quack 후보가 있고 각 region에 실행 후보가 존재;
- 비교할 region은 독립 reference가 있어야 함. Reference가 없는 region은 Triton 필요;
- cross-kernel split tuning parameter가 없음.

독립 PyTorch reference는 recognition이 보존한 원본 summary 또는 whole-domain ordered stores와
view로 생성한다. RoPE API 식을 reference로 복사하지 않는다. Register projection은
FP32로 보관하고 관측되는 store는 storage dtype을 따른다. `prepare`는 각 후보에
동일한 입력을 주고, 출력 정확성과 입력 비변경을 확인한 후 통과 후보만 측정한다.
JIT/초기화는 첫 실행에서 수행해 timing 앞에 둔다. 이후 실행은 선택을 다시 하지 않는다.
이 검사는 sample 입력에서의 검증이며 모든 입력의 수치 동치 증명이 아니다.
Launch boundary 인자는 `RegionFacts.global_values()`에서 받는다. Triton lowering이
실패해도 Quack의 raw-region 지원 검사는 실행되며, 독립 reference가 있으면 Quack만
있는 region도 비교 프로그램으로 만든다. 이 경우 Triton 거절 이유를 manifest에 남긴다.

Reference를 구성하지 못한 region은 원래 Triton을 실행하며
`comparison: not_performed`를 남긴다. 전체 비교 조건을 만족하지 못하면 기존
`triton_program` 경로로 가고 manifest에 영역별 후보·거절 정보를 보존한다.
프로그램별 mutation/multiple outputs 지원 확대, provider 공통 register rounding 계약,
weight packing cache, Native/CuTe까지의 통합 선택은 별도 작업이다.

## 검증 기록과 현재 테스트 범위

2026-09-26 커밋 정리 과정에서 미커밋으로 추가했던 pattern/Quack/RoPE/실행 파일 테스트와
Llama/SwiGLU fixture를 삭제했다. 해당 기능의 전용 회귀 테스트는 추후 다시 작성한다.
아래는 삭제 전 수행한 검증 기록이며 현재 checkout에서 테스트를 바로 재현할 수 있다는
뜻은 아니다. 기존에 커밋된 공통 plan·Triton 테스트는 유지한다.

- CPU mock으로 packing, norm 인자, launch 인자와 reference/호출 계약을 확인했다.
- PRO 5000 SM120, Quack 0.6.5, Triton 3.8.0에서 BF16/FP16 RoPE Quack 12개와
  Triton 10개 호출이 `rtol=.01, atol=.001` 기준을 통과했다. Half/adjacent,
  token 축 재배치/conjugate, partial half, projection과 N=48 Quack-only를 포함한다.
- 위 GPU 검증은 Quack GEMM `tuned=False`, Triton 첫 config만 사용했으며 성능 비교가 아니다.
  큰 Llama block의 독립 reference 정확성이나 모든 config/하드웨어 검증으로 확대하지 않는다.

개별 실행 로그와 작업 기록은 로컬에 보관하며 이 저장소의 배포 문서에는 포함하지 않는다.
