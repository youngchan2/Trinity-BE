# Trinity analyzer 설계 — Python 분류 함수에서 Rust 구현으로

상태: **Rust 수집 → fp16 fallback 분석 계획 → Triton emitter/wrapper 구현**.
구체적인 현재 API·지원 범위·검증 결과는 [TRITON_FALLBACK.md](TRITON_FALLBACK.md)에 기록했다.
아래 내용은 최초 설계 기록이며, 모든 제안 필드가 그대로 구현되었다는 뜻은 아니다.
Python generator와 profile은 변경하지 않았다.
Inductor 조사와 현재 구현 범위는 [INDUCTOR_ANALYSIS.md](INDUCTOR_ANALYSIS.md)에 기록했다.

## 1. 확인한 실패와 설계의 출발점

2026-09-05 14:55:50 UTC에 `Trinity-BE/backend/profile_result/ffn/ffn_llama.json`을 읽었다.

- 기록 839개: 유한 execution time 835개, Infinity 4개. 모두 `benchmarked: true`.
- 839개 모두 profile의 IR, evaluation 목록의 IR, 생성 파일 헤더가 일치했다.
- 생성 파일 839개는 읽는 동안 크기/mtime이 변하지 않았으며 Python 구문 분석을 통과했다.
- 생성 모듈의 kernel 개수: 2개인 후보 88개, 3개 173개, 4개 235개, 5개 343개.
- profile SHA-256: `a0379ccd12f735f2176e9d54226f60f81c744c007b190ec19df2e9b486f01ee1`.
- evaluation IR SHA-256: `0feffe3052fee9726e9dbed71e23fb3c105ceef5645e6b0b423a4e86759f9644`.

| IR ID | evaluation 파일 행 | 실패 함수 | 잘못 추가된 루프 밖 참조의 소스 행 |
|---|---:|---|---|
| 177 | 178 | `kernel_2` | 172, 180 |
| 307 | 308 | `kernel_3` | 198, 206 |
| 1019 | 1020 | `kernel_1` | 141, 149 |
| 1230 | 1231 | `kernel_2` | 167, 175 |

evaluation 파일은 `Trinity-BE/backend/evaluation/ffn/llama_ffn_cost6_kern5_wo_scheduler2.txt`,
소스는 `Trinity-BE/backend/generated_kernels/ffn_llama/kernel_<IR ID>.py`이다.
네 오류 모두 `NameError('attn_O_norm is not defined')`이다.

```text
for p:
    for k:
        attn_O_norm = normalize(...)
        FF1a = update(FF1a, attn_O_norm, ...)
        FF1b = update(FF1b, attn_O_norm, ...)
    attn_O_norm = attn_O_norm + 0.0   # emitter가 추가
attn_O_norm = attn_O_norm + 0.0       # emitter가 추가
```

원본 IR에서 `attn_O_norm`의 정의와 모든 사용은 동일한 안쪽 k-loop 안에 있다.
해당 k-loop 밖에서는 이 지역 값이 필요하지 않다.

기존 analyzer를 직접 실행하면 네 사례 모두 다음과 같다.

- `identify_accumulators`: `attn_O_norm`을 포함하지 않는다.
- `identify_cross_sloop_tensors`: `attn_O_norm`을 포함하지 않는다.
- `identify_sloop_intermediate_tensors`: `attn_O_norm`을 포함하지 않는다.
- emitter 내부 `count_sloops_using_tensor('attn_O_norm')`: **2**를 반환한다.

직접 원인은 `codegen/loops.py:209-284`의 별도 분석이다. `check_loads`는 자식
SLOOP를 건너뛰지만, 전달받은 body 자체가 SLOOP이면 그 안으로 들어간다. 따라서
`p.body == sloop k`인 경우 k 내부의 사용을 p와 k에서 중복 집계한다.
`find_stored_tensors`에도 같은 형태의 경계 문제가 있다. 이에 따라 불필요한
`+ 0.0`이 루프 밖에 생성되고, Triton이 해당 지역 값을 해석하지 못한다.

이 문제의 분석상 해결은 `attn_O_norm`을 루프 종료 후 참조 대상에서 제외하는 것이다.
이 값에 kernel-top zeros를 추가하는 것은 필요 없는 수명을 만드는 처리다.
`+ 0.0` compiler workaround 자체는 target emitter의 책임이며, analyzer에는
workaround용 텐서 분류 대신 실제 scope/사용 관계를 제공한다.

유한 execution time은 실행·측정 성공의 기록이다. 수치 정확성의 증거로 취급하지 않는다.
이번 조사에서는 GPU 실행이나 수정 후 컴파일을 수행하지 않았다.

## 2. 범위와 기본 자료

기존 AST의 `seq/ploop/sloop` 순서를 유지하며 analyzer 결과를 별도로 붙인다.
현재 코퍼스의 최상위 독립 sloop도 기존 경계 처리에 포함한다.
첫 구현은 `trinity-lowering`의 작은 analyzer 모듈로 시작한다. 전체 PhysicalPlan을
Triton 문장 단위로 확장하거나 새로운 scheduling/fusion 탐색을 추가하지 않는다.

식별자는 다음 정도면 충분하다.

- `TensorId`: 텐서 이름을 intern한 ID. 이름은 진단과 소스 출력에 보존한다.
- `KernelId`, `ScopeId`: 커널과 실제 루프/블록 발생 위치. 동명 k-loop를 구분한다.
- `StmtId`, `AccessId`: 원본 문장과 개별 load/store 발생 위치.
- 지역 값의 정의는 해당 store의 AccessId로 참조한다. 별도의 SSA IR 변환은 요구하지 않는다.

grouped tensor의 접근은 원래 문장과 항목 순서를 보존하면서 텐서별로 수집한다.
`(Q1,K1,V1)`의 self-load는 각 대상에 대응시켜야 한다.

## 3. 필요한 set

set은 빠른 조회와 진단을 위한 파생 결과다. 의미의 근거는 다음 절의 접근/의존/계획
레코드에 둔다. emitter가 set을 임시로 넣고 빼며 의미를 바꾸지 않는다.

| set | 범위 / 원소 | 의미 |
|---|---|---|
| `declared_inputs`, `declared_outputs` | program / TensorId | IR의 외부 표기. 실제 read/write와 구분 |
| `reads`, `writes` | kernel / TensorId | 해당 커널의 실제 load/store 대상 |
| `mutated_inputs` | program / TensorId | input으로 표기되었으며 실제 write가 있는 텐서 |
| `cross_kernel_tensors` | program / TensorId | 앞 커널의 정의를 뒤 커널이 소비하는 텐서 |
| `global_tensors` | kernel / TensorId | 최종 global load/store 계획 때문에 포인터 인자가 필요한 텐서 |
| `local_values` | kernel / 정의 AccessId | 지역 변수로 나타낼 값. 같은 텐서의 여러 정의를 구분 |
| `loop_carried` | loop / 누적 또는 recurrence 레코드 ID | 다음 반복이 이전 반복의 값을 필요로 하는 갱신 |
| `live_in` | scope / 값의 공급원 참조 | scope 내부의 정의 전에 외부에서 공급되어야 하는 값 |
| `live_out` | scope / 정의 참조 | 해당 scope 종료 후, 덮어쓰기 전에 실제로 필요한 값 |

`global_tensors`와 `local_values`에는 같은 텐서에 관련된 항목이 함께 존재할 수 있다.
예를 들어 FFN 388의 FF1a는 지역에서 누적한 후 global에 저장한다.

`loop_carried`는 다음 반복을 위한 값이고, `live_out`은 해당 scope 종료 뒤에 필요한 값이다.
loop 결과로 내보낼 지역 값은 scope 안에서 정의된 값 중 live_out에 포함되는 값이다.
반복 중 새로 정의되지 않는 외부 입력까지 loop 결과로 내보내지는 않는다.
global로 보존할 값은 그에 대응하는 store action이 소비자로 연결된다.

`defined_before[point]`는 분석 중 사용하는 집합/조회다. 각 지역 사용이 동일 반복에서
앞선 정의, 명시된 초기화, 혹은 연결된 loop-carried 값으로 공급되는지 확인한다.
루프의 zero-trip 가능성과 Triton의 loop 결과 표현에 필요한 초기 정의도 점검한다.
근거 없는 read-before-write를 자동 zeros로 해결하지 않는다.

다음은 독립적인 최종 set으로 만들지 않는다.

- `cross_sloop_tensors`: 관계 조회/디버그 요약으로 남기되 global·zeros 판단의 근거로 단독 사용하지 않는다.
- `kernel_accumulators`: 기존처럼 external/global 여부가 섞인 이름 집합을 제거한다.
- `nested_accumulators`, `zero_init_tensors`: reset/init 계획의 조회 결과로 대체한다.
- `fp32_tensors`: 식의 계산 dtype과 저장 dtype을 별도 정보로 다룬다.
- `stored_accumulators`: store action 계획으로 대체한다. "이미 저장했다"는 출력 bookkeeping은 emitter에만 둔다.

## 4. set이 가리킬 상세 정보

### 4.1 ScopeInfo / AccessInfo — 사실 수집

`ScopeInfo`는 parent, kernel, kind, 원본 노드, 순서가 있는 자식 문장, loop 변수·bound·step을 담는다.
각 사용에는 가장 가까운 scope 하나를 연결하고, 조상 scope 관계는 parent를 따라 조회한다.
조상에 포함된다는 사실을 독립적인 사용 횟수로 더하지 않는다.

`AccessInfo`는 다음을 담는다.

- 대상 TensorId, read/write, AccessId, StmtId, ScopeId, 원본 위치.
- 원본 index와 loop 변수의 binding. `tile(k)`의 k가 어느 loop인지 명시한다.
- 접근 region: fulltile/tile/elem/const_tile, 시작·너비, 의존하는 loop ID.
- 지역 tile shape, padding 후 shape와 유효 범위. 전역 tensor shape와 구분한다.

동일한 변수명/인덱스 문자열만으로 같은 타일이라고 판단하지 않는다.
부분 접근의 관계는 same/disjoint/overlap/unknown으로 남길 수 있으며, 같은 지역 값으로
치환하려면 해당 read의 값을 실제로 제공한다는 근거가 있어야 한다.

### 4.2 ReadSource / Dependence — 어느 정의를 읽는가

각 load를 외부 메모리, 앞선 store가 만든 값, loop-carried 값에 연결한다.
관계에는 producer와 consumer, 접근 region, 넘는 kernel/scope, 같은 타일의 유지인지
여러 타일의 보존인지, 미해결 조건을 기록한다.

원본 순서대로 RHS의 load를 먼저 처리하고 그 후 store 정의를 등록한다.
이전 커널에서 읽어 온 값을 현재 커널에서 다시 쓰는 경우도 추적해야 하므로,
`tensor -> 마지막 writer kernel` 하나로 전체 프로그램의 관계를 대체하지 않는다.

첫 구현에서는 원본 seq/loop 트리를 순회하는 구조화된 분석으로 처리한다.
별도 일반 CFG나 범용 alias framework를 추가하지 않는다.

### 4.3 RecurrenceInfo — 누적과 일반 갱신

대상 접근 region, update store, 이전 값을 읽는 load, carry loop, update 종류,
초깃값의 출처, reset 위치, 완료 후 소비자를 담는다.
shape와 accumulator dtype도 기록한다.

- `T = T + contribution`의 self-load는 같은 region의 이전 값인지 검사한다.
- 곱하기 1의 위치, add 피연산자 순서와 grouped tensor 표현 차이를 처리한다.
- `O = O / C_sum` 같은 갱신은 additive reduction과 구분한다.
- RHS의 loop 의존은 load index뿐 아니라 scalar 변수와 앞선 지역 정의의 의존도 고려한다.
- 현재 IR의 additive accumulator implicit-zero와 nested reset은 명시적인 legacy 규칙으로 둔다.
  각 결정에 적용한 규칙을 기록하며, 일반 self-load만으로 zero-init을 추론하지 않는다.

### 4.4 InitAction / LoadAction / StoreAction — emitter가 실행할 결정

초기화와 저장에는 순서가 있으므로 이름 set 대신 순서가 있는 action 목록을 사용한다.
위치는 `before(node)`, `loop_body_entry(loop)`, `after(loop)` 등 AST상의 지점으로 표현한다.

| 계획 | 필요한 정보 |
|---|---|
| `InitAction` | 대상 지역 값/carry, 실행 위치, shape, dtype, zero 또는 명시된 초기값, 근거 |
| `LoadAction` | 원본 load, 지역 정의/carry/global 중 공급원, 접근 region, shape/dtype/cast |
| `StoreAction` | 원본 정의/최종 carry, 소비할 값, 저장 위치, 목적 텐서·region·dtype, 필요 이유 |

IR의 store는 지역 값 갱신으로 lowering될 수 있다. global store action은 실제 외부 가시성이나
뒤쪽 소비자의 materialization 요구가 있을 때 계획한다. accumulator라서 항상 저장하지 않는다.
header와 wrapper의 tensor 인자는 최종 global action들에서 구한다.

global 필요 이유는 output 가시성, input mutation, 커널 간 전달, 여러 타일 보존으로 구분한다.
같은 커널 안의 global scratch는 ploop 프로그램 사이 주소가 겹치는지도 기록한다.
FFN 177의 FF1a/FF1b는 여러 타일 보존이 필요하지만 저장 주소에 n이 없어 ownership 검토도 필요하다.
"global 필요"와 "현재 global 주소가 적법"은 각각 확인한다. 미확정이면 원인과 함께 남긴다.

### 4.5 ValueInfo / Cast 정보 — shape와 precision

식/지역 정의별 tile shape, 계산 dtype, reduction/accumulator dtype을 저장하고,
외부 텐서별 storage dtype과 각 사용·저장 경계의 cast를 연결한다.

첫 정책은 현재 경로를 기준으로 fp32 지역 계산·누적과 fp16 dot operand 변환을 명시한다.
storage dtype은 입력 계약으로 받고, legacy 실행 경로의 fp16 가정도 명시한다.
예를 들어 FFN 244의 attn_O3는 fp32 누적 후 fp16 global store를 거치므로
다음 kernel에서도 통계값을 fp32로 유지하려면 runtime 텐서 dtype 계약도 함께 바뀌어야 한다.

## 5. 대표 사례의 기대 결과

| 사례 | loop-carried | loop 종료 뒤 필요한 값 / 저장 | 초기화 |
|---|---|---|---|
| 실패 177/307/1019/1230의 attn_O_norm | 없음 | k 밖 지역 참조 없음, global store 없음 | 없음, k 안 대입으로 정의 |
| FFN 4의 FF1a/FF1b | GEMM k | 같은 p 반복의 FF2 계산까지 지역 유지 | 각 p 반복에서 k 진입 전 0 |
| FFN 177의 FF1a/FF1b | GEMM k | 첫 p 루프의 각 타일을 보존, 뒤 p 루프가 재사용 | 각 p 반복에서 k 진입 전 0 |
| FFN 388의 FF1a/FF1b | GEMM k | 다음 kernel에 전달하는 global store | 해당 k 진입 전 0 |
| FFN 4의 FF2 | p | p 완료 후 output store | p 진입 전 0 |
| vanilla 591의 C_exp | 없음 | 같은 p 반복 안에서만 소비 | 없음 |
| vanilla 591의 O | p의 additive update | p 뒤 나눗셈, transform, O2 저장으로 연결 | 누적 시작 전 0; 나눗셈 전 reset 없음 |
| vanilla 543의 C/C_exp/C_div | 없음 | 별도 p 루프 사이에서 여러 타일 보존 | 대입 전에 무조건 zero-init하지 않음 |

FFN 4의 attn_O_norm은 위 실패 사례와 다르다. 한 k-loop가 여러 타일을 만들고
뒤의 다른 k-loop가 읽으므로, 이름이 같아도 동일한 local-only 분류를 적용하지 않는다.

## 6. 기존 Python 함수와 이행 순서

| 기존 함수 | 새 역할 |
|---|---|
| `collect_tensors`, `collect_tensor_usage`, `collect_intermediate_tensors` | ScopeInfo/AccessInfo 및 read/write 요약 수집 |
| `identify_cross_kernel_tensors` | 순서와 정의를 보존하는 ReadSource/Dependence에서 파생 |
| `identify_cross_sloop_tensors`, `identify_sloop_intermediate_tensors` | scope 관계, loop-carried, live_out 조회로 대체 |
| `identify_cross_sloop_memory_tensors` | read 공급 가능성과 필요한 타일 수명에서 materialization 결정 |
| `identify_accumulators`, `find_nested_sloop_accumulators` | RecurrenceInfo 및 명시된 init/reset 위치 결정 |
| `generate_intermediate_allocations`, `generate_*accumulator_init/stores` | InitAction/StoreAction으로 분석과 출력 분리 |
| `loops.py` 내부 사용 횟수 재분석 | 제거. analyzer의 실제 loop 결과 정보를 소비 |

1. Python 함수의 관찰 결과와 이 문서의 기대 결과를 fixture로 고정한다.
2. Rust에서 ScopeInfo/AccessInfo와 reads/writes부터 구현한다. **완료**: 직접 scope의 접근,
   동명 loop binding, grouped tensor 대응, 커널별 요약과 원본 트리 보존을 구현했다.
3. read의 정의 연결, loop-carried/live_out을 계산한다. 네 실패 사례에서 불필요한 loop 밖 사용이 없어야 한다.
4. FFN 4/177/388로 지역 유지·여러 타일 보존·커널 간 전달을 분리하고 init/load/store 계획을 만든다.
5. shape/dtype/cast 정보를 연결하고 emitter가 확정된 계획만 소비하게 한다.

회귀 검증에는 같은 tensor 이름의 재정의, 같은 이름의 서로 다른 loop, next-iteration carry와
loop exit 사용의 구분, input mutation, zero-trip/초기값의 의미를 포함한다.
코퍼스의 finite/Infinity 결과는 컴파일·실행 회귀 기준으로 사용하며 수치 검증은 별도로 수행한다.
실제 Triton 컴파일 단계에서 네 NameError의 소멸과 정상 사례의 유지도 확인해야 한다.
현재 계획 단계에서는 그 검증을 완료했다고 표시하지 않는다.
