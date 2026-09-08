# Rust analyzer와 기존 Trinity 출력 규약

목표는 분석을 개선하면서 **`Trinity/backend/codegen`이 생성하는 코드 형식과 실행
인터페이스를 유지하는 것**이다. Rust emitter는 원본의 kernel header, autotune,
loop/index/load/store, inline expression과 wrapper 규약을 따른다.

## 분석 결과와 emitter의 연결

```text
IrNode / S-expression
    ↓ analyzer::analyze / analyze_text
ProgramAnalysis
  원본 IR, TensorId/ScopeId/AccessId
  그룹별 ValueExpr, 정확한 lexical loop binding, 접근별 view layout
    + Options { shapes, symbols }
    ↓ triton::lower
TritonPlan
  TileAccess: 너비, shape, 시작점, extent, stride, loop-end
  TensorPlan: register / global / materialized, initialization, export_scope
  KernelPlan.register_accesses: 어떤 load/store가 지역 변수에 연결되는지
    ↓ TritonPlan::emit
기존 형태의 kernel_N + TENSOR_PARAMS / BLOCK_PARAMS + forward(...)
```

동일 tensor도 접근마다 분류가 다를 수 있다. 예를 들어 FFN 177의 `FF1a`는 첫 번째
`p` loop 안의 `k` loop에서는 지역 accumulator이고, 누적이 끝나면 global에 저장한다.
다음 `p` loop의 read는 global에서 읽는다. Emitter가 tensor 이름만으로 다시 분류하지 않는다.
`attn_O_norm`은 정의된 `k` 안에서만 사용되므로 초기화나 루프 밖 keepalive를 추가하지 않는다.

FFN 4의 normalization처럼 여러 tile을 sibling loop 사이에서 보존하는 값은 기존 backend의
cross-sloop global tensor에 해당한다. 호출자가 전달한 fp16 tensor를 사용한다.
이전 시도의 프로그램별 fp32 scratch, 자동 tensor 할당, program 0 저장 선택은 제거했다.

## 생성 코드 규약

- `kernel_0`, `kernel_1`, ... 이름과 기존 loop 순서/경계를 유지한다.
- `attn_O1`, `FF1a` 같은 IR tensor 이름을 지역 변수에 사용한다.
- Global tensor는 `NAME_ptr`, `NAME_stride0: tl.constexpr`, ... 인자로 받는다.
- Loop는 `n`, `k`, `p`와 `BLOCK_N`, `BLOCK_K`, `BLOCK_P`를 사용한다.
  실제로 shadowing되는 이름만 scope별로 구분한다.
- Global load는 `offset_N`, `mask_N`, `temp_N`으로 생성하고 계산식은 inline으로 조합한다.
  같은 scope에서 재사용 가능한 load는 공유하며 store/loop 경계에서 cache를 무효화한다.
- 원본과 같은 autotune 후보 정책(축 크기에 맞춘 power-of-two 후보, 최대 16개 조합)을 사용한다.
  Literal step은 wrapper에서 `BLOCK_*=값`으로 전달한다.
- `TENSOR_PARAMS`, `BLOCK_PARAMS`, 명시적 `forward(TensorA, TensorB, ..., block_k=16, ...)`를
  생성한다. Wrapper는 인자로 받은 tensor와 stride를 kernel에 전달하고 `None`을 반환한다.
- Output 및 intermediate 할당은 기존 benchmark/caller의 책임이다.

Global 저장과 dot 입력은 fp16, 지역 계산/reduction/accumulator는 fp32이다.
작은 M/N/K의 GEMV는 원본 `dot_lowering.py`처럼 fp16 곱셈 뒤 fp32 sum을 사용한다.

분석상 필요한 차이는 유지한다: 잘못된 루프 밖 참조 제거, 실제 초기화/저장 scope,
접근별 view shape와 tile width, tail reduction identity/mask.
Validity는 축별 predicate로 보관하며 mask tensor를 `tl.sum`으로 축약하지 않는다.
`tl.load(..., other=0)`의 무효 위치는 이미 0이므로 dot 앞에서 다시 `tl.where`로 감싸지
않는다. Dot/reduction은 축약하는 축의 validity만 소비하고 나머지 축의 predicate는
결과에 전달한다. `exp`처럼 0을 다른 값으로 바꾸는 연산 뒤의 sum/dot, max/min의
tail identity에는 필요한 `tl.where`를 유지한다. Store는 자체 접근 mask를 사용하며,
register 대입마다 결과 전체를 다시 mask하지 않는다.

## 사용

`Options.shapes`에는 사용되는 모든 tensor의 구체적인 shape를 넣는다. `symbols`는
분석할 때 사용할 `tile_k` 등의 값이다. 출력의 symbolic loop step은 고정 숫자로 치환하지
않고 원본처럼 `BLOCK_*`와 autotune으로 남긴다. 원본 규약과 동일하게 wrapper의
`block_*` 인자는 유지되지만 autotuned kernel의 실제 block 값은 autotune이 정한다.

```shell
cargo run --locked --offline --example emit_triton -- program.ir shapes.txt generated.py tile_n=128 tile_k=64 tile_p=64
```

`shapes.txt`는 각 줄이 `NAME dim dim ...`인 파일이다. IR이 번호 목록이면 출력 인자를
디렉터리로 사용하여 `kernel_<IR ID>.py`를 생성한다. `dummydata`는 완성된 IR이 아니므로
제외해야 한다. 라이브러리에서는 `lower(analysis, options)?.emit()`으로 연결한다.
Rust optimizer의 native metadata adapter는 아직 별도 연결이 필요하다.

재생성 결과는 `generated_kernels/{ffn_llama,ffn_falcon,vanilla_llama,vanilla_falcon}/`에 있다.
파일은 검토할 수 있도록 프로젝트 내부에 두며 Git에는 generated artifact로 제외한다.

## 원본 출력과 비교

`tests/fixtures/triton_reference/`에는 **원래 Python backend로 생성한** FFN 177과 vanilla
591 Llama 출력과 vanilla 591 Falcon 출력이 있다. `tests/compare_triton_reference.py`는 다음을 비교한다.

- Benchmark metadata, kernel/forward 인자와 autotune 설정: AST 일치.
- Wrapper의 kernel 호출과 stride 전달: AST 일치.
- Kernel의 loop 구조, 계산식, 주소와 mask: temporary alias를 펼쳐서 비교.

공백/주석/임시 변수 이름과 동일 scope의 독립적인 zero 초기화 순서는 무시한다.
알려진 두 차이만 별도 정규화한다: 불필요한 `x = x + 0` keepalive 제거와,
이미 fp16 pointer에 저장되는 값에 fp16 cast를 명시하는 차이이다.
이 비교를 소스의 byte 단위 동일성이나 GPU 정확성 검증이라고 부르지 않는다.

## 검증 및 남은 범위

```shell
cargo fmt -p trinity-lowering -- --check
cargo test -p trinity-lowering --locked --offline
cargo clippy -p trinity-lowering --all-targets --locked --offline -- -D warnings
cargo build --locked --offline --example emit_triton
python3 tests/compare_triton_reference.py target/debug/examples/emit_triton
python3 tests/triton_cpu.py target/debug/examples/emit_triton
python3 tests/check_triton_source.py generated_kernels/ffn_llama generated_kernels/ffn_falcon generated_kernels/vanilla_llama generated_kernels/vanilla_falcon
```

Llama/Falcon 각각 FFN 839개와 vanilla 170개, 합계 2,018개를 생성하고 Python 문법 및
kernel 변수 정의 관계를 검사했다. NumPy 기반 13개 smoke test는 작은 GEMM, padding,
누적 초기화, global materialization, cache 갱신과 축소한 FFN 4/177·vanilla 591을 실행한다.

추가된 두 tail 회귀 테스트는 exp 이후 dot, dot/exp 이후 sum에서 padding 원소가
연산에 섞이지 않는지 확인한다. Vanilla Llama/Falcon 591은 원본 shape와
`Trinity-BE/backend/.venv`의 PyTorch 2.8.0/Triton 3.4.0으로 RTX PRO 6000 Blackwell에서
컴파일·autotune·실행했고, PyTorch 기준 출력 및 KV cache 갱신 비교를 통과했다.
두 경우 출력 최대 절대 오차는 7.63e-6이다. 이전 FFN Llama 177/307/1019/1230 GPU
smoke test도 통과했으며 이번 재생성에서 FFN Llama 소스는 변경되지 않았다.
**전체 코퍼스의 GPU 실행·수치 정확성·성능을 검증한 것은 아니다.** 원본의 global tensor
계약과 schedule을 유지하며, 별도의 프로그램 간 동기화나 buffer 관리 체계를 추가하지 않는다.
현재 정적이고 비어 있지 않은 loop 범위, 최대 3개 중첩 ploop, 2D/3D dot을 지원한다.
일반적인 alias/affine region 분석, 동적 shape, mloop, concat, cast는 후속 범위다.
View는 명시적 layout에 따른 contiguous reinterpretation으로 한정한다.
