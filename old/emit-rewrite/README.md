# 이전 emit/CUDA 구현 — 참조 전용

기존 emitter와 CUDA 코드 생성 구현을 활성 경로에서 분리했다. 이 코드는 빌드하거나
실행 fallback으로 호출하지 않는다. 새 provider와 renderer는 활성 소스에 작성한다.
**Python 소스·binding·테스트·예제는 이 디렉터리에 보존하지 않는다.**

## 보존 범위

`snapshot/` 아래 경로는 원래 `crates/trinity-lowering/` 기준 경로다.

| 경로 | 내용 |
| --- | --- |
| `src/emit/` | Prepare, body/phase, 주소·접근·region 검증, Streamed/Persistent 실행 분석, renderer·템플릿 |
| `src/implementation/` | 기존 CUDA 구현과 enumeration·schedule·fusion 연결 계약 |
| `src/fusion/`, `src/plan/rewrite.rs` | Emit 검증에 의존하던 fusion 엔진과 statement rewrite |
| `src/plan/loops.rs`, `src/tests.rs` | 제거한 C++ index 렌더링과 emit을 호출하던 Rust 회귀 참조 |
| `tests/` | 기존 Rust 테스트, support, Loop IR fixture |

커밋이나 index blob이 아닌 당시 작업 트리를 보존했다. Staged·unstaged 수정과 Git에
추가되지 않은 소스를 포함하며, 이미 삭제된 파일은 복원하지 않는다.
[MANIFEST.json](MANIFEST.json)에 파일별 크기·SHA-256,
[WORKTREE_STATUS.txt](WORKTREE_STATUS.txt)에 이 범위의 보존 당시 Git 상태를 기록한다.
가상환경, Python, native runtime, compiler, 빌드 설정, 외부 라이브러리는 보존 범위가 아니다.

## 활성 소스의 현재 경계

- `src/emit/`에는 새 진입점 `mod.rs`만 있다. `emit()`은 `EmitError::Unavailable`을
  반환하며 Python 호출에는 `NotImplementedError`를 전달한다.
- 이전 `src/emit/cuda/`와 `src/implementation/cuda/`는 활성 경로에서 제거했다.
- 격리 직후 추출했던 `implementation/schedules` 및 자동 확장 호출부는 이후
  [old/plan-rewrite](../plan-rewrite/README.md)에 별도로 보존하고 활성 경로에서 제거했다.
  현재 `implementation/definitions`에는 ID와 후보 열거만 남아 있다. `schedule` 계약은 없다.
- Loop IR의 주소·bounds 검증은 `src/plan/address.rs`가 담당한다.
- `CudaSource`와 allocation metadata는 `src/compile/source.rs`가 소유한다.
  이전 Body/Execution 검사 API는 제거했다.
- Native ABI header는 `src/native/abi.h`로 이동했다. Native runtime은 유지한다.
- Built-in fusion rules는 비활성화했다. 빈 rule 목록은 원본 plan을 유지한다.
  Rule 적용 요청은 새 검증기 전까지 `FusionError::Unavailable`로 거부한다.
- Rust의 emission 전용 회귀는 보존본에 남기고 planning 검증은 활성 테스트에서 유지한다.
  Python 파일은 원래 위치에서 필요한 연결만 수정한다.

현재는 새 CUDA 소스를 생성할 수 없다. 소스 생성·GPU 실행이 필요한 기존 Python 예제와
테스트는 provider/renderer 이식 후 다시 연결한다. 기존 emit을 복사본에 위임하지 않는다.

## 책임과 호출 관계

```text
이전: emit → prepare → body/접근/배치/자원 준비 → render → CudaSource
                         ↑ fuse의 후보 검증

현재: lower_ir / 명시적 Builder → 이름·operand 정규화 → PhysicalPlan 검증
      whole-tensor Candidate lower → ExplicitProgramRequired

새 경로: PhysicalPlan → scope → KernelProvider → 합성·메모리·실행 계획
                      → template rendering → compile → native runtime
```

기존 `body/builder.rs`의 GEMM/SIMT 생성은 provider로, shared 배치·register 전달은
메모리 계획과 scope 합성으로 분리한다. `Body::sequence()`의 phase 평탄화와 CUDA
재토큰화는 순서 있는 scope 및 명시적 symbol/template binding으로 대체한다.
Region·coverage·수명·Persistent dependency 검사는 새 분석 단계에서 복원할 책임이다.

## 후속 단계

현재 구현 계획은 [PLAN.md](../../../../docs/draft/PLAN.md)의
**Emit 재구성 단계별 계획 — 2026-09-16**에서 관리한다.

1. 명시적 Plan 계약으로 전환하고 자동 확장·schedule 계약 제거 (완료).
2. 입력 경로별 공통 검증 통합과 accumulator·provider 입력 계약 확정.
3. Scope·Kernel·KernelProvider 계약과 우선순위 fallback.
4. Native pointwise 하나의 template 기반 Streamed 경로.
5. Region·의존성·local 수명·메모리·자원 분석.
6. Hopper GEMM/CuTe와 phase fusion·중간 store 제거.
7. 나머지 연산·통신과 Native Persistent 실행.
8. Streamed opaque artifact·공용 cubin launcher·실제 Triton provider.

`plan`과 Python은 기존 위치에서 수정한다. Opaque는 우선 Streamed만 대상으로 하며,
Native `.so` 경로를 유지하고 kernel별 C++ wrapper는 생성하지 않는다.
Persistent opaque 호출·분할·fallback은 후속 설계다.

## 보존본 무결성

`MANIFEST.json`의 각 `path`에 대해 `snapshot/<path>`의 크기와 SHA-256을 확인한다.
보존 파일 목록은 manifest와 일치해야 한다. 활성 소스는 이후 단계에서 변경되므로
무결성 비교 대상이 아니다. Python 파일과 `src/python.rs`는 보존본에 없어야 한다.

격리 검증 결과: 82개 보존 파일이 격리 전 SHA-256과 일치한다. 두 활성 crate의 Rust
테스트 94개, native ABI의 CPU fixture 테스트 1개, fmt 및 clippy(`-D warnings`)가 통과했다.
Old는 Cargo target에 포함되지 않으며 활성 코드에 이전 CUDA emitter 호출이 없다.
`ARCHITECTURE.md`는 변경하지 않았다. GPU 실행 및 기존 Python emission 테스트는 실행하지 않았다.
