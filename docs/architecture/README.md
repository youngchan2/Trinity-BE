# Trinity Lowering Architecture

Trinity Lowering Repository는 선택된 연산 구현 및 실행 순서를 이용해 CUDA source와 컴파일 아티팩트를 생성합니다. 이 문서는 공개 인터페이스와 세 단계의 책임 경계를 안내합니다.

아래 공개 인터페이스는 Native CUDA 경로입니다. 공통 PhysicalPlan을 소비하는
Triton source/launch 경로와 제한된 Triton/Quack 후보 비교 경로도 별도로 존재합니다.
통합된 compile·benchmark·선택 경로가 모두 완성된 것은 아닙니다.
현재 작업 트리의 공통 계약은 [Planning](planning.md), provider별 진입점과 지원 차이는
[Triton provider](triton-provider.md#selection-boundary-and-current-limits)를 따릅니다.

![](../images/trinity-lowering-architecture.svg)

## 인터페이스

```rust
// text IR 기반 PhysicalPlan 구성
pub fn lower_ir(
    text: &str,
    config: &IrConfig,
) -> Result<Vec<PhysicalPlan>, IrError>;

// PhysicalPlan 수동 구성 Builder API
pub struct PhysicalPlanBuilder { /* ... */ }

// CUDA 소스 생성
pub fn emit(plan: &PhysicalPlan) -> Result<CudaSource, EmitError>;

// 생성된 소스 컴파일
pub fn compile(source: CudaSource) -> Result<CudaArtifact, CompileError>;
pub fn compile_with_config(
    source: CudaSource,
    config: &CompileConfig,
) -> Result<CudaArtifact, CompileError>;
```

## 구성

### Planning

[Planning](planning.md)

Text IR 또는 `PhysicalPlanBuilder` 입력을 받아, Tensor Binding, 연산, Loop와 실행 순서를 `PhysicalPlan`으로 구성합니다.

### Emission

[Emission](emission.md)

`PhysicalPlan`을 참조해 지원 가능 커널 구현을 선택하고, 구체적 실행 배치를 담은 `CudaSource`를 생성합니다.

### Compile

[Compile 소스](../../src/compile/mod.rs)

`CudaSource`를 공유 라이브러리와 메타데이터를 포함한 `CudaArtifact`로 변환해, 실제 실행 가능한 형태로 변환합니다.

## 탐색

| 확인할 내용                 | 시작 파일                                                                                                                                            |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 공개 API                    | [src/lib.rs](../../src/lib.rs)                                                                                                                       |
| Plan 구성·확정              | [plan/builder.rs](../../src/plan/builder.rs)                                                                                                         |
| 공통 emit·검증 진입점       | [emit/mod.rs](../../src/emit/mod.rs)                                                                                                                 |
| 공통 후보 입력 준비 | [emit/prepare.rs](../../src/emit/prepare.rs)                                                                                                         |
| Scheduled IR → 공통 Plan    | [plan/scheduled.rs](../../src/plan/scheduled.rs) |
| 공통 접근·storage 분석      | [analysis/physical.rs](../../src/analysis/physical.rs), [analysis/storage/mod.rs](../../src/analysis/storage/mod.rs) |
| 소스·자원 반환 타입         | [compile/source.rs](../../src/compile/source.rs) |
| Native 구현 수집·자원 계약  | [emit/collect.rs](../../src/emit/collect.rs), [emit/native/interface.rs](../../src/emit/native/interface.rs) |
| Native 본문 결합            | [emit/combine/mod.rs](../../src/emit/combine/mod.rs) |
| Symbol·Code 표현            | [emit/native/code.rs](../../src/emit/native/code.rs) |
| 실행 방식 선택·배치 연결    | [emit/cuda/execution.rs](../../src/emit/cuda/execution.rs)                                                                                           |
| 논리 검증                   | [emit/cuda/validation.rs](../../src/emit/cuda/validation.rs)                                                                                         |
| 공통 Provider 후보 계약     | [emit/provider.rs](../../src/emit/provider.rs) |
| Native phase·CTA·binding 계약 | [emit/native/mod.rs](../../src/emit/native/mod.rs), [emit/native/bindings.rs](../../src/emit/native/bindings.rs) |
| 공통 syntax / logical dtype | [analysis/ir.rs](../../src/analysis/ir.rs), [analysis/dtype.rs](../../src/analysis/dtype.rs) |
| Triton program provider     | [emit/provider/triton/program.rs](../../src/emit/provider/triton/program.rs) |
| Triton lowering·codegen     | [triton/lowering/mod.rs](../../src/triton/lowering/mod.rs), [triton/codegen/mod.rs](../../src/triton/codegen/mod.rs) |
| 후보 발견·Python 실행 구성  | [emit/candidate.rs](../../src/emit/candidate.rs), [emit/program/mod.rs](../../src/emit/program/mod.rs) |
| CUDA 소스·Body 렌더링       | [emit/cuda/render.rs](../../src/emit/cuda/render.rs)                                                                                                 |
| compile/link와 artifact     | [compile/cuda/mod.rs](../../src/compile/cuda/mod.rs), [compile/cuda/artifact.rs](../../src/compile/cuda/artifact.rs)                                 |
