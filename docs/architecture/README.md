# Trinity Lowering Architecture

Trinity Lowering Repository는 선택된 연산 구현 및 실행 순서를 이용해 CUDA source와 컴파일 아티팩트를 생성합니다. 이 문서는 공개 인터페이스와 세 단계의 책임 경계를 안내합니다.

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

[Compile](compile.md)

`CudaSource`를 공유 라이브러리와 메타데이터를 포함한 `CudaArtifact`로 변환해, 실제 실행 가능한 형태로 변환합니다.

## 탐색

| 확인할 내용                 | 시작 파일                                                                                                                                            |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 공개 API                    | [src/lib.rs](../../src/lib.rs)                                                                                                                       |
| Plan 구성·확정              | [plan/builder.rs](../../src/plan/builder.rs)                                                                                                         |
| 공통 emit·검증 진입점       | [emit/mod.rs](../../src/emit/mod.rs)                                                                                                                 |
| 준비 결과 trait·플랫폼 분기 | [emit/prepare.rs](../../src/emit/prepare.rs)                                                                                                         |
| 내부 소스 variant           | [emit/source.rs](../../src/emit/source.rs)                                                                                                           |
| CUDA 준비·자원 확정         | [emit/cuda/prepare.rs](../../src/emit/cuda/prepare.rs), [emit/cuda/requirements.rs](../../src/emit/cuda/requirements.rs)                             |
| Task·실행 domain            | [emit/cuda/domain.rs](../../src/emit/cuda/domain.rs)                                                                                                 |
| Body·Phase 모델             | [emit/cuda/body/mod.rs](../../src/emit/cuda/body/mod.rs)                                                                                             |
| Symbol·Code·CUDA 토큰화     | [emit/cuda/body/code.rs](../../src/emit/cuda/body/code.rs)                                                                                           |
| Body 목록 구성·재사용       | [emit/cuda/body/program.rs](../../src/emit/cuda/body/program.rs)                                                                                     |
| 실행 방식 선택·배치 연결    | [emit/cuda/execution.rs](../../src/emit/cuda/execution.rs)                                                                                           |
| 논리 검증                   | [emit/cuda/validation.rs](../../src/emit/cuda/validation.rs)                                                                                         |
| 단일 Body 생성·backend 계약 | [emit/cuda/body/builder.rs](../../src/emit/cuda/body/builder.rs), [emit/cuda/backend.rs](../../src/emit/cuda/backend.rs)                             |
| Streamed grid·launch        | [emit/cuda/streamed/mod.rs](../../src/emit/cuda/streamed/mod.rs)                                                                                     |
| Persistent 실행·workspace   | [emit/cuda/persistent/mod.rs](../../src/emit/cuda/persistent/mod.rs)                                                                                 |
| Persistent 접근·의존성      | [emit/cuda/persistent/access.rs](../../src/emit/cuda/persistent/access.rs), [emit/cuda/persistent/graph.rs](../../src/emit/cuda/persistent/graph.rs) |
| CUDA 소스·Body 렌더링       | [emit/cuda/render.rs](../../src/emit/cuda/render.rs)                                                                                                 |
| 실행 방식별 렌더링          | [emit/cuda/streamed/render.rs](../../src/emit/cuda/streamed/render.rs), [emit/cuda/persistent/render.rs](../../src/emit/cuda/persistent/render.rs)   |
| compile/link와 artifact     | [compile/cuda/mod.rs](../../src/compile/cuda/mod.rs), [compile/cuda/artifact.rs](../../src/compile/cuda/artifact.rs)                                 |
