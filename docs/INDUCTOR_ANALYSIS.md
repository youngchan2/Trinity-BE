# Inductor의 분석 결과 전달과 Trinity의 첫 구현

기준은 프로젝트 requirements에 맞춘 **PyTorch v2.8.0** 공개 소스다. 최초 조사에서는
기본 Python 환경에 torch가 없어 정적 소스를 조사했다. Inductor 실행이나 최신 버전 전체에 대한
검증을 의미하지 않는다.

## 실제로 정보가 보관되는 곳

Inductor는 여러 분석 결과를 각 단계의 객체에 보관하고 연결한다.

| 보관 주체 | 보관하는 정보 | 소비하는 단계 |
|---|---|---|
| `LoopBody` | 실행 가능한 body graph, 인덱스 식, 접근 종류별 기록, 반복·reduction 변수 범위 | 의존 분석과 body codegen |
| scheduler node | body와 범위, read/write 의존, 아직 충족되지 않은 의존 | 스케줄 처리와 backend codegen |
| scheduler buffer | 정의한 연산, 사용자 목록 | 사용 관계와 수명 처리 |
| body graph의 node metadata | 전파된 dtype 등의 optimization context | 연산 codegen |
| SIMD kernel features / kernel context | 선택된 노드 순서·인덱싱, 현재 노드, 값 범위, CSE와 출력 상태 | 최종 kernel source 생성 |

`LoopBody`는 연산 body를 FX graph로 보존하면서 memory usage의 항목이 index 식을
참조하게 한다. 텐서 이름 외에 실제 접근 식을 보존한다는 점이 Trinity의 접근 레코드에
해당한다. [LoopBody 소스](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/loop_body.py)

`SchedulerNode._compute_attrs`는 body와 반복 범위를 만들고 의존 추출 결과를 연결한다.
`set_read_writes`는 읽기/쓰기와 미충족 의존을 갱신한다. 루프가 바뀌면
`refresh_dependencies`로 관련 결과와 캐시를 갱신한다. scheduler의 버퍼 객체에는
정의한 연산과 사용자가 연결되어 있다.
[Scheduler 소스](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/scheduler.py)

이름은 실제 식별 키로 사용된다. 그러나 scheduler의 의존 비교는 접근 index와 size도
확인한다. MemoryDep 생성 호출에는 이름, index, 변수, size, mode가 전달된다.
dependencies.py 본문은 공개 fetch에서 가져오지 못했으며, 여기서는 scheduler의
생성·소비 코드를 근거로 설명한다.
[의존 생성·비교가 있는 scheduler](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/scheduler.py)

## Codegen으로 연결되는 경로

```mermaid
flowchart LR
    A[LoopBody와 인덱스 정보] --> B[SchedulerNode와 의존 정보]
    B --> C[SIMD kernel features와 노드 순서]
    C --> D[Kernel의 인덱싱 확정]
    D --> E[Node body를 codegen handler로 실행]
    E --> F[Triton 소스와 호출 코드]
```

SIMD scheduling은 kernel features를 받아 kernel 후보를 구성한다. 노드 순회를 통해
인덱싱을 먼저 확정하고, 이후 각 노드의 codegen을 호출한다. 완성된 kernel에서 소스를
얻어 정의하고 호출 코드와 연결한다.
[SIMD scheduling 소스](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/codegen/simd.py)

노드의 codegen은 현재 kernel/node와 indexing handler를 설정한 상태로 보존된 body를
실행한다. 따라서 분석 결과뿐 아니라 동일한 연산 body도 codegen에 전달된다.
[SchedulerNode.codegen](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/scheduler.py)

dtype 전파는 body graph node의 metadata에 결과를 기록한다. kernel context는 현재
노드의 bounds를 참조하고, CSE proxy는 이전 store의 지역 값을 load에 공급하기도 한다.
즉 실제 emission에도 분석과 상태가 있다. Trinity에서는 초기화·저장 위치 결정을
명시적 계획으로 분리하되, 임시 변수·출력 캐시까지 analyzer로 옮길 필요는 없다.
[공통 codegen 소스](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/codegen/common.py)

## Trinity에 적용한 결정

1. **원본과 결과를 함께 소유한다.** `ProgramAnalysis`는 변경되지 않은 입력 트리와
   접근·scope 레코드를 보존한다. codegen은 공유 참조로 이 결과를 읽도록 한다.
2. **풍부한 접근 기록에서 요약 set을 만든다.** `AccessInfo`에는 접근 index와 문장·scope가
   있고, 커널/직접 scope의 `ReadWrites`는 텐서 조회용 요약이다.
3. **scope는 명시한다.** Trinity에는 이미 ploop/sloop가 있으므로 발생 위치마다 ScopeId를
   부여한다. 각 load/store는 가장 가까운 scope 한 곳에만 속한다.
4. **최적화와 수집을 구분한다.** 현재 pass는 기존 seq 순서와 kernel 경계를 보존하며
   익명 계산식을 재결합하거나 fusion·tiling을 탐색하지 않는다.
5. **후속 의미 분석과 출력 상태를 구분한다.** 정의 연결, live_in/out, recurrence,
   init/store, shape/dtype는 후속 pass다. 임시 이름과 CSE는 emitter 상태다.

이 설계는 Inductor의 객체를 그대로 옮긴 구조가 아니다. 공통으로 채택한 부분은
"원본 body + 접근 정보 + 단계별 결과"의 연결이다. Inductor의 name-keyed store cache를
Trinity의 여러 sloop에 그대로 적용하지 않는다. 어느 정의와 타일을 재사용할 수 있는지는
Trinity의 접근 관계에서 확인해야 한다.

## 현재 구현과 확인한 범위

- `src/analyzer/ir.rs`: 기존 S-expression 파일을 위한 작은 syntax adapter.
  `IrNode` 직접 구성도 가능하다. Rust optimizer의 native IR/metadata adapter는 후속 작업이다.
- `src/analyzer/collect.rs`: 구조화된 순회, 그룹별 load/store 대응, kernel/scope별 수집.
- `src/analyzer/model.rs`: 소유권이 있는 결과 객체, 순서가 있는 접근, 구조화된 index와 loop binding.
- `tests/analyzer.rs`: 실제 실패 4개와 grouped tensor, 동명 루프, mutation 등의 회귀 테스트.

후속 구현으로 `src/triton/plan.rs`에서 shape, 접근 공급원, accumulator, 초기화와 저장 위치를
결정하고, `src/triton/emit.rs`에서 Triton 소스와 Python wrapper를 생성한다.
view layout은 접근별로 보존한다. 별도의 scheduler나 범용 buffer 관리 계층은 추가하지 않았다.
현재의 구체적인 지원 범위와 검증 한계는 [TRITON_FALLBACK.md](TRITON_FALLBACK.md)를 따른다.
Rust의 소스 생성 검증을 기존 Python generator의 GPU 컴파일 오류 수정으로 해석하지 않는다.

전체 Llama 코퍼스에서 placeholder가 없는 FFN 839개와 vanilla 170개가 수집에 성공했다.
각각 kernel region 3,350개/170개, load/store 접근 25,819개/6,661개를 보존한다.
유효한 입력의 파싱·수집 검증이며 numerical correctness 검증은 아니다.

실패 4개에서 `attn_O_norm`은 정의된 루프 안의 register로만 유지하고, 별도 초기화나
루프 밖 export를 생성하지 않는다. FFN 4의 normalization은 sibling loop 사이에
여러 tile을 보존하는 기존 fp16 global tensor로 계획한다. Emitter는 원래 Python backend의
출력 규약을 따르며, 별도 private scratch나 자동 할당 wrapper를 만들지 않는다.
이후 `Trinity-BE/backend/.venv`에서 FFN Llama 네 케이스와 vanilla Llama/Falcon 591의
GPU smoke test를 수행했다. 구체적인 검증 범위는 [TRITON_FALLBACK.md](TRITON_FALLBACK.md)를 따른다.
