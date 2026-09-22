# CUDA

## 실행 배치

### Streamed

Streamed는 일반 CUDA kernel grid에 작업을 배치합니다. 단일 GPU 실행을 지원합니다.

### Persistent

Persistent는 작업을 Worker CTA 위에서 실행하고 다중 GPU 실행으로 확장하기 위한 후속 설계입니다.
현재 새 `emit()` 경로에서는 지원하지 않습니다. 구현된 범위와 후속 설계의 구분은
[Emission](../../emission.md)을 참고합니다.
