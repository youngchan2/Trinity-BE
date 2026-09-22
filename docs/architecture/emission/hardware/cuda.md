# CUDA

## 실행 배치

### Streamed

Streamed는 일반 CUDA kernel grid에 작업을 배치합니다. 단일 GPU 실행을 지원합니다.

### Persistent

Persistent는 작업을 Worker CTA 위에서 실행할 수 있게 구성합니다. 다중 GPU 실행을 지원합니다.
