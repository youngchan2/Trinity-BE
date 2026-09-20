# Explicit Plan 전환 전 Rust 코드 — 참조 전용

`implementation/schedules`는 emit 격리 중 추출한 코드여서 이전 emit 보존본과 별도로 보존한다.
이번 보존본에는 자동 확장 제거 전 implementation, plan, Rust 호출부와 테스트를 담았다.
`snapshot/`의 경로는 최상위 Trinity 저장소 기준이며, 당시 작업 트리를 그대로 복사했다.
`MANIFEST.json`에 파일별 SHA-256과 크기를 기록한다.

이 코드는 빌드·테스트 대상이나 실행 fallback에 연결하지 않는다. Python은 보존하지 않는다.
기존 `old/emit-rewrite/snapshot`과 그 manifest는 변경하지 않는다.

## 전환 결과

- Plan은 명시된 Loop·Operation을 보존한다. Compute 본문 누락은 오류이며, finalize에서
  loop·tile·본문을 생성하지 않는다. Schedule trait과 파생 coordinates는 제거했다.
- `implementation/definitions`에는 ID·attributes·후보 열거만 남아 있다.
- Python은 현재 위치에서 explicit expression 및 loop 노드를 전달하도록 수정했다.
- Whole-tensor Candidate의 자동 확장 경로는 `ExplicitProgramRequired`로 중단했다.
  기존 extraction은 유지하며 Candidate에서 명시적 프로그램을 전달하는 계약은 후속 과제다.

검증: 보존 파일 32개의 크기와 SHA-256 일치. 기존 emit 보존본 82개도 무결성을 확인했다.
두 활성 crate의 Rust 테스트 85개와 관련 Python 테스트 8개 통과. 기존 emitter는 여전히
사용 불가이며, GPU 실행과 전체 Python emission 테스트는 이번 검증에 포함하지 않았다.
