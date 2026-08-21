# 모듈 ①·② 실행 결과

## 파일 구성
- `pipeline.py` — 전체 파이프라인 코드 (Colab/로컬 모두 실행 가능)
- `module1_output.json` — ①제출 분석 모듈 출력 (90건 전체)
- `module2_output.json` — ②취약 개념 추정 모듈 출력 (학생 6명)
- `evaluation_report.json` — Macro-F1 평가 결과

## 실행 방법
```bash
pip install pandas numpy scikit-learn openpyxl
apt-get install default-jdk   # javac 필요 (컴파일 판정용)
python3 pipeline.py
```
`모듈1_2용_데이터셋.xlsx`가 같은 폴더에 있어야 합니다.

## 결과 요약
- 90건 중 **32건 컴파일 실패** (문법 오류·미완성 코드 포함, 실제 데이터 특성상 많음)
- Macro-F1 **0.7778** (목표 0.65 이상 달성, decay=0.7 · threshold=0.5 기본값에서 이미 충족)
- 예시 검증(problem_id 22, noTeenSum): `compile_ok: false`, `error_pattern: helper_function_missing_return`
  → 가이드라인 문서의 JSON 예시는 `compile_ok: true`로 되어 있으나, 실제 코드(`fixTeen`이 값을 반환하지 않음)를 컴파일하면 에러가 나는 게 맞습니다. 문서 예시는 형식 설명용으로 보이며, 본 파이프라인 결과가 실제 컴파일 동작을 정확히 반영합니다.

## 설계상 알아둘 점 (한계)
1. **복합 개념 태그 처리**: "조건문, 반복문, 배열/문자열"처럼 한 문항이 여러 개념에 걸쳐 있으면, 실패 시 해당 개념 전부에 실패로 반영됩니다. 이 때문에 실제로는 조건문을 잘하는 학생도 복합 문항 실패가 누적되면 조건문 취약도가 다소 과대평가될 수 있습니다 (예: 학생 04c32d4d는 조건문 단독 문항은 거의 만점이지만 종합 취약도는 0.632로 나옴). 개선하려면 "각 문항의 개념 중 실제로 처음 등장하는 개념에 가중치를 더 주는" 방식 등을 다음 버전에서 검토할 수 있습니다.
2. **평가(hold-out) 방식**: 학생별 마지막 시도를 라벨로 숨기고 이전 이력만으로 예측했습니다. 학생이 6명뿐이라 평가 샘플 수가 적어(추가 검증 필요), 실서비스 규모 데이터에서는 재검증이 필요합니다.
3. **컴파일 판정**: `class Solution { ... }`로 감싸서 컴파일했습니다. `import`가 필요한 API를 쓰는 코드가 있다면 `cannot find symbol`로 잡혀 `undeclared_variable_or_method`로 태깅될 수 있으니, 실제 문항에 없는지 확인이 필요합니다 (이번 90건에는 해당 사례 없음).

## 다음 모듈(③④)에 전달할 것
`module2_output.json`의 `weakest_concept` 필드가 ③모듈(보완 문제 추천)의 입력이 됩니다.
