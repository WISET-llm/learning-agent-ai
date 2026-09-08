# learning-agent-ai

LLM 기반 자율형 프로그래밍 학습 에이전트 — 파일럿 구현

학생의 Java 코드 제출 1건을 입력받아 **① 제출 분석 → ② 취약 개념 추정 → ③ 보완 문제 추천 → ④ 개념카드 기반 LLM 피드백** 4단계를 거쳐 "다음에 풀 문제 + 학습 피드백 텍스트"를 만들어내는 파이프라인입니다. 각 모듈은 독립 실행이 가능하며, `src/e2e_pipeline.py`의 `run_pipeline()`이 이들을 메모리상에서 체이닝하는 단일 진입점입니다.

## 전체 흐름

```
StudentSubmissions (Code, binary_score, Score, timestep, 개념군_매핑)
        │
        ▼
┌─ ① submission_analysis ─────────────────────────────────────┐
│  javac로 컴파일 → 실패 테스트 수 집계 → 규칙 기반 error_pattern 태깅  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼  {compile_ok, test_pass_rate, error_pattern}
┌─ ② weak_concept ────────────────────────────────────────────┐
│  개념군별 시간가중 통과율(decay=0.7) → 취약도 0~1 → weakest_concept    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼  {concept_scores, weakest_concept}
┌─ ③ recommender ─────────────────────────────────────────────┐
│  문항 메타(난이도·선수개념) 구축 → 최근 통과율 기준 3갈래 추천 규칙        │
└──────────────────────────────┬──────────────────────────────┘
                               ▼  {recommended_problem_id, reason}
┌─ ④ feedback ────────────────────────────────────────────────┐
│  개념카드 + Claude API → 3~4문장 피드백 → 정답 노출 여부 자동 판정      │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
              {feedback_text, direct_answer_leak, judge_reasoning}
```

## 디렉터리 구조

```
learning-agent-ai/
├── src/
│   ├── pipeline.py                  # 배치 오케스트레이터: 원본 xlsx → ①②③④ 순차 실행, results/ 저장
│   ├── e2e_pipeline.py              # run_pipeline(): 제출 1건을 4모듈에 통과시키는 단일 진입점 + 배치 재생 + 통합 평가
│   ├── submission_analysis/
│   │   └── submission_analysis.py   # ① 컴파일 판정(javac) + error_pattern 규칙 태깅
│   ├── weak_concept/
│   │   └── weak_concept.py          # ② 개념별 취약도 산출 + Macro-F1 hold-out 평가 + 하이퍼파라미터 튜닝
│   ├── recommender/
│   │   └── recommender.py           # ③ 문항 메타 테이블 구축 + 추천 규칙 + 베이스라인 3종 + Recall@k/NDCG 평가
│   └── feedback/
│       └── feedback.py              # ④ 프롬프트/Claude 호출 + 규칙·LLM-judge 이중 정답노출 검사 + 베이스라인 비교
├── data/
│   ├── concept_cards.json           # 개념군(조건문/반복문/배열·문자열) 3종 개념카드
│   ├── problem_meta.json            # ③이 생성한 파일럿 12문항 메타(난이도·선수개념·요구사항)
│   └── raw/                         # (gitignore) 모듈1_2용_데이터셋.xlsx, 모듈3_4용_데이터셋.xlsx
├── results/
│   ├── module1_output.json          # ① 출력 (제출 90건)
│   ├── module2_output.json          # ② 출력 (학생 6명)
│   ├── evaluation_report.json       # ② Macro-F1 평가
│   ├── module3_output.json          # ③ 출력
│   ├── module3_evaluation.json      # ③ hold-out 평가 (제안 vs 베이스라인)
│   ├── e2e_output.json              # E2E 배치 재생 결과 (90건)
│   ├── e2e_evaluation_report.json   # ②③④ 평가 통합 리포트
│   └── README.md                    # 모듈별 결과 상세·설계 한계·통합 이슈 정리
├── tests/
│   ├── test_recommender.py          # ③ 단위 테스트
│   ├── test_feedback.py             # ④ 단위 테스트 (mock client, API 키 불필요)
│   └── test_e2e_pipeline.py         # 통합 테스트 (javac 필요)
├── docs/
│   └── 모듈별_입출력_가이드라인.md      # 모듈별 입력 컬럼 / 출력 스키마 요구사항
└── baselines/ configs/ evaluation/ experiments/ notebooks/   # (placeholder)
```

## 모듈별 동작

### ① 제출 분석 — `submission_analysis.py`

| 함수 | 역할 |
|---|---|
| `load_data(path)` | `StudentSubmissions` 시트 로드, `binary_score` 문자열을 리스트로 파싱 |
| `check_compile(code)` | 코드를 `class Solution { ... }`로 감싸 `javac` 실행 (10초 타임아웃) → `(성공여부, stderr)` |
| `tag_error_pattern(binary_score, compile_ok, stderr)` | 컴파일 실패 시 stderr 키워드로 태깅(`helper_function_missing_return`, `undeclared_variable_or_method`, `type_mismatch`, `syntax_error_*`, `compile_error_other`), 성공 시 테스트 결과로 `all_passed` / `partial_failure_logic` / `all_tests_failed_logic` |
| `build_module1_output(df)` | 행별 레코드 생성. 동일 코드는 컴파일 캐시로 재사용 |

이 모듈은 "무엇이 어떻게 틀렸는지"만 요약하고 개념 판단은 하지 않습니다.

### ② 취약 개념 추정 — `weak_concept.py`

- `compute_concept_scores(subject_df, decay=0.7)`: 학생의 제출 이력을 timestep 순으로 보며, 각 제출의 `개념군_매핑` 태그(콤마로 복수 가능)에 `decay^(max_t - t)` 가중치로 통과율을 누적 → **취약도 = 1 − 가중 평균 통과율**. 시도가 `MIN_ATTEMPTS=2` 미만인 개념은 `None`(데이터 부족).
- `weakest_concept` = 취약도가 가장 높은 개념.
- `evaluate_macro_f1(df)`: 학생별 마지막 제출을 숨기고 이전 이력으로 예측한 "취약(≥0.5)"이 실제 실패를 맞히는지 Macro-F1로 평가. `tune_hyperparams`가 decay×threshold 그리드서치.
- 결과: **Macro-F1 0.7778** (목표 0.65 달성).

### ③ 보완 문제 추천 — `recommender.py`

**문항 메타 테이블 구축** (`build_problem_meta`)
- `Real_17_TestCased` ⨝ `Real_50Problems`의 KC 원-핫 18개 컬럼 → 활성 KC 수 `n_kc`.
- 난이도: `n_kc ≤ 4` 하 / `5~6` 중 / `≥ 7` 상.
- 파일럿 12문항 선정: 복수 개념군 문항 전부 + 조건문 단독 문항 ProblemID 순 6개.
- 선수개념(`derive_concept_prerequisites`): 단독 문항이 없는 개념군은 평균 KC가 더 낮은 단독 개념군을 선수개념으로 삼음 → 현재 데이터에서는 `반복문 ← 조건문`, `배열/문자열 ← 조건문`. 하드코딩이 아니라 데이터 분포로 계산.

**추천 규칙** (`recommend_problem`) — `recent_pass_rate = 1 − concept_scores[weakest]` 기준

| 최근 통과율 | 우선순위 |
|---|---|
| < 0.3 | 선수개념 문항 → 같은 개념 쉬운 문항 → 같은 개념 유사 난이도 |
| 0.3 ~ 0.6 | 같은 개념 쉬운 문항 → 유사 난이도 → 선수개념 |
| ≥ 0.6 또는 정보 없음 | 같은 개념 유사 난이도 → 쉬운 문항 → 선수개념 |

이미 푼 문제는 제외, 갈래가 비면 파일럿 풀 미해결 문제로 폴백(`fallback_any_unsolved`), 전부 풀었으면 `None`(→ `no_candidate`).

**평가** (`evaluate_recommenders`): 슬라이딩 hold-out으로 "지금까지 이력으로 다음 문제 예측" 포인트 57개를 만들고 random / by_difficulty / by_pass_rate 베이스라인과 Recall@1·3, NDCG@3 비교. 제안 시스템 **Recall@3 0.7193** (목표 0.70 달성). 단, 커리큘럼 순서대로 제출된 소규모 데이터 특성상 by_difficulty 베이스라인이 더 높게 나온 점은 `results/README.md`에 원인 분석이 있습니다.

### ④ 개념카드 · LLM 피드백 — `feedback.py`

- 네트워크를 타는 함수는 `call_claude()` 하나뿐. 나머지는 모두 이 함수를 거치므로 테스트에서 `client`에 mock을 넣어 API 키 없이 검증 가능.
- `generate_feedback(...)`: 취약 개념·취약도·추천 문제·추천 사유·개념카드(설명/자주 틀리는 포인트/풀이 전 점검 항목)를 프롬프트에 넣고, 시스템 프롬프트로 **정답 코드·의사코드 금지, 3~4문장, 4요소(취약 개념→확인 포인트→다음 문제→풀이 전 점검) 포함, 격려 어조**를 강제.
- 정답 노출 판정 `evaluate_feedback`: 1차 정규식 필터(코드 블록, `if(...){`, `return ...;` 등)에 걸리면 즉시 leak 확정(LLM 비용 절약), 아니면 2차 LLM-as-judge가 "판정: 예/아니오 + 이유" 형식으로 판정.
- `evaluate_module4`: 파일럿 12문항 각각에 대해 제안 시스템(개념카드 포함) vs 일반 LLM 베이스라인(문제 설명만)의 정답 노출 비율 비교. 목표 10% 이하.
- 기본 모델은 `claude-sonnet-5`, `MODULE4_MODEL` 환경변수로 변경 가능. 추천 문제가 없는 학생(`no_candidate`)은 LLM 호출 없이 정형 축하 문구 사용.

### E2E 통합 — `e2e_pipeline.py`

```python
run_pipeline(subject_id, new_submission, submission_history, problem_meta, concept_cards, model, client)
→ {"subject_id", "status", "module1", "module2", "module3", "module4"}
```

- `new_submission`은 원본 형태(`Code`, `binary_score`, `ProblemID` → ①을 실제 실행)와 이미 분석된 ①레코드 형태(`Code` 없음 → 값 재사용) 둘 다 허용.
- 모듈 간 스키마 차이는 어댑터에서만 흡수합니다: `_normalize_submission` (PascalCase→snake_case), `_history_to_concept_df` (①출력에 없는 개념 태그를 `problem_meta.concept_groups`로 역매핑), `timestep` 미지정 시 이력 마지막+1 자동 배정.
- 어느 단계가 실패해도 예외를 올리지 않고 해당 모듈에 `{"error", "stage"}`, 이후 단계는 `{"skipped": true}`로 기록하며 `status="partial_failure"`.
- `run_batch(limit)`: `module1_output.json`의 90건을 학생별 timestep 순으로 재생. `build_e2e_evaluation_report()`: ②③④ 평가를 하나로 병합.

## 설치 및 실행

### 요구사항

- Python 3.9+
- JDK (`javac`) — 모듈① 컴파일 판정용
- Anthropic API 키 — 모듈④ 실행 시에만 필요 (없어도 파이프라인은 죽지 않고 ④를 건너뜀)

```bash
pip install pandas numpy scikit-learn openpyxl anthropic pytest
sudo apt-get install default-jdk          # 또는 JDK 별도 설치
export ANTHROPIC_API_KEY=sk-ant-...       # 선택
export MODULE4_MODEL=claude-sonnet-5      # 선택 (기본값)
```

### 데이터 배치

`data/raw/`는 gitignore 대상이라 원본 xlsx를 직접 넣어야 합니다.

| 파일 | 필요한 모듈 | 없을 때 동작 |
|---|---|---|
| `data/raw/모듈1_2용_데이터셋.xlsx` | ①② | 커밋된 `results/module1_output.json`·`module2_output.json` 재사용 |
| `data/raw/모듈3_4용_데이터셋.xlsx` | ③ (메타 구축·평가) | ③ 실행 불가 — 필수 |

### 실행 명령

```bash
# 저장소 루트에서
python3 src/pipeline.py                     # ①→②→③→④ 배치 전체 실행 (results/·data/ 산출)
python3 src/e2e_pipeline.py                 # 제출 이력 재생 → results/e2e_output.json, e2e_evaluation_report.json
python3 src/e2e_pipeline.py --limit 20      # 모듈④ LLM 호출 건수 제한

# 모듈 단독 실행
python3 src/recommender/recommender.py
python3 src/feedback/feedback.py

# 테스트 (API 키 불필요, javac는 필요)
pytest tests/ -v
```

### 코드에서 직접 호출

```python
import sys, json
sys.path.insert(0, "src")
from e2e_pipeline import run_pipeline
from feedback import load_concept_cards

problem_meta = json.load(open("data/problem_meta.json", encoding="utf-8"))
concept_cards = load_concept_cards("data/concept_cards.json")

new_submission = {
    "ProblemID": 12,
    "Code": "public int ... { ... }",
    "binary_score": [1, 1, 0, 0, 1],
}
history = [  # module1_output.json 레코드 스키마
    {"problem_id": 1, "timestep": 0, "test_pass_rate": 1.0, "compile_ok": True,
     "n_failed": 0, "n_total": 15, "error_pattern": "all_passed"},
]

result = run_pipeline("student_01", new_submission, history, problem_meta, concept_cards)
print(result["module3"]["recommended_problem_id"], result["module4"]["feedback_text"])
```

## 산출물 스키마

```jsonc
// results/module1_output.json (①)
{"subject_id": "...", "problem_id": 1, "timestep": 2, "compile_ok": true,
 "test_pass_rate": 1.0, "n_failed": 0, "n_total": 15, "error_pattern": "all_passed"}

// results/module2_output.json (②)
{"subject_id": "...", "concept_scores": {"조건문": 0.632, "반복문": 0.63, "배열/문자열": 0.66},
 "weakest_concept": "배열/문자열", "n_submissions": 17}

// results/module3_output.json (③)
{"subject_id": "...", "recommended_problem_id": 12,
 "recommendation_type": "same_concept_similar", "reason": "동일 개념(조건문)의 유사 난이도 문항 추천 — ..."}

// data/problem_meta.json (③이 생성)
{"problems": [{"problem_id": 1, "assignment_id": 439, "concept_groups": ["조건문"], "n_kc": 4,
               "difficulty": "하", "requirement": "...", "prerequisite_concept": null}, ...],
 "concept_prerequisites": {"조건문": null, "반복문": "조건문", "배열/문자열": "조건문"}}

// results/module4_output.json (④, API 키 있을 때)
{"subject_id": "...", "feedback_text": "...", "direct_answer_leak": false, "judge_reasoning": "..."}
```

## 현재 결과 요약

| 모듈 | 지표 | 결과 | 목표 |
|---|---|---|---|
| ② 취약 개념 추정 | Macro-F1 | 0.7778 | ≥ 0.65 ✅ |
| ③ 문제 추천 | Recall@3 | 0.7193 | ≥ 0.70 ✅ |
| ④ LLM 피드백 | 정답 노출 비율 | 미측정 (API 키 없는 환경에서 실행됨) | ≤ 10% |

## 알려진 한계 / 다음 단계

- 학생 6명·파일럿 12문항 규모라 평가 표본이 작습니다. 실서비스 규모에서 재검증이 필요합니다.
- 복합 개념 문항 실패가 모든 구성 개념에 반영되어 취약도가 과대평가될 수 있습니다.
- `run_pipeline`은 `problem_meta`(파일럿 12문항)만 받으므로 그 밖의 문항은 ②에서 개념 신호가 소실됩니다. 전체 문항 concept map을 별도로 넘기는 옵션 인자 추가를 검토 중입니다.
- 모듈④의 정답 노출 비율은 `ANTHROPIC_API_KEY` 설정 후 `python3 src/feedback/feedback.py`를 실행해 실측해야 합니다.

자세한 설계 근거와 통합 시 발견된 이슈는 [`results/README.md`](results/README.md), 입출력 요구사항은 [`docs/모듈별_입출력_가이드라인.md`](docs/모듈별_입출력_가이드라인.md)를 참고하세요.
