# 모듈 ①·②·③·④ 실행 결과 · 통합 파이프라인(E2E)

## 모듈 ①·② 제출 분석 · 취약 개념 추정

### 파일 구성
- `src/submission_analysis/submission_analysis.py` — ①제출 분석 모듈 코드
- `src/weak_concept/weak_concept.py` — ②취약 개념 추정 모듈 코드 + Macro-F1 평가
- `src/pipeline.py` — 위 두 모듈(및 ③모듈)을 순서대로 실행하는 오케스트레이터
- `module1_output.json` — ①제출 분석 모듈 출력 (90건 전체)
- `module2_output.json` — ②취약 개념 추정 모듈 출력 (학생 6명)
- `evaluation_report.json` — Macro-F1 평가 결과

### 실행 방법
```bash
pip install pandas numpy scikit-learn openpyxl
apt-get install default-jdk   # javac 필요 (컴파일 판정용)
python3 src/pipeline.py       # 저장소 루트에서 실행 (data/, results/ 상대경로를 그대로 사용)
```
`data/raw/모듈1_2용_데이터셋.xlsx`가 있으면 모듈①②를 처음부터 실행하고, 없으면(이 저장소
기본 상태) 이미 커밋된 `module1_output.json`·`module2_output.json`을 재사용해 곧바로
모듈③까지 이어서 실행합니다.

### 결과 요약
- 90건 중 **32건 컴파일 실패** (문법 오류·미완성 코드 포함, 실제 데이터 특성상 많음)
- Macro-F1 **0.7778** (목표 0.65 이상 달성, decay=0.7 · threshold=0.5 기본값에서 이미 충족)
- 예시 검증(problem_id 22, noTeenSum): `compile_ok: false`, `error_pattern: helper_function_missing_return`
  → 가이드라인 문서의 JSON 예시는 `compile_ok: true`로 되어 있으나, 실제 코드(`fixTeen`이 값을 반환하지 않음)를 컴파일하면 에러가 나는 게 맞습니다. 문서 예시는 형식 설명용으로 보이며, 본 파이프라인 결과가 실제 컴파일 동작을 정확히 반영합니다.

### 설계상 알아둘 점 (한계)
1. **복합 개념 태그 처리**: "조건문, 반복문, 배열/문자열"처럼 한 문항이 여러 개념에 걸쳐 있으면, 실패 시 해당 개념 전부에 실패로 반영됩니다. 이 때문에 실제로는 조건문을 잘하는 학생도 복합 문항 실패가 누적되면 조건문 취약도가 다소 과대평가될 수 있습니다 (예: 학생 04c32d4d는 조건문 단독 문항은 거의 만점이지만 종합 취약도는 0.632로 나옴). 개선하려면 "각 문항의 개념 중 실제로 처음 등장하는 개념에 가중치를 더 주는" 방식 등을 다음 버전에서 검토할 수 있습니다.
2. **평가(hold-out) 방식**: 학생별 마지막 시도를 라벨로 숨기고 이전 이력만으로 예측했습니다. 학생이 6명뿐이라 평가 샘플 수가 적어(추가 검증 필요), 실서비스 규모 데이터에서는 재검증이 필요합니다.
3. **컴파일 판정**: `class Solution { ... }`로 감싸서 컴파일했습니다. `import`가 필요한 API를 쓰는 코드가 있다면 `cannot find symbol`로 잡혀 `undeclared_variable_or_method`로 태깅될 수 있으니, 실제 문항에 없는지 확인이 필요합니다 (이번 90건에는 해당 사례 없음).

### 다음 모듈(③④)에 전달할 것
`module2_output.json`의 `weakest_concept` 필드가 ③모듈(보완 문제 추천)의 입력이 됩니다.

---

## 모듈 ③ 보완 문제 추천

### 파일 구성
- `src/recommender/recommender.py` — 문항 메타 테이블 구축 + 추천 로직 + 베이스라인 3종 + 평가
  (모듈①②와 입력 데이터셋이 달라 별도 패키지로 분리했고, 개념별 취약도 계산 함수는
  `weak_concept` 모듈 것을 그대로 import해서 재사용합니다)
- `data/problem_meta.json` — 파일럿 12문항의 난이도·선수개념·개념군 메타 테이블 (새로 생성)
- `module3_output.json` — `module2_output.json`의 학생별 `weakest_concept` 기준 추천 결과
- `module3_evaluation.json` — hold-out 평가 결과 (제안 시스템 vs 베이스라인 3종)
- `tests/test_recommender.py` — pytest 단위 테스트 15건

### 실행 방법
```bash
pip install pandas numpy scikit-learn openpyxl pytest
python3 src/pipeline.py                 # 모듈①②③ 전체 실행 (또는 아래처럼 ③만 단독 실행)
python3 src/recommender/recommender.py  # 저장소 루트 기준 절대경로를 쓰므로 어디서 실행해도 됨
pytest tests/test_recommender.py -v
```
`data/raw/모듈3_4용_데이터셋.xlsx`가 필요합니다(Real_50Problems·Real_17_TestCased 시트).

### 문항 메타데이터 테이블 구축 규칙
- **파일럿 문항 선정 (17개 중 12개)**: Real_17_TestCased 중 복수 개념군 문항(조건문+반복문+배열/문자열
  등) 6개는 전부 포함했습니다 — 이 17문항 풀에서 반복문·배열/문자열은 단독 문항이 전혀 없고
  이 6개만이 해당 개념 신호를 담고 있기 때문입니다. 나머지는 단일 개념군(조건문 단독) 문항 중
  ProblemID 오름차순 상위 6개를 포함했습니다. 제외된 5개(ProblemID 20, 21, 22, 24, 25)는 모두
  조건문 단독이고 n_kc(4~6)가 이미 포함된 6개와 겹치는 범위라 정보 손실이 적습니다.
- **난이도(하/중/상)**: Real_50Problems의 KC 원-핫 컬럼 중 활성화된 개수(`n_kc`, 문항 복잡도
  proxy)를 기준으로 합니다. 실제 17문항의 n_kc 분포가 3~8(중앙값 5)이라, 하: n_kc ≤ 4 / 중: n_kc
  5~6 / 상: n_kc ≥ 7로 경계를 고정했습니다(코드 상수 `DIFFICULTY_LOW_MAX`, `DIFFICULTY_HIGH_MIN`).
- **선수개념**: "개념군 X가 항상 다른 개념군과 결합된 문항에만 등장하고(X 단독 문항이 없음),
  그 결합 문항들의 평균 KC 개수가 어떤 단독 개념군 Y의 평균보다 높으면 Y를 X의 선수개념으로
  본다"는 규칙을 `derive_concept_prerequisites()` 함수로 계산했습니다(하드코딩이 아니라 문항
  메타 리스트의 실제 분포에서 계산하므로, 나중에 반복문 단독 문항이 추가되는 등 데이터가
  바뀌면 자동으로 갱신됩니다). 결과: **조건문**(선수개념 없음 — 단독 문항 6개 존재) ←
  **반복문**, **배열/문자열**(둘 다 단독 문항이 없고 항상 조건문과 결합되어 나타나며, 결합
  문항의 평균 n_kc가 조건문 단독 문항 평균보다 높음 → 조건문이 선수개념).
- 개념카드 관련 필드(설명/대표예제/자주틀리는포인트)는 `problem_meta.json`에 넣지 않았습니다.
  가이드라인상 개념카드는 문항 단위가 아니라 개념군(조건문/반복문/배열·문자열) 단위로
  모듈④가 별도로 작성하는 자원이기 때문입니다.

### 추천 로직 (제안 시스템)
`recommend_problem(weakest_concept, solved_problem_ids, problem_meta, concept_prerequisites,
recent_pass_rate)` — 최근 통과율(모듈②의 `concept_scores[weakest_concept]` 취약도를
`1 - 취약도`로 역산)을 기준으로 세 갈래 중 하나를 선택합니다.
- 통과율 **< 30%**: 선수개념 문제로 되돌아감 (기초가 불안정하므로 조건문부터 재확인)
- 통과율 **30~60%**: 같은 개념 쉬운 문제로 완충
- 통과율 **≥ 60%** (또는 정보 없음): 같은 개념 유사(중~상) 난이도로 진행

각 갈래 안에서는 이미 푼 문제를 제외한 뒤 난이도·problem_id로 결정적 정렬하고, 후보가 없으면
다음 갈래로 폴백하며, 그래도 없으면 파일럿 풀 전체의 미해결 문제를 난이도 오름차순으로
채웁니다(`fallback_any_unsolved`) — `weakest_concept`이 `None`이거나 메타 테이블에 없는
개념이어도 에러 없이 추천이 나옵니다(pytest로 검증).

### 결과 요약
- `module3_output.json`: `module2_output.json` 6명 중 2명만 파일럿 12문항 밖에 남은 문제가 있어
  실제 추천(`same_concept_easy`, `same_concept_similar`)이 나갔고, 나머지 4명은 이미 파일럿
  12문항을 전부 풀어 `no_candidate`로 처리됐습니다 — 학생 6명뿐인 파일럿 데이터의 한계입니다.
- hold-out 평가(`module3_evaluation.json`, 슬라이딩 방식으로 평가 포인트 57개 확보):

| 방법 | Recall@1 | Recall@3 | NDCG@3 |
|---|---|---|---|
| 제안 시스템 (proposed) | 0.4211 | **0.7193** | 0.5794 |
| 무작위 (random) | 0.2632 | 0.5789 | 0.4486 |
| 난이도만 (by_difficulty) | 0.4561 | 0.8596 | 0.7015 |
| 정답률 기반 (by_pass_rate) | 0.3333 | 0.7018 | 0.5405 |

- 목표(Recall@3 ≥ 0.70) **달성** — 제안 시스템 0.7193.
- 다만 "난이도만" 베이스라인이 제안 시스템보다 더 높은 Recall@3/NDCG@3을 기록했습니다. 아래
  한계점 3-1에 이유를 정리했습니다.

### 설계상 알아둘 점 (한계)
1. **"난이도만" 베이스라인이 더 높게 나온 이유(추정)**: 이 파일럿 문항은 AssignmentID
   (439→487→492→494→502)별로 난이도가 대체로 증가하는 커리큘럼 순서로 배치돼 있고, 학생들도
   대체로 이 순서(≈난이도 순서)대로 제출한 것으로 보입니다. 그래서 "개념과 무관하게 다음으로
   쉬운 미해결 문제"를 고르는 단순 베이스라인이 실제 학생 행동과 우연히 잘 맞아떨어집니다.
   개념 인식 추천(제안 시스템)의 이점은 "어떤 개념이 취약한지"를 정확히 짚어 되돌아가거나
   다른 개념으로 분기하는 데 있는데, 문항 수(12개)·학생 수(6명)가 적어 고정된 커리큘럼 순서를
   벗어나는 경우가 거의 없다 보니 이 차이가 Recall@3 수치로는 뚜렷하게 드러나지 않았습니다.
   문항 풀이 커지고 학생마다 다른 순서·다른 취약 개념을 겪게 되면 개념 인식 추천의 이점이 더
   분명해질 것으로 예상합니다.
2. **평가셋 소규모**: 학생 6명, 파일럿 문항 12개라는 원천적 한계로 마지막 제출 1건만
   라벨로 쓰면 평가 포인트가 6개뿐입니다. 슬라이딩 hold-out(학생별 매 시점마다 "지금까지의
   이력으로 다음 문제 예측" 포인트 생성)으로 57개까지 늘렸지만, 여전히 소규모 통계라 추가
   검증이 필요합니다.
3. **파일럿 외 문항으로의 라벨 이탈**: 정답 레이블(실제로 다음에 푼 문제)이 파일럿 12문항
   밖(제외된 5개 조건문 단독 문항)인 경우 평가에서 제외했습니다(전체 전이 포인트 78건 중
   21건, `n_skipped_true_label_outside_pilot_pool`). 이 문항들도 메타 테이블에 포함했다면 더
   많은 평가 포인트를 확보할 수 있었을 것입니다.
4. **정답률 베이스라인의 정보 단순화**: `by_pass_rate`는 전체 로그(다른 학생 포함)의 문항별
   평균 통과율을 고정값으로 한 번만 계산해 사용합니다 — 매 hold-out 시점마다 다시 계산하지
   않았으므로 약간의 낙관적 편향이 있을 수 있습니다(제안 시스템에는 영향 없이 베이스라인에만
   해당하는 단순화입니다).

### 다음 모듈(④)에 전달할 것
`module3_output.json`의 `recommended_problem_id`(+ `recommendation_type`, `reason`)가 ④모듈
(개념카드·LLM 피드백)의 입력이 됩니다. `recommendation_type`이 `no_candidate`인 학생(현재
6명 중 4명)은 ④모듈에서 "축하 피드백 + 파일럿 문항 소진 안내" 같은 별도 분기가 필요합니다.

---

## 모듈 ④ 개념카드 · LLM 피드백

### 파일 구성
- `data/concept_cards.json` — 조건문/반복문/배열·문자열 3개 개념군 개념카드(설명·대표예제·
  자주틀리는포인트·풀이 전 점검 항목)
- `src/feedback/feedback.py` — 피드백 생성 + 정답 직접 제시 자동 평가 로직
- `results/module4_output.json` — 학생별 피드백 텍스트 + 채점 결과 (Claude API 키가 있어야
  생성됨)
- `results/module4_evaluation.json` — 파일럿 12문항 전체 기준 제안 시스템 vs 일반 LLM
  베이스라인의 정답 노출 비율 비교 (Claude API 키가 있어야 생성됨)
- `tests/test_feedback.py` — pytest 단위 테스트 15건 (전부 목 client로 실행, API 키 불필요)

### 실행 방법
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...   # 또는 ant auth login
python3 src/feedback/feedback.py      # 단독 실행 (또는 python3 src/pipeline.py 로 ①~④ 전체 실행)
pytest tests/test_feedback.py -v      # API 키 없이도 통과함 (전부 목 client 사용)
```
기본 모델은 `claude-sonnet-5`이며 `MODULE4_MODEL` 환경변수로 바꿀 수 있습니다. API 키가 없으면
`feedback.py`·`pipeline.py` 모두 에러 없이 안내 메시지만 출력하고 해당 단계를 건너뜁니다.

### 설계 요약
- **개념카드**: `ProblemMeta_난이도표_예시` 시트의 컬럼 구조(설명/대표예제/자주틀리는포인트)를
  참고해 조건문·반복문·배열/문자열 3개 개념군 카드를 새로 작성했습니다(문항 단위가 아니라
  개념군 단위 — 가이드라인상 이 편이 맞다고 판단한 이유는 아래 한계점 참고). 여기에 연구계획서가
  언급한 "풀이 전 점검 항목"을 추가 필드로 넣었습니다.
- **피드백 생성**: `call_claude()` 한 함수만 실제로 네트워크를 타고, `generate_feedback` /
  `generate_baseline_feedback` / `judge_direct_answer_leak`은 전부 이 함수를 거쳐서만 LLM을
  호출합니다. 시스템 프롬프트에 "정답 코드/로직 금지", "3~4문장", "취약 개념 언급→구체적 확인
  포인트→다음 문제→풀이 전 점검"의 4요소, "격려하는 어조"를 명시적으로 강제했습니다.
- **정답 직접 제시 자동 평가**: 1차로 정규식 기반 규칙 필터(코드 블록, `if(...){`, `return ...;`
  같은 완성된 구문 패턴)를 걸러 걸리면 바로 확정하고, 걸리지 않으면 2차로 별도 Claude 호출로
  "이 피드백이 정답을 직접 알려주는가?"를 예/아니오 + 이유로 판정하게 합니다. 규칙 기반에서
  걸릴 때는 LLM judge를 호출하지 않아 비용을 아낍니다.
- **베이스라인 비교**: `module3_output.json` 기준 실제 추천을 받은 학생은 6명 중 2명뿐이라
  평가 표본이 너무 적습니다. 그래서 평가용으로는 파일럿 12문항 전체 각각에 대해 "이 문항이
  방금 추천됐다고 가정"하고 제안 시스템 피드백(개념카드 포함)과 일반 LLM 베이스라인 피드백
  (취약 개념 맥락 없이 문제 설명만 제공)을 하나씩 생성해 표본을 12개로 늘렸습니다.

### 결과 요약
**이 저장소 환경에는 ANTHROPIC_API_KEY가 없어 실제 Claude 호출은 이루어지지 않았습니다.**
그래서 `results/module4_output.json`·`results/module4_evaluation.json`은 아직 생성되지
않았고, 정답 노출 비율(목표 10% 이하)도 실측하지 못했습니다. 대신:
- 코드 구조·프롬프트·규칙 기반 필터·분기 로직은 `pytest tests/test_feedback.py`(15건, 전부
  목 client 사용)로 검증했습니다 — 예: "이미 파일럿을 다 푼 학생은 LLM을 호출하지 않는다",
  "규칙 기반에서 코드가 걸리면 judge를 호출하지 않는다", "베이스라인은 취약 개념 맥락을
  프롬프트에 넣지 않는다" 등.
- `ANTHROPIC_API_KEY`를 설정한 뒤 `python3 src/feedback/feedback.py`(또는 `pipeline.py`)를
  실행하면 바로 실제 결과가 생성되도록 준비돼 있습니다.

### 설계상 알아둘 점 (한계)
1. **실제 LLM 호출 결과 미검증**: 위에서 설명한 대로 API 키가 없어 정답 노출 비율 목표(10%
   이하) 달성 여부를 이번 세션에서는 확인하지 못했습니다. 시스템 프롬프트가 규칙을 명시하고
   있다는 것과, 실제로 그 규칙을 모델이 지키는지는 별개의 문제이므로 실행 후 반드시 재검토가
   필요합니다.
2. **LLM-judge의 신뢰도**: 정답 직접 제시 여부를 같은 계열의 LLM으로 판정하다 보니, judge
   자체가 관대하거나 엄격하게 치우칠 가능성이 있습니다. 표본이 작을 때는 사람이 일부를 직접
   검수해 judge 판정과 비교해보는 과정이 필요합니다.
3. **개념카드가 문항이 아닌 개념군 단위**: 가이드라인은 "실제 8~12개 문항에 맞는 개념카드"를
   요청했지만, 이 12개 문항이 3개 개념군(조건문/반복문/배열·문자열)으로만 나뉘고 개념군별
   설명·자주틀리는포인트가 문항마다 크게 다르지 않아, 문항 12개 각각의 카드 대신 개념군 3개
   카드로 작성했습니다(각 카드 설명에 관련 실제 문항 번호를 예시로 언급). 문항별로 더 세분화된
   카드가 필요하면 이 구조를 그대로 확장할 수 있습니다.
4. **평가용 문항 단위 시나리오는 가정임**: `module4_evaluation.json`의 12개 평가 포인트는
   실제 학생-추천 쌍이 아니라 "이 문항이 막 추천됐다고 가정"한 시나리오입니다. 실제 학생
   맥락(취약도 수치, 이전 통과율 등)이 빠져 있어, 실제 서비스에서의 노출 비율과는 차이가 있을
   수 있습니다.

### 다음 단계
`ANTHROPIC_API_KEY`를 설정하고 `python3 src/feedback/feedback.py`를 실행해 실제 결과를
생성한 뒤, 정답 노출 비율이 목표(10% 이하)를 달성하는지 확인하고 이 섹션의 "결과 요약"을
실측치로 갱신해야 합니다.

---

## 전체 파이프라인 실행 방법 (모듈 ①→②→③→④ 통합, E2E)

### 파일 구성
- `src/e2e_pipeline.py` — 학생 제출 1건을 모듈①→②→③→④ 순서로 메모리상에서 체이닝하는
  `run_pipeline()` 단일 진입 함수 + 배치 재생(`run_batch`) + 통합 평가(`build_e2e_evaluation_report`)
- `results/e2e_output.json` — `module1_output.json`의 전체 제출 이력(90건)을 학생별
  timestep 순으로 재생한 결과 (아래 결과 요약 참고)
- `results/e2e_evaluation_report.json` — 모듈②③④ 개별 평가를 하나로 합친 리포트
- `tests/test_e2e_pipeline.py` — pytest 통합 테스트 4건

### 실행 방법
```bash
pip install pandas numpy scikit-learn openpyxl anthropic pytest
export ANTHROPIC_API_KEY=sk-ant-...        # 모듈④까지 실제로 호출하려면 필요 (없어도 죽지 않음)
python3 src/e2e_pipeline.py                # 배치 실행 (results/e2e_output.json, e2e_evaluation_report.json 생성)
python3 src/e2e_pipeline.py --limit 20     # 모듈④ LLM 호출 비용을 제한하고 싶을 때
pytest tests/test_e2e_pipeline.py -v       # 통합 테스트 (API 키 없이 통과, javac는 필요)
```
필요한 데이터: `results/module1_output.json`(이미 커밋됨), `data/problem_meta.json`(모듈③
산출물, 이미 커밋됨), `data/concept_cards.json`(이미 커밋됨), `data/raw/모듈3_4용_데이터셋.xlsx`
(평가 리포트의 모듈③ 재계산용). 예상 실행 시간: 이 저장소 기준(제출 90건, 컴파일 재실행 없음)
API 키 없이 수 초, API 키가 있으면 모듈④가 학생당 최대 2회 Claude를 호출하므로 호출 대상
건수 × 왕복 지연시간만큼 늘어납니다.

### 결과 요약
`results/e2e_output.json`(90건) 중 **4건이 4개 모듈을 모두 성공적으로 통과**했고(`status: "ok"`),
**86건은 모듈④에서 `ANTHROPIC_API_KEY`가 없어 실패**로 기록됐습니다(`status: "partial_failure"`,
`module4.error`에 원인 명시, 프로세스는 죽지 않고 다음 제출로 계속 진행). 성공한 4건은 모두
파일럿 12문항을 이미 다 푼 시점 이후의 제출이라 모듈④가 LLM 없이 정형 문구(`NO_CANDIDATE_FEEDBACK`)
로 처리된 경우입니다 — 즉 API 키 없이도 "4개 모듈을 모두 거친 완전한 결과"가 실제로 나온다는
것을 확인했지만, 취약 개념이 있고 추천 문제가 있는 정상 케이스의 실제 LLM 피드백은 API 키가
있어야 나옵니다. `results/e2e_evaluation_report.json`은 모듈②(Macro-F1 0.7778, 목표 달성)·
모듈③(Recall@3 0.7193, 목표 달성)은 기존 산출물을 재사용/재계산해 채웠고, 모듈④는
`ANTHROPIC_API_KEY`가 없어 `target_achieved: null`로 남아 있습니다.

### 통합 시 발견된 이슈
1. **컬럼명 컨벤션 불일치**: `new_submission`은 원본 데이터 컨벤션(`Code`, `binary_score`,
   `ProblemID`, PascalCase)을 쓰지만 모듈①②③ 내부는 `problem_id`/`test_pass_rate` 같은
   snake_case를 씁니다. `_normalize_submission()`에서 흡수했고, 모듈 코드는 수정하지
   않았습니다.
2. **모듈②가 필요로 하는 개념 태그가 모듈①출력에는 없음**: `compute_concept_scores`는 제출
   행마다 `개념군_매핑` 태그가 있어야 하는데, `module1_output.json` 스키마에는 이 태그가
   전혀 없습니다(원래 별도 `ConceptTags`/`개념군_매핑` 컬럼에서 왔습니다). `_history_to_concept_df()`
   가 `problem_meta`의 `concept_groups`로 `problem_id → 개념 태그` 역매핑을 만들어 메웠습니다.
   다만 `problem_meta`(파일럿 12문항)에 없는 문제(제외된 5개 조건문 단독 문항 등)는 개념
   신호가 소실됩니다 — 모듈③ 자체 평가(`recommender.py`)는 별도의 "전체 17문항 concept map"
   으로 이 문제를 우회했지만, `run_pipeline`은 `problem_meta` 인자 하나만 받는 요청 스펙이라
   같은 우회를 적용할 채널이 없습니다. **다음 팀원이 볼 점**: 전체 문항의 concept map을 별도
   테이블로 유지하거나, `run_pipeline`에 옵션 인자를 추가하는 것을 검토하면 좋겠습니다.
3. **`timestep`이 `new_submission` 스펙에 없음**: 모듈②가 시간 가중치 계산에 `timestep`을
   필수로 요구하는데, 요청된 `new_submission` 스키마에는 없습니다. 이력의 마지막 `timestep+1`을
   자동 배정해 "가장 최근 제출"로 간주했습니다(호출자가 명시하면 그 값을 우선).
4. **모듈③이 두 개의 인자(문항 리스트 + 선수개념 맵)를 기대**: `recommend_problem()`은
   `problem_meta`(리스트)와 `concept_prerequisites`(dict)를 별도로 받지만, 통합 함수
   스펙은 `problem_meta` 하나뿐입니다. `data/problem_meta.json`과 같은 번들 구조
   (`{"problems": [...], "concept_prerequisites": {...}}`)라고 간주하고 내부에서 분리했습니다.
5. **배치 재생 모드에는 원본 코드가 없음**: `results/module1_output.json`에는 컴파일
   판정에 필요한 원본 `Code`/`binary_score`가 없습니다(원본 `data/raw/모듈1_2용_데이터셋.xlsx`가
   이 저장소에 없음 — 모듈①② 섹션 참고). 그래서 배치 모드에서는 모듈①을 실제로 재컴파일하지
   않고 이미 계산된 결과를 그대로 재사용합니다. `run_pipeline`은 `new_submission`에 `Code`
   키가 있는지로 두 모드를 자동 판별합니다 — 원본 xlsx가 있는 환경에서 raw 제출을 넘기면
   모듈①이 실제로 재실행됩니다.
6. **모듈④ 목표 검증은 API 키에 의존적**: 위 결과 요약대로, `ANTHROPIC_API_KEY`가 없는
   환경에서는 정답 노출 비율을 실측할 수 없습니다. `build_e2e_evaluation_report()`는 키가
   없으면 기존 `results/module4_evaluation.json`을 재사용하도록 폴백을 넣었지만, 이 파일도
   아직 없어 `module4.target_achieved`는 `null`입니다.
