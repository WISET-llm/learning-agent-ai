# -*- coding: utf-8 -*-
"""
엔드투엔드 통합 파이프라인 — 학생 제출 1건이 모듈 ①→②→③→④를 순서대로 통과해
"추천 문제 + 피드백 텍스트"까지 나오는 단일 진입점(run_pipeline)을 제공한다.

지금까지 4개 모듈(submission_analysis/weak_concept/recommender/feedback)은 각자 파일을
읽고 각자 결과 json을 저장하는 독립 스크립트였다. 이 파일은 그 모듈들의 함수를 그대로
가져와 메모리에서 체이닝만 한다 — 모듈 코드 자체는 하나도 수정하지 않았고, 모듈 간
스키마 차이는 이 파일의 어댑터 함수(_normalize_submission, _history_to_concept_df 등)에서
흡수한다. 통합 과정에서 발견된 불일치는 파일 하단 "통합 시 발견된 이슈" 주석과
results/README.md에 정리했다.

입력: results/module1_output.json(제출 이력 재생용), data/problem_meta.json,
      data/concept_cards.json, data/raw/모듈3_4용_데이터셋.xlsx(평가 리포트용)
출력: results/e2e_output.json, results/e2e_evaluation_report.json
"""

import argparse
import json
import os
import sys

import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SRC_DIR)
for _pkg in ("submission_analysis", "weak_concept", "recommender", "feedback"):
    sys.path.insert(0, os.path.join(_SRC_DIR, _pkg))

from submission_analysis import check_compile, load_data, tag_error_pattern  # noqa: E402
from weak_concept import compute_concept_scores, evaluate_macro_f1, tune_hyperparams  # noqa: E402
from recommender import (  # noqa: E402
    build_problem_concept_map,
    build_problem_meta,
    evaluate_recommenders,
    load_problem_pool,
    recommend_problem,
)
from feedback import (  # noqa: E402
    DEFAULT_MODEL as FEEDBACK_DEFAULT_MODEL,
    NO_CANDIDATE_FEEDBACK,
    evaluate_feedback,
    evaluate_module4,
    generate_feedback,
    generate_problem_level_feedback,
    load_concept_cards,
)

MODULE1_2_DATA_PATH = os.path.join(ROOT, "data", "raw", "모듈1_2용_데이터셋.xlsx")
MODULE3_4_DATA_PATH = os.path.join(ROOT, "data", "raw", "모듈3_4용_데이터셋.xlsx")
MODULE1_OUTPUT_PATH = os.path.join(ROOT, "results", "module1_output.json")
EVAL_REPORT_PATH = os.path.join(ROOT, "results", "evaluation_report.json")
MODULE4_EVAL_PATH = os.path.join(ROOT, "results", "module4_evaluation.json")
PROBLEM_META_PATH = os.path.join(ROOT, "data", "problem_meta.json")
CONCEPT_CARDS_PATH = os.path.join(ROOT, "data", "concept_cards.json")
E2E_OUTPUT_PATH = os.path.join(ROOT, "results", "e2e_output.json")
E2E_EVAL_PATH = os.path.join(ROOT, "results", "e2e_evaluation_report.json")


# ---------------------------------------------------------------------------
# 어댑터 — 모듈 간 스키마 차이를 이 레이어에서만 흡수한다 (모듈 코드는 건드리지 않음)
# ---------------------------------------------------------------------------

def _normalize_submission(raw_submission: dict) -> dict:
    """new_submission을 내부 스키마로 정규화.

    두 형태를 모두 받는다:
    - 원본 제출(Code, binary_score, ProblemID 포함): 모듈①을 실제로 재실행한다.
    - 이미 분석된 모듈① 레코드(problem_id, compile_ok, test_pass_rate, ... — 예:
      module1_output.json의 레코드): 원본 Code가 없어 재컴파일이 불가능하므로, 계산된
      값을 그대로 재사용한다. (배치 재생 모드에서 이 경우가 쓰인다 — 아래 '통합 시
      발견된 이슈' #3 참고)
    """
    if "Code" in raw_submission:
        binary_score = raw_submission["binary_score"]
        n_total = len(binary_score)
        n_failed = binary_score.count(0)
        return {
            "problem_id": int(raw_submission["ProblemID"]),
            "code": raw_submission["Code"],
            "binary_score": binary_score,
            "n_total": n_total,
            "n_failed": n_failed,
            "test_pass_rate": (n_total - n_failed) / n_total if n_total else 0.0,
            "precomputed": None,
        }

    return {
        "problem_id": int(raw_submission["problem_id"]),
        "precomputed": {
            "compile_ok": raw_submission["compile_ok"],
            "test_pass_rate": raw_submission["test_pass_rate"],
            "n_failed": raw_submission["n_failed"],
            "n_total": raw_submission["n_total"],
            "error_pattern": raw_submission["error_pattern"],
        },
    }


def _history_to_concept_df(submission_history, new_row, problem_concept_map):
    """모듈②의 compute_concept_scores가 기대하는 스키마(timestep, 개념군_매핑, Score)로
    제출 이력 + 새 제출을 합쳐 DataFrame을 만든다.

    모듈①출력 스키마(module1_output.json)에는 개념 태그가 없으므로, problem_meta의
    concept_groups로 problem_id -> 개념 태그를 역매핑한다. problem_meta(파일럿 문항)에
    없는 problem_id는 개념 신호 없이 건너뛴다 — '통합 시 발견된 이슈' #2 참고.
    """
    records = []
    for row in list(submission_history) + [new_row]:
        concept_tag = problem_concept_map.get(row["problem_id"])
        if concept_tag is None:
            continue
        records.append(
            {
                "timestep": row["timestep"],
                "개념군_매핑": concept_tag,
                "Score": row["test_pass_rate"],
            }
        )
    if not records:
        return pd.DataFrame(columns=["timestep", "개념군_매핑", "Score"])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 모듈별 단일-제출 실행 (각 모듈의 기존 함수를 그대로 호출)
# ---------------------------------------------------------------------------

def _run_module1(new_submission, submission_history):
    norm = _normalize_submission(new_submission)
    timestep = new_submission.get("timestep")
    if timestep is None:
        timestep = max((h["timestep"] for h in submission_history), default=-1) + 1

    if norm["precomputed"] is not None:
        p = norm["precomputed"]
        return {
            "problem_id": norm["problem_id"],
            "timestep": timestep,
            "compile_ok": p["compile_ok"],
            "test_pass_rate": p["test_pass_rate"],
            "n_failed": p["n_failed"],
            "n_total": p["n_total"],
            "error_pattern": p["error_pattern"],
        }

    compile_ok, stderr = check_compile(norm["code"])
    error_pattern = tag_error_pattern(norm["binary_score"], compile_ok, stderr)
    return {
        "problem_id": norm["problem_id"],
        "timestep": timestep,
        "compile_ok": compile_ok,
        "test_pass_rate": norm["test_pass_rate"],
        "n_failed": norm["n_failed"],
        "n_total": norm["n_total"],
        "error_pattern": error_pattern,
    }


def _run_module2(module1_result, submission_history, problem_concept_map):
    new_row = {
        "timestep": module1_result["timestep"],
        "problem_id": module1_result["problem_id"],
        "test_pass_rate": module1_result["test_pass_rate"],
    }
    concept_df = _history_to_concept_df(submission_history, new_row, problem_concept_map)
    concept_scores = compute_concept_scores(concept_df) if not concept_df.empty else {}
    valid_scores = {k: v for k, v in concept_scores.items() if v is not None}
    weakest_concept = max(valid_scores, key=valid_scores.get) if valid_scores else None
    return {"concept_scores": concept_scores, "weakest_concept": weakest_concept}


def _run_module3(module1_result, module2_result, submission_history, problem_meta):
    solved_problem_ids = {h["problem_id"] for h in submission_history} | {module1_result["problem_id"]}
    weakest_concept = module2_result["weakest_concept"]
    concept_scores = module2_result["concept_scores"]

    recent_pass_rate = None
    if weakest_concept is not None and concept_scores.get(weakest_concept) is not None:
        recent_pass_rate = 1 - concept_scores[weakest_concept]

    rec = recommend_problem(
        weakest_concept, solved_problem_ids, problem_meta["problems"],
        problem_meta.get("concept_prerequisites", {}), recent_pass_rate,
    )
    if rec is None:
        return {
            "recommended_problem_id": None,
            "recommendation_type": "no_candidate",
            "reason": "파일럿 문항 풀을 모두 해결하여 추천할 문제가 없음",
        }
    return rec


def _run_module4(weakest_concept, concept_scores, module3_result, problem_meta_by_id, concept_cards,
                  model, client):
    if module3_result["recommended_problem_id"] is None:
        return {
            "feedback_text": NO_CANDIDATE_FEEDBACK,
            "direct_answer_leak": False,
            "judge_reasoning": "파일럿 문항을 모두 해결해 정형 문구를 사용, LLM 미호출",
        }

    concept_card = concept_cards.get(weakest_concept)
    feedback_text = generate_feedback(
        weakest_concept, concept_scores, module3_result["recommended_problem_id"],
        module3_result["reason"], concept_card, model=model, client=client,
    )
    problem = problem_meta_by_id.get(module3_result["recommended_problem_id"])
    requirement = problem["requirement"] if problem else ""
    eval_result = evaluate_feedback(feedback_text, requirement, model=model, client=client)
    return {
        "feedback_text": feedback_text,
        "direct_answer_leak": eval_result["direct_answer_leak"],
        "judge_reasoning": eval_result["judge_reasoning"],
    }


# ---------------------------------------------------------------------------
# 단일 진입점
# ---------------------------------------------------------------------------

def run_pipeline(subject_id, new_submission, submission_history, problem_meta, concept_cards,
                  model: str = FEEDBACK_DEFAULT_MODEL, client=None) -> dict:
    """학생 제출 1건을 모듈 ①→②→③→④ 순서로 통과시켜 통합 결과를 반환한다.

    new_submission: 최소 Code, binary_score, ProblemID를 포함하는 dict. timestep을 명시하지
        않으면 submission_history의 마지막 timestep+1로 자동 배정한다(가장 최근 제출로 간주).
        원본 Code 없이 이미 분석된 모듈①레코드(problem_id/compile_ok/test_pass_rate/...)를
        넘겨도 동작한다 — 이 경우 모듈①을 재실행하지 않고 주어진 값을 그대로 쓴다.
    submission_history: 이 학생의 과거 제출 이력(module1_output.json 레코드 스키마의
        dict 리스트: problem_id, timestep, test_pass_rate, ...). 첫 제출이면 빈 리스트.
    problem_meta: build_problem_meta()/data/problem_meta.json과 같은 스키마
        ({"problems": [...], "concept_prerequisites": {...}}).
    concept_cards: load_concept_cards()가 반환하는 {개념군: 카드dict}.
    client: 모듈④(Claude API) 호출에 쓸 anthropic 클라이언트. None이면 실제 클라이언트를
        새로 만든다 — 테스트에서는 반드시 목(mock) 객체를 넘겨야 한다.

    반환값은 {"subject_id", "status", "module1", "module2", "module3", "module4"}이며 항상
    이 5개 키를 갖는다. 어느 단계가 실패해도 예외를 올리지 않고 해당 모듈 키에
    {"error": ..., "stage": ...}를 기록하고, 그 뒤 단계는 {"skipped": True, "reason": ...}로
    표시한다(뒷단계가 앞단계 출력에 의존하므로) — status는 "ok" 또는 "partial_failure".
    """
    result = {"subject_id": subject_id, "status": "ok",
              "module1": None, "module2": None, "module3": None, "module4": None}

    problem_meta_by_id = {p["problem_id"]: p for p in problem_meta["problems"]}
    problem_concept_map = {p["problem_id"]: ", ".join(p["concept_groups"]) for p in problem_meta["problems"]}

    try:
        result["module1"] = _run_module1(new_submission, submission_history)
    except Exception as e:
        result["module1"] = {"error": str(e), "stage": "module1"}
        result["status"] = "partial_failure"

    if "error" in result["module1"]:
        result["module2"] = {"skipped": True, "reason": "module1 실패로 건너뜀"}
    else:
        try:
            result["module2"] = _run_module2(result["module1"], submission_history, problem_concept_map)
        except Exception as e:
            result["module2"] = {"error": str(e), "stage": "module2"}
            result["status"] = "partial_failure"

    if "error" in result["module2"] or "skipped" in result["module2"]:
        result["module3"] = {"skipped": True, "reason": "module2 실패로 건너뜀"}
    else:
        try:
            result["module3"] = _run_module3(result["module1"], result["module2"], submission_history, problem_meta)
        except Exception as e:
            result["module3"] = {"error": str(e), "stage": "module3"}
            result["status"] = "partial_failure"

    if "error" in result["module3"] or "skipped" in result["module3"]:
        result["module4"] = {"skipped": True, "reason": "module3 실패로 건너뜀"}
    else:
        try:
            result["module4"] = _run_module4(
                result["module2"]["weakest_concept"], result["module2"]["concept_scores"],
                result["module3"], problem_meta_by_id, concept_cards, model, client,
            )
        except Exception as e:
            result["module4"] = {"error": str(e), "stage": "module4"}
            result["status"] = "partial_failure"

    return result


# ---------------------------------------------------------------------------
# 배치 실행 — 학생별 제출 이력을 timestep 순으로 재생
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_batch(limit=None, model: str = FEEDBACK_DEFAULT_MODEL, client=None):
    """results/module1_output.json의 전체 학생 제출 이력을 timestep 순으로 하나씩
    흘려보내며 run_pipeline을 반복 호출한다.

    이 저장소에는 원본 코드(Code/binary_score)가 없는 module1_output.json만 있어(원본
    모듈1_2용_데이터셋.xlsx 미보유), 배치 모드에서는 모듈①을 실제로 재컴파일하지 않고
    이미 계산된 결과를 재사용한다 — run_pipeline이 new_submission에 'Code' 키가 없으면
    자동으로 이 모드로 동작한다. 원본 xlsx가 있는 환경에서는 StudentSubmissions 시트의
    행을 그대로 new_submission으로 넘기면 모듈①이 실제로 재실행된다.

    limit: 처리할 최대 제출 건수. 모듈④(Claude API) 호출 비용을 제한하는 용도 —
    무제한으로 두면 학생 수 × 제출 건수만큼 LLM을 호출하게 된다.
    """
    module1_records = _load_json(MODULE1_OUTPUT_PATH)
    problem_meta = _load_json(PROBLEM_META_PATH)
    concept_cards = load_concept_cards(CONCEPT_CARDS_PATH)

    by_subject = {}
    for r in module1_records:
        by_subject.setdefault(r["subject_id"], []).append(r)
    for records in by_subject.values():
        records.sort(key=lambda r: r["timestep"])

    results = []
    n_calls = 0
    for sid, records in by_subject.items():
        history = []
        for record in records:
            if limit is not None and n_calls >= limit:
                break
            result = run_pipeline(sid, record, history, problem_meta, concept_cards, model=model, client=client)
            results.append(result)
            history.append(record)
            n_calls += 1
        if limit is not None and n_calls >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# 통합 평가 리포트 — 모듈②③④ 개별 평가를 한 번에 실행해 하나로 합침
# ---------------------------------------------------------------------------

def build_e2e_evaluation_report():
    """모듈②(Macro-F1) / 모듈③(Recall@3) / 모듈④(정답 노출 비율) 평가를 한 번에 실행해
    하나의 리포트로 합친다. 새 평가 지표를 만들지 않고 각 모듈이 이미 갖고 있는 평가
    함수를 그대로 재사용한다."""
    report = {}

    if os.path.exists(MODULE1_2_DATA_PATH):
        df = load_data(MODULE1_2_DATA_PATH)
        default_eval = evaluate_macro_f1(df)
        best_eval = tune_hyperparams(df)
        report["module2"] = {
            "default_params": default_eval,
            "best_params": best_eval,
            "target_macro_f1": 0.65,
            "target_achieved": bool(best_eval and best_eval["macro_f1"] >= 0.65),
        }
    elif os.path.exists(EVAL_REPORT_PATH):
        report["module2"] = _load_json(EVAL_REPORT_PATH)
        report["module2"]["_source"] = "원본 데이터셋 없음 — 기존 results/evaluation_report.json 재사용"
    else:
        report["module2"] = {"error": "평가 불가 — 원본 데이터셋과 기존 evaluation_report.json 모두 없음"}

    problem_meta_bundle = build_problem_meta(MODULE3_4_DATA_PATH)
    module1_records = _load_json(MODULE1_OUTPUT_PATH)
    full_concept_map = build_problem_concept_map(load_problem_pool(MODULE3_4_DATA_PATH))
    report["module3"] = evaluate_recommenders(module1_records, problem_meta_bundle, full_concept_map)

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        concept_cards = load_concept_cards(CONCEPT_CARDS_PATH)
        problem_meta_by_id = {p["problem_id"]: p for p in problem_meta_bundle["problems"]}
        problem_level_records = generate_problem_level_feedback(problem_meta_bundle["problems"], concept_cards)
        report["module4"] = evaluate_module4(problem_level_records, problem_meta_by_id)
    elif os.path.exists(MODULE4_EVAL_PATH):
        report["module4"] = _load_json(MODULE4_EVAL_PATH)
        report["module4"]["_source"] = "ANTHROPIC_API_KEY 없음 — 기존 results/module4_evaluation.json 재사용"
    else:
        report["module4"] = {"error": "평가 불가 — ANTHROPIC_API_KEY 없음, 기존 module4_evaluation.json도 없음"}

    achieved = {m: report[m].get("target_achieved") for m in ("module2", "module3", "module4")}
    report["summary"] = {
        "module2_target_achieved": achieved["module2"],
        "module3_target_achieved": achieved["module3"],
        "module4_target_achieved": achieved["module4"],
        "all_targets_achieved": (
            all(achieved.values()) if all(v is not None for v in achieved.values()) else None
        ),
    }
    return report


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="모듈 ①→②→③→④ 통합 파이프라인 배치 실행")
    parser.add_argument("--limit", type=int, default=None,
                         help="처리할 최대 제출 건수 (모듈④ LLM 호출 비용 제한용)")
    args = parser.parse_args()

    print("[1/2] 배치 실행 중 (module1_output.json 제출 이력을 학생별 timestep 순으로 재생)...")
    results = run_batch(limit=args.limit)
    with open(E2E_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"  완료: {len(results)}건 처리 ({n_ok}건 4개 모듈 모두 성공, {len(results) - n_ok}건 일부 단계 실패/스킵)")

    print("[2/2] 통합 평가 리포트 생성 중 (모듈②③④ 평가를 재사용해 병합)...")
    evaluation = build_e2e_evaluation_report()
    with open(E2E_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    print(f"  모듈② 목표 달성: {evaluation['summary']['module2_target_achieved']}")
    print(f"  모듈③ 목표 달성: {evaluation['summary']['module3_target_achieved']}")
    print(f"  모듈④ 목표 달성: {evaluation['summary']['module4_target_achieved']}")
    print("완료. results/e2e_output.json / results/e2e_evaluation_report.json 생성됨")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 통합 시 발견된 이슈 (results/README.md에도 동일 내용 정리)
#
# 1. new_submission의 컬럼명(Code/binary_score/ProblemID, PascalCase)이 모듈①②③ 내부
#    스키마(code/problem_id/test_pass_rate, snake_case)와 다르다. _normalize_submission()
#    에서 흡수했다.
# 2. 모듈②의 compute_concept_scores는 제출 행마다 '개념군_매핑' 태그가 있어야 하는데,
#    module1_output.json 스키마에는 개념 태그가 전혀 없다(원래 별도 ConceptTags 시트에서
#    왔음). _history_to_concept_df()가 problem_meta의 concept_groups로 problem_id->개념
#    태그 역매핑을 만들어 채운다. 단, problem_meta(파일럿 12문항)에 없는 problem_id(파일럿
#    에서 제외된 5개 조건문 단독 문항 등)는 개념 신호가 소실된다 — recommender.py가 자체
#    평가에서는 별도의 '전체 17문항 concept map'으로 이 문제를 우회했지만, run_pipeline은
#    problem_meta 인자 하나만 받는 요청 시그니처라 같은 우회를 적용할 채널이 없다. 전체
#    문항의 concept map을 별도로 유지하는 것이 다음 개선 포인트다.
# 3. compute_concept_scores는 'timestep' 컬럼이 필수인데 new_submission 스펙에는 timestep이
#    없다. run_pipeline은 submission_history의 마지막 timestep+1을 자동 배정한다(새 제출이
#    가장 최근이라고 가정) — 호출자가 timestep을 직접 넘기면 그 값을 우선한다.
# 4. 모듈③의 recommend_problem(weakest_concept, solved_ids, problem_meta, concept_prereqs,
#    recent_pass_rate)은 문항 리스트와 선수개념 맵을 별도 인자로 받지만, run_pipeline의
#    요청 시그니처는 problem_meta 하나뿐이다. problem_meta.json과 같은 번들 구조
#    ({"problems": [...], "concept_prerequisites": {...}})라고 간주하고 내부에서 분리해
#    넘긴다.
# 5. 배치 재생 모드(results/module1_output.json)에는 원본 Code/binary_score가 없어 모듈①을
#    실제로 재컴파일할 수 없다(원본 모듈1_2용_데이터셋.xlsx가 이 저장소에 없음). 이미 계산된
#    모듈①결과를 그대로 재사용하도록 _normalize_submission()이 두 입력 형태를 모두
#    받아들이게 했다 — 원본 xlsx가 있는 환경에서 raw Code를 넘기면 모듈①이 실제로
#    재실행된다.
# ---------------------------------------------------------------------------
