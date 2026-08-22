# -*- coding: utf-8 -*-
"""
전체 파이프라인 오케스트레이터 — 모듈 ①→②→③을 순서대로 실행한다.
모듈별 실제 로직은 아래 패키지에 있고, 이 파일은 그것들을 불러와 순서대로 실행하고
results/·data/ 아래에 산출물을 저장하는 역할만 한다.
  - src/submission_analysis/ — ① 제출 분석 (컴파일 판정 + 실패 패턴 태깅)
  - src/weak_concept/        — ② 취약 개념 추정
  - src/recommender/         — ③ 보완 문제 추천
  - src/feedback/            — ④ 개념카드·LLM 피드백 (아직 미구현)

입력: data/raw/모듈1_2용_데이터셋.xlsx, data/raw/모듈3_4용_데이터셋.xlsx
출력: results/module1_output.json, results/module2_output.json, results/evaluation_report.json,
      data/problem_meta.json, results/module3_output.json, results/module3_evaluation.json

주의: data/raw/모듈1_2용_데이터셋.xlsx는 이 저장소에 커밋돼 있지 않을 수 있다(이미
module1/2_output.json으로 처리되어 원본 파일만 별도 보관 중일 가능성). 그 경우
모듈①②를 새로 돌리는 대신 기존 results/module1_output.json, module2_output.json을
그대로 재사용해 모듈③까지 이어서 실행한다.
"""

import json
import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SRC_DIR)
for _pkg in ("submission_analysis", "weak_concept", "recommender"):
    sys.path.insert(0, os.path.join(_SRC_DIR, _pkg))

from submission_analysis import build_module1_output, load_data  # noqa: E402
from weak_concept import build_module2_output, evaluate_macro_f1, tune_hyperparams  # noqa: E402
from recommender import (  # noqa: E402
    build_module3_output,
    build_problem_concept_map,
    build_problem_meta,
    evaluate_recommenders,
    load_problem_pool,
)

MODULE1_2_DATA_PATH = os.path.join(ROOT, "data", "raw", "모듈1_2용_데이터셋.xlsx")
MODULE3_4_DATA_PATH = os.path.join(ROOT, "data", "raw", "모듈3_4용_데이터셋.xlsx")

MODULE1_OUTPUT_PATH = os.path.join(ROOT, "results", "module1_output.json")
MODULE2_OUTPUT_PATH = os.path.join(ROOT, "results", "module2_output.json")
EVAL_REPORT_PATH = os.path.join(ROOT, "results", "evaluation_report.json")
PROBLEM_META_PATH = os.path.join(ROOT, "data", "problem_meta.json")
MODULE3_OUTPUT_PATH = os.path.join(ROOT, "results", "module3_output.json")
MODULE3_EVAL_PATH = os.path.join(ROOT, "results", "module3_evaluation.json")


def run_module1_2():
    """모듈 ①②를 원본 데이터로 처음부터 실행. 반환: (module1_outputs, module2_outputs)."""
    print("[1/7] 모듈①②용 데이터 로드 중...")
    df = load_data(MODULE1_2_DATA_PATH)
    print(f"  총 {len(df)}건 제출, 학생 {df['SubjectID'].nunique()}명")

    print("[2/7] 모듈 ① 실행 중 (Java 컴파일 검사 포함, 시간이 걸릴 수 있음)...")
    module1_outputs = build_module1_output(df)
    with open(MODULE1_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(module1_outputs, f, ensure_ascii=False, indent=2)
    n_compile_fail = sum(1 for o in module1_outputs if not o["compile_ok"])
    print(f"  완료: {len(module1_outputs)}건 처리, 컴파일 실패 {n_compile_fail}건")

    print("[3/7] 모듈 ② 실행 중...")
    module2_outputs = build_module2_output(df)
    with open(MODULE2_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(module2_outputs, f, ensure_ascii=False, indent=2)
    print(f"  완료: 학생 {len(module2_outputs)}명 취약도 산출")

    print("[4/7] 모듈② 하이퍼파라미터 튜닝 및 평가 중...")
    default_eval = evaluate_macro_f1(df)
    best_eval = tune_hyperparams(df)
    evaluation_report = {
        "default_params": default_eval,
        "best_params": best_eval,
        "target_macro_f1": 0.65,
        "target_achieved": bool(best_eval and best_eval["macro_f1"] >= 0.65),
    }
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
    print(f"  기본 파라미터 Macro-F1: {default_eval.get('macro_f1')}")
    print(f"  최적 파라미터 Macro-F1: {best_eval.get('macro_f1') if best_eval else None}")

    return module1_outputs, module2_outputs


def reuse_module1_2_outputs():
    """원본 데이터셋이 없을 때, 이미 저장소에 커밋된 결과를 그대로 불러온다."""
    print("[1-4/7] data/raw/모듈1_2용_데이터셋.xlsx가 없어 모듈①②를 건너뛰고 "
          "기존 results/module1_output.json · module2_output.json을 재사용합니다.")
    with open(MODULE1_OUTPUT_PATH, encoding="utf-8") as f:
        module1_outputs = json.load(f)
    with open(MODULE2_OUTPUT_PATH, encoding="utf-8") as f:
        module2_outputs = json.load(f)
    print(f"  제출 이력 {len(module1_outputs)}건, 학생 {len(module2_outputs)}명 로드")
    return module1_outputs, module2_outputs


def run_module3(module1_outputs, module2_outputs):
    print("[5/7] 모듈③ 문항 메타데이터 테이블 구축 중...")
    problem_meta_bundle = build_problem_meta(MODULE3_4_DATA_PATH)
    with open(PROBLEM_META_PATH, "w", encoding="utf-8") as f:
        json.dump(problem_meta_bundle, f, ensure_ascii=False, indent=2)
    print(f"  완료: 파일럿 문항 {len(problem_meta_bundle['problems'])}개")

    print("[6/7] 모듈 ③ 추천 실행 중...")
    module3_outputs = build_module3_output(
        module1_outputs, module2_outputs,
        problem_meta_bundle["problems"], problem_meta_bundle["concept_prerequisites"],
    )
    with open(MODULE3_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(module3_outputs, f, ensure_ascii=False, indent=2)
    print(f"  완료: 학생 {len(module3_outputs)}명 추천 생성")

    print("[7/7] 모듈③ hold-out 평가 실행 중 (제안 시스템 vs 베이스라인 3종)...")
    full_concept_map = build_problem_concept_map(load_problem_pool(MODULE3_4_DATA_PATH))
    evaluation = evaluate_recommenders(module1_outputs, problem_meta_bundle, full_concept_map)
    with open(MODULE3_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    print(f"  평가 포인트 {evaluation['n_eval_points']}개")
    for m, s in evaluation["methods"].items():
        print(f"  {m}: Recall@1={s['recall@1']} Recall@3={s['recall@3']} NDCG@3={s['ndcg@3']}")
    print(f"  목표(Recall@3>=0.70) 달성 여부: {evaluation['target_achieved']}")


def main():
    if os.path.exists(MODULE1_2_DATA_PATH):
        module1_outputs, module2_outputs = run_module1_2()
    else:
        module1_outputs, module2_outputs = reuse_module1_2_outputs()

    run_module3(module1_outputs, module2_outputs)

    print("완료. results/ · data/ 아래 산출물이 모두 생성되었습니다.")


if __name__ == "__main__":
    main()
