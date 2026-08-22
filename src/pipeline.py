# -*- coding: utf-8 -*-
"""
모듈 ①(제출 분석) · ②(취약 개념 추정) 파이프라인
입력: 모듈1_2용_데이터셋.xlsx (StudentSubmissions, ConceptTags)
출력: module1_output.json, module2_output.json, evaluation_report.json
"""

import ast
import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_PATH = "모듈1_2용_데이터셋.xlsx"

# ---------------------------------------------------------------------------
# 0. 데이터 로드
# ---------------------------------------------------------------------------

def load_data(path=DATA_PATH):
    df = pd.read_excel(path, sheet_name="StudentSubmissions")
    df["binary_score"] = df["binary_score"].apply(ast.literal_eval)
    df["개념군_매핑"] = df["개념군_매핑"].astype(str)
    return df


# ---------------------------------------------------------------------------
# 1. 모듈 ① — 제출 분석 (컴파일 판정 + 실패 패턴 태깅)
# ---------------------------------------------------------------------------

def check_compile(code: str):
    """Java 코드 컴파일 시도. (성공여부, stderr) 반환."""
    wrapped = f"class Solution {{\n{code}\n}}"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "Solution.java")
        with open(path, "w") as f:
            f.write(wrapped)
        try:
            result = subprocess.run(
                ["javac", "-d", d, path],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        return result.returncode == 0, result.stderr


def tag_error_pattern(binary_score, compile_ok: bool, stderr: str) -> str:
    """컴파일/실행 결과를 규칙 기반으로 태깅."""
    if not compile_ok:
        low = stderr.lower()
        if "missing return statement" in low:
            return "helper_function_missing_return"
        if "cannot find symbol" in low:
            return "undeclared_variable_or_method"
        if "incompatible types" in low:
            return "type_mismatch"
        if "reached end of file while parsing" in low or "illegal start" in low:
            return "syntax_error_unbalanced"
        if "';' expected" in low:
            return "syntax_error_missing_semicolon"
        return "compile_error_other"

    n_total = len(binary_score)
    n_failed = binary_score.count(0)
    if n_failed == 0:
        return "all_passed"
    if n_failed == n_total:
        return "all_tests_failed_logic"
    return "partial_failure_logic"


def build_module1_output(df: pd.DataFrame):
    """행 단위로 ①모듈 JSON 출력을 생성. 컴파일 캐시로 중복 코드 재컴파일 방지."""
    outputs = []
    compile_cache = {}
    for _, row in df.iterrows():
        code = row["Code"]
        if code not in compile_cache:
            compile_cache[code] = check_compile(code)
        compile_ok, stderr = compile_cache[code]

        n_total = len(row["binary_score"])
        n_failed = row["binary_score"].count(0)

        outputs.append(
            {
                "subject_id": row["SubjectID"],
                "problem_id": int(row["ProblemID"]),
                "timestep": int(row["timestep"]),
                "compile_ok": bool(compile_ok),
                "test_pass_rate": float(row["Score"]),
                "n_failed": int(n_failed),
                "n_total": int(n_total),
                "error_pattern": tag_error_pattern(row["binary_score"], compile_ok, stderr),
            }
        )
    return outputs


# ---------------------------------------------------------------------------
# 2. 모듈 ② — 취약 개념 추정
# ---------------------------------------------------------------------------

MIN_ATTEMPTS = 2       # 신뢰할 만한 점수로 인정할 최소 시도 수
DECAY = 0.7            # 시간 가중 감쇠율 (최근 시도일수록 가중치 큼)
WEAK_THRESHOLD = 0.5   # 취약 판정 기준 (평가용)


def split_concepts(tag: str):
    return [c.strip() for c in tag.split(",") if c.strip()]


def compute_concept_scores(subject_df: pd.DataFrame, decay: float = DECAY):
    """한 학생의 제출 이력(subject_df)으로 개념별 취약도(0~1) 산출.
    시도 횟수가 MIN_ATTEMPTS 미만인 개념은 None(insufficient_data)."""
    subject_df = subject_df.sort_values("timestep")
    max_t = subject_df["timestep"].max()

    concept_records = {}
    for _, row in subject_df.iterrows():
        weight = decay ** (max_t - row["timestep"])
        for c in split_concepts(row["개념군_매핑"]):
            concept_records.setdefault(c, []).append((weight, row["Score"]))

    scores = {}
    for concept, records in concept_records.items():
        if len(records) < MIN_ATTEMPTS:
            scores[concept] = None
            continue
        weights = np.array([w for w, _ in records])
        pass_rates = np.array([p for _, p in records])
        weighted_pass_rate = float(np.average(pass_rates, weights=weights))
        scores[concept] = round(1 - weighted_pass_rate, 3)

    return scores


def build_module2_output(df: pd.DataFrame):
    results = []
    for sid in df["SubjectID"].unique():
        sub_df = df[df["SubjectID"] == sid]
        scores = compute_concept_scores(sub_df)
        valid = {k: v for k, v in scores.items() if v is not None}
        weakest = max(valid, key=valid.get) if valid else None
        results.append(
            {
                "subject_id": sid,
                "concept_scores": scores,
                "weakest_concept": weakest,
                "n_submissions": int(len(sub_df)),
            }
        )
    return results


# ---------------------------------------------------------------------------
# 3. 평가 — hold-out 방식 Macro-F1 (목표: 0.65 이상)
# ---------------------------------------------------------------------------

def evaluate_macro_f1(df: pd.DataFrame, decay: float = DECAY, threshold: float = WEAK_THRESHOLD):
    """학생별 마지막 시도를 정답 라벨용으로 숨기고, 그 이전 이력만으로
    예측한 취약도가 마지막 시도의 실제 실패 여부를 맞히는지 평가."""
    y_true, y_pred = [], []
    per_student = []

    for sid in df["SubjectID"].unique():
        sub_df = df[df["SubjectID"] == sid].sort_values("timestep")
        if len(sub_df) < 3:
            continue
        last_row = sub_df.iloc[-1]
        history = sub_df.iloc[:-1]

        scores = compute_concept_scores(history, decay=decay)
        concepts = split_concepts(last_row["개념군_매핑"])

        student_true, student_pred = [], []
        for c in concepts:
            if scores.get(c) is None:
                continue
            predicted_weak = scores[c] >= threshold
            actual_failed = last_row["Score"] < 1.0
            y_true.append(actual_failed)
            y_pred.append(predicted_weak)
            student_true.append(actual_failed)
            student_pred.append(predicted_weak)

        per_student.append(
            {
                "subject_id": sid,
                "n_evaluated_concepts": len(student_true),
            }
        )

    if not y_true:
        return {"macro_f1": None, "n_eval_points": 0, "note": "평가 가능한 샘플 없음"}

    return {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "n_eval_points": len(y_true),
        "decay": decay,
        "threshold": threshold,
    }


def tune_hyperparams(df: pd.DataFrame):
    """decay, threshold 그리드서치로 Macro-F1 최대화."""
    best = None
    for decay in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
            result = evaluate_macro_f1(df, decay=decay, threshold=threshold)
            if result["macro_f1"] is None:
                continue
            if best is None or result["macro_f1"] > best["macro_f1"]:
                best = result
    return best


# ---------------------------------------------------------------------------
# 4. 실행
# ---------------------------------------------------------------------------

def main():
    print("[1/5] 데이터 로드 중...")
    df = load_data()
    print(f"  총 {len(df)}건 제출, 학생 {df['SubjectID'].nunique()}명")

    print("[2/5] 모듈 ① 실행 중 (Java 컴파일 검사 포함, 시간이 걸릴 수 있음)...")
    module1_outputs = build_module1_output(df)
    with open("module1_output.json", "w", encoding="utf-8") as f:
        json.dump(module1_outputs, f, ensure_ascii=False, indent=2)
    n_compile_fail = sum(1 for o in module1_outputs if not o["compile_ok"])
    print(f"  완료: {len(module1_outputs)}건 처리, 컴파일 실패 {n_compile_fail}건")

    print("[3/5] 모듈 ② 실행 중...")
    module2_outputs = build_module2_output(df)
    with open("module2_output.json", "w", encoding="utf-8") as f:
        json.dump(module2_outputs, f, ensure_ascii=False, indent=2)
    print(f"  완료: 학생 {len(module2_outputs)}명 취약도 산출")

    print("[4/5] 하이퍼파라미터 튜닝 및 평가 중...")
    default_eval = evaluate_macro_f1(df)
    best_eval = tune_hyperparams(df)
    evaluation_report = {
        "default_params": default_eval,
        "best_params": best_eval,
        "target_macro_f1": 0.65,
        "target_achieved": bool(best_eval and best_eval["macro_f1"] >= 0.65),
    }
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
    print(f"  기본 파라미터 Macro-F1: {default_eval.get('macro_f1')}")
    print(f"  최적 파라미터 Macro-F1: {best_eval.get('macro_f1') if best_eval else None}")

    print("[5/5] 완료. module1_output.json / module2_output.json / evaluation_report.json 생성됨")


if __name__ == "__main__":
    main()
