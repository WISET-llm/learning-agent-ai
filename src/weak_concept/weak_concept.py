# -*- coding: utf-8 -*-
"""
모듈 ② 취약 개념 추정
입력: 모듈 ①의 출력(오류 요약) 또는 원본 제출 이력(timestep, 개념군_매핑, Score)
출력: 학생별 개념군 취약도 점수 + weakest_concept
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

MIN_ATTEMPTS = 2       # 신뢰할 만한 점수로 인정할 최소 시도 수
DECAY = 0.7            # 시간 가중 감쇠율 (최근 시도일수록 가중치 큼)
WEAK_THRESHOLD = 0.5   # 취약 판정 기준 (평가용)


def split_concepts(tag: str):
    return [c.strip() for c in tag.split(",") if c.strip()]


def compute_concept_scores(subject_df, decay: float = DECAY):
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


def build_module2_output(df):
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
# 평가 — hold-out 방식 Macro-F1 (목표: 0.65 이상)
# ---------------------------------------------------------------------------

def evaluate_macro_f1(df, decay: float = DECAY, threshold: float = WEAK_THRESHOLD):
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


def tune_hyperparams(df):
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
