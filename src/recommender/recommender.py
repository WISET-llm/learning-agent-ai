# -*- coding: utf-8 -*-
"""
모듈 ③ 보완 문제 추천 파이프라인
입력: data/raw/모듈3_4용_데이터셋.xlsx (Real_50Problems, Real_17_TestCased)
      results/module1_output.json (학생별 제출 이력 재구성 — hold-out 평가용)
      results/module2_output.json (weakest_concept, concept_scores)
출력: data/problem_meta.json, results/module3_output.json, results/module3_evaluation.json

모듈 ①②(submission_analysis/weak_concept)와 다른 입력 데이터셋을 사용하므로 패키지를
분리했으며, 개념별 취약도 계산 로직(compute_concept_scores 등)은 weak_concept 모듈 것을
그대로 재사용한다.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weak_concept"))
from weak_concept import compute_concept_scores, split_concepts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, "data", "raw", "모듈3_4용_데이터셋.xlsx")
MODULE1_OUTPUT_PATH = os.path.join(ROOT, "results", "module1_output.json")
MODULE2_OUTPUT_PATH = os.path.join(ROOT, "results", "module2_output.json")
PROBLEM_META_PATH = os.path.join(ROOT, "data", "problem_meta.json")
MODULE3_OUTPUT_PATH = os.path.join(ROOT, "results", "module3_output.json")
MODULE3_EVAL_PATH = os.path.join(ROOT, "results", "module3_evaluation.json")

KC_COLS = [
    "If/Else", "NestedIf", "While", "For", "NestedFor", "Math+-*/", "Math%",
    "LogicAndNotOr", "LogicCompareNum", "LogicBoolean", "StringFormat",
    "StringConcat", "StringIndex", "StringLen", "StringEqual", "CharEqual",
    "ArrayIndex", "DefFunction",
]

# ---------------------------------------------------------------------------
# 0. 문항 메타데이터 테이블 구축
#    (Real_17_TestCased 중 파일럿 문항을 선정하고, Real_50Problems의 KC 원-핫 컬럼으로
#     난이도·선수개념을 "재정의"한다. ProblemMeta_난이도표_예시 시트는 값이 가상이라
#     구조만 참고하고 실제 값은 여기서 새로 채운다.)
# ---------------------------------------------------------------------------

# 난이도 3단계 임계값: 활성 KC 플래그 개수(n_kc, 문항 복잡도 proxy) 기준.
# 실제 17문항의 n_kc 분포는 3~8(최빈값 5)이며, 조건문 단독 문항은 대체로 n_kc 3~6,
# 반복문·배열/문자열이 결합된 문항은 n_kc 5~8로 자연스럽게 더 높게 몰린다.
# 이를 반영해 하/중/상 경계를 다음과 같이 고정했다.
DIFFICULTY_LOW_MAX = 4   # n_kc <= 4 -> 하
DIFFICULTY_HIGH_MIN = 7  # n_kc >= 7 -> 상 (5~6은 중)


def load_problem_pool(path=DATA_PATH):
    """Real_50Problems(KC 원-핫)와 Real_17_TestCased(실제 테스트케이스 보유 문항)를
    ProblemID 기준으로 병합해, 실제 17문항의 KC 플래그·개념군 매핑을 로드."""
    p50 = pd.read_excel(path, sheet_name="Real_50Problems")
    p17 = pd.read_excel(path, sheet_name="Real_17_TestCased")
    p17 = p17.rename(
        columns={
            "실제 문제 설명(Requirement)": "Requirement",
            "개념군 매핑": "개념군_매핑",
        }
    )
    merged = p17.merge(
        p50[["AssignmentID", "ProblemID"] + KC_COLS], on=["AssignmentID", "ProblemID"], how="left"
    )
    merged["n_kc"] = merged[KC_COLS].notna().sum(axis=1)
    return merged


def build_problem_concept_map(pool: pd.DataFrame) -> dict:
    """problem_id -> '조건문, 반복문, 배열/문자열' 콤마 문자열. (전체 17문항 대상,
    compute_concept_scores가 기대하는 것과 동일한 포맷이라 그대로 재사용 가능)"""
    return {int(pid): tag for pid, tag in zip(pool["ProblemID"], pool["개념군_매핑"])}


def assign_difficulty(n_kc: int) -> str:
    if n_kc <= DIFFICULTY_LOW_MAX:
        return "하"
    if n_kc >= DIFFICULTY_HIGH_MIN:
        return "상"
    return "중"


def select_pilot_problems(pool: pd.DataFrame, max_single_concept: int = 6) -> pd.DataFrame:
    """실제 17문항 중 8~12개 파일럿 문항 선정 (결정적·재현 가능한 규칙).

    - 복수 개념군 문항(예: "조건문, 반복문, 배열/문자열")은 전부 포함한다. 이 17문항 풀에는
      반복문·배열/문자열이 단독으로 등장하는 문항이 없고, 이 복수 개념군 문항들만이 두
      개념의 신호를 담고 있기 때문이다.
    - 단일 개념군(조건문 단독) 문항은 ProblemID 오름차순으로 max_single_concept개만 포함한다.
      (결과적으로 이 6개의 n_kc가 3~6 사이에 고르게 분포해 난이도 스펙트럼을 확보하게 된다.)
    """
    is_multi = pool["개념군_매핑"].str.contains(",")
    multi = pool[is_multi]
    single = pool[~is_multi].sort_values("ProblemID").head(max_single_concept)
    selected = pd.concat([multi, single]).sort_values("ProblemID").reset_index(drop=True)
    return selected


def derive_concept_prerequisites(problem_rows: list) -> dict:
    """개념군별 선수개념을 문항 메타 테이블만 보고 도출한다 (재사용 가능한 규칙).

    규칙: 개념군 X가 항상 다른 개념군과 결합된 문항에만 등장하고(X 단독 문항이 없음),
    X를 포함하는 문항들의 평균 KC 개수보다 평균 KC 개수가 더 낮은 '단독' 개념군 Y가 있으면
    Y를 X의 선수개념으로 간주한다. X가 단독 문항으로도 존재하면(이미 기초 단위로 볼 수 있음)
    선수개념은 없음(None)으로 둔다.

    실제 데이터 근거: 이 파일럿 풀에서 반복문·배열/문자열은 단독 문항이 전혀 없고 항상
    조건문과 결합돼 등장하며, 조건문 단독 문항의 평균 KC 개수가 결합 문항 평균보다 낮다.
    -> 조건문이 반복문·배열/문자열의 선수개념이 된다.

    이 함수는 하드코딩된 개념명이 아니라 problem_rows의 실제 분포로부터 계산하므로,
    나중에 문항 풀이 바뀌어(예: 반복문 단독 문항이 추가되어) 전제가 달라져도 자동으로
    선수개념 맵이 갱신된다.
    """
    all_concepts = set()
    for row in problem_rows:
        all_concepts.update(row["concept_groups"])

    pure_avg_kc = {}
    for c in all_concepts:
        pure_rows = [r for r in problem_rows if r["concept_groups"] == [c]]
        if pure_rows:
            pure_avg_kc[c] = sum(r["n_kc"] for r in pure_rows) / len(pure_rows)

    prereq_map = {}
    for c in all_concepts:
        if c in pure_avg_kc:
            prereq_map[c] = None
            continue
        containing = [r for r in problem_rows if c in r["concept_groups"]]
        avg_kc_with_c = sum(r["n_kc"] for r in containing) / len(containing)
        candidates = {k: v for k, v in pure_avg_kc.items() if v < avg_kc_with_c}
        prereq_map[c] = min(candidates, key=candidates.get) if candidates else None

    return prereq_map


def build_problem_meta(path=DATA_PATH) -> dict:
    """실제 17문항 중 파일럿 문항을 선정하고 난이도·선수개념을 재정의한 메타 테이블 생성.
    반환값은 problem_meta.json과 동일한 스키마의 dict."""
    pool = load_problem_pool(path)
    selected = select_pilot_problems(pool)
    excluded_ids = sorted(set(pool["ProblemID"]) - set(selected["ProblemID"]))

    rows = []
    for _, row in selected.iterrows():
        concept_groups = split_concepts(row["개념군_매핑"])
        rows.append(
            {
                "problem_id": int(row["ProblemID"]),
                "assignment_id": int(row["AssignmentID"]),
                "concept_groups": concept_groups,
                "n_kc": int(row["n_kc"]),
                "difficulty": assign_difficulty(row["n_kc"]),
                "requirement": row["Requirement"],
            }
        )

    prereq_map = derive_concept_prerequisites(rows)
    for row in rows:
        prereqs = {prereq_map.get(c) for c in row["concept_groups"] if prereq_map.get(c)}
        row["prerequisite_concept"] = sorted(prereqs)[0] if prereqs else None

    return {
        "problems": rows,
        "concept_prerequisites": prereq_map,
        "difficulty_rule": {
            "basis": "n_kc(Real_50Problems KC 원-핫 컬럼 중 활성화된 개수)",
            "low_max_n_kc": DIFFICULTY_LOW_MAX,
            "high_min_n_kc": DIFFICULTY_HIGH_MIN,
        },
        "selection_note": (
            f"Real_17_TestCased {len(pool)}개 중 복수 개념군 문항 전부 + "
            f"단일 개념군(조건문) 문항 ProblemID 오름차순 상위 {6}개 = {len(rows)}개 선정. "
            f"제외된 ProblemID: {excluded_ids}"
        ),
    }


# ---------------------------------------------------------------------------
# 1. 모듈 ③ — 추천 로직 (제안 시스템)
# ---------------------------------------------------------------------------

# 최근 통과율에 따른 갈래 선택 기준 (근거: 통과율이 매우 낮으면 같은 개념을 더 풀어도
# 실패가 반복될 가능성이 높으므로 선수개념으로 되돌아가 기초를 다지고, 어느 정도
# 통과했으면 같은 개념에서 난이도를 유지/상향해 진도를 이어간다).
PASS_RATE_PREREQ_MAX = 0.3   # 미만이면 선수개념 우선
PASS_RATE_SIMILAR_MIN = 0.6  # 이상이면 유사(중~상) 난이도 우선, 그 사이는 쉬운 문제로 완충

_DIFF_ORDER = {"하": 0, "중": 1, "상": 2}


def _problems_with_concept(problem_meta: list, concept):
    if not concept:
        return []
    return [p for p in problem_meta if concept in p["concept_groups"]]


def _sorted_by_difficulty(candidates, difficulty_filter, exclude_ids):
    pool = [
        p for p in candidates
        if p["problem_id"] not in exclude_ids and p["difficulty"] in difficulty_filter
    ]
    return sorted(pool, key=lambda p: (_DIFF_ORDER[p["difficulty"]], p["n_kc"], p["problem_id"]))


def rank_recommendations(weakest_concept, solved_problem_ids, problem_meta, concept_prerequisites,
                          recent_pass_rate=None):
    """weakest_concept에 대한 추천 후보를 우선순위대로 정렬해 반환한다.
    반환: [(problem_id, recommendation_type, meta_dict), ...]

    갈래 우선순위 (recent_pass_rate 기준):
    - < PASS_RATE_PREREQ_MAX(0.3): 선수개념 문제 -> 같은 개념 쉬운 문제 -> 같은 개념 유사 문제
    - PASS_RATE_PREREQ_MAX ~ PASS_RATE_SIMILAR_MIN(0.6) 미만: 같은 개념 쉬운 문제 -> 유사 문제 -> 선수개념
    - >= PASS_RATE_SIMILAR_MIN, 또는 정보 없음(None): 같은 개념 유사 문제 -> 쉬운 문제 -> 선수개념
    각 갈래 내에서는 이미 푼 문제를 제외하고 난이도·problem_id로 결정적 정렬한다.
    세 갈래가 모두 비면(해당 개념/선수개념 문항이 없거나 이미 다 풂) 파일럿 풀 전체에서
    미해결 문제를 난이도 오름차순으로 채워 넣는다(fallback_any_unsolved) — 에러를 내지 않는다.
    """
    solved = set(solved_problem_ids)
    same_concept = _problems_with_concept(problem_meta, weakest_concept)
    prereq_concept = concept_prerequisites.get(weakest_concept) if weakest_concept else None
    prereq_candidates = _problems_with_concept(problem_meta, prereq_concept)

    easy = _sorted_by_difficulty(same_concept, {"하"}, solved)
    similar = _sorted_by_difficulty(same_concept, {"중", "상"}, solved)
    prereq = _sorted_by_difficulty(prereq_candidates, {"하", "중", "상"}, solved)

    if recent_pass_rate is not None and recent_pass_rate < PASS_RATE_PREREQ_MAX:
        buckets = [(prereq, "prerequisite_concept"), (easy, "same_concept_easy"), (similar, "same_concept_similar")]
    elif recent_pass_rate is not None and recent_pass_rate < PASS_RATE_SIMILAR_MIN:
        buckets = [(easy, "same_concept_easy"), (similar, "same_concept_similar"), (prereq, "prerequisite_concept")]
    else:
        buckets = [(similar, "same_concept_similar"), (easy, "same_concept_easy"), (prereq, "prerequisite_concept")]

    ranked, seen = [], set()
    for bucket, rec_type in buckets:
        for p in bucket:
            if p["problem_id"] in seen:
                continue
            ranked.append((p["problem_id"], rec_type, p))
            seen.add(p["problem_id"])

    if len(ranked) < 3:
        rest = sorted(
            [p for p in problem_meta if p["problem_id"] not in solved and p["problem_id"] not in seen],
            key=lambda p: (_DIFF_ORDER.get(p["difficulty"], 1), p["problem_id"]),
        )
        for p in rest:
            ranked.append((p["problem_id"], "fallback_any_unsolved", p))
            seen.add(p["problem_id"])

    return ranked


def _build_reason(rec_type, weakest_concept, prereq_concept, recent_pass_rate, chosen):
    pct = f"{recent_pass_rate:.0%}" if recent_pass_rate is not None else "정보 없음"
    concept_label = "/".join(chosen["concept_groups"])
    if rec_type == "prerequisite_concept":
        return f"{weakest_concept} 최근 통과율 {pct}로 매우 낮아 선수개념({prereq_concept}) 문항으로 되돌아감"
    if rec_type == "same_concept_easy":
        return f"동일 개념({concept_label})의 쉬운 문항 추천 — 최근 통과율 {pct}로 완충 필요"
    if rec_type == "same_concept_similar":
        return f"동일 개념({concept_label})의 유사 난이도 문항 추천 — 최근 통과율 {pct}로 양호, 난이도 유지"
    return "동일/선수 개념 후보가 모두 소진되어 파일럿 풀 내 미해결 문제로 대체 추천"


def recommend_problem(weakest_concept, solved_problem_ids, problem_meta, concept_prerequisites,
                       recent_pass_rate=None):
    """모듈②의 weakest_concept을 입력받아 다음에 풀 문제 1개를 추천한다.

    problem_meta: build_problem_meta()["problems"] (list[dict])
    concept_prerequisites: build_problem_meta()["concept_prerequisites"] (dict)
    recent_pass_rate: weakest_concept 관련 문항의 최근 통과율(0~1). 보통 모듈②
        concept_scores[weakest_concept](=취약도=1-통과율)를 역산해서 넘긴다.

    weakest_concept이 None이거나 problem_meta에 해당 개념 문항이 없어도 에러 없이
    파일럿 풀 내 미해결 문제로 대체 추천한다. 추천할 문제가 전혀 없으면(모든 문제를 이미
    풀었으면) None을 반환한다.
    """
    ranked = rank_recommendations(weakest_concept, solved_problem_ids, problem_meta,
                                  concept_prerequisites, recent_pass_rate)
    if not ranked:
        return None

    problem_id, rec_type, chosen = ranked[0]
    prereq_concept = concept_prerequisites.get(weakest_concept) if weakest_concept else None
    reason = _build_reason(rec_type, weakest_concept, prereq_concept, recent_pass_rate, chosen)
    return {"recommended_problem_id": problem_id, "recommendation_type": rec_type, "reason": reason}


def build_module3_output(module1_records, module2_records, problem_meta, concept_prerequisites):
    """module2_output.json의 각 학생 레코드에 대해 추천 결과를 생성."""
    solved_by_subject = {}
    for r in module1_records:
        solved_by_subject.setdefault(r["subject_id"], set()).add(r["problem_id"])

    results = []
    for rec in module2_records:
        sid = rec["subject_id"]
        weakest = rec["weakest_concept"]
        solved = solved_by_subject.get(sid, set())
        recent_pass_rate = None
        if weakest is not None and rec["concept_scores"].get(weakest) is not None:
            recent_pass_rate = 1 - rec["concept_scores"][weakest]

        rec_result = recommend_problem(weakest, solved, problem_meta, concept_prerequisites, recent_pass_rate)
        if rec_result is None:
            results.append(
                {
                    "subject_id": sid,
                    "recommended_problem_id": None,
                    "recommendation_type": "no_candidate",
                    "reason": "파일럿 문항 풀을 모두 해결하여 추천할 문제가 없음",
                }
            )
        else:
            results.append({"subject_id": sid, **rec_result})
    return results


# ---------------------------------------------------------------------------
# 2. 베이스라인 3종 (동일 인터페이스: recommended_problem_id / recommendation_type / reason)
# ---------------------------------------------------------------------------

def _rank_ids_random(problem_meta, solved, rng):
    candidates = [p["problem_id"] for p in problem_meta if p["problem_id"] not in solved]
    rng.shuffle(candidates)
    return candidates


def _rank_ids_by_difficulty(problem_meta, solved):
    candidates = sorted(
        [p for p in problem_meta if p["problem_id"] not in solved],
        key=lambda p: (_DIFF_ORDER.get(p["difficulty"], 1), p["problem_id"]),
    )
    return [p["problem_id"] for p in candidates]


def _rank_ids_by_pass_rate(problem_meta, solved, global_pass_rates, student_avg_pass_rate):
    candidates = [
        p for p in problem_meta if p["problem_id"] not in solved and p["problem_id"] in global_pass_rates
    ]
    candidates = sorted(
        candidates,
        key=lambda p: (abs(global_pass_rates[p["problem_id"]] - student_avg_pass_rate), p["problem_id"]),
    )
    return [p["problem_id"] for p in candidates]


def recommend_random(weakest_concept, solved_problem_ids, problem_meta, rng=None):
    """베이스라인 A: 개념/난이도를 전혀 고려하지 않고 미해결 문제 중 무작위 1개."""
    rng = rng if rng is not None else np.random.default_rng()
    ranked = _rank_ids_random(problem_meta, set(solved_problem_ids), rng)
    if not ranked:
        return None
    return {"recommended_problem_id": ranked[0], "recommendation_type": "random", "reason": "무작위 베이스라인"}


def recommend_by_difficulty(weakest_concept, solved_problem_ids, problem_meta):
    """베이스라인 B: 개념 구분 없이 파일럿 풀 전체에서 가장 쉬운 미해결 문제."""
    ranked = _rank_ids_by_difficulty(problem_meta, set(solved_problem_ids))
    if not ranked:
        return None
    chosen = next(p for p in problem_meta if p["problem_id"] == ranked[0])
    return {
        "recommended_problem_id": ranked[0],
        "recommendation_type": "by_difficulty",
        "reason": f"난이도 기준 베이스라인 — 파일럿 풀 전체 최저 난이도({chosen['difficulty']}) 미해결 문항",
    }


def recommend_by_pass_rate(weakest_concept, solved_problem_ids, problem_meta, global_pass_rates,
                            student_avg_pass_rate):
    """베이스라인 C: 개념 구분 없이, 학생의 평균 통과율과 전역(전체 학생) 문항별 평균
    통과율이 가장 가까운 미해결 문제를 추천 (KC 기반 난이도가 아닌 실측 정답률 기반)."""
    ranked = _rank_ids_by_pass_rate(problem_meta, set(solved_problem_ids), global_pass_rates, student_avg_pass_rate)
    if not ranked:
        return None
    chosen_rate = global_pass_rates[ranked[0]]
    return {
        "recommended_problem_id": ranked[0],
        "recommendation_type": "by_pass_rate",
        "reason": (
            f"정답률 기준 베이스라인 — 학생 평균 통과율({student_avg_pass_rate:.0%})과 "
            f"가장 가까운 전역 통과율({chosen_rate:.0%}) 문항"
        ),
    }


# ---------------------------------------------------------------------------
# 3. 평가 — hold-out 방식 Recall@1 / Recall@3 / NDCG@3 (목표: 제안 시스템 Recall@3 0.70 이상)
# ---------------------------------------------------------------------------

def recall_at_k(ranked_ids, true_id, k):
    return 1.0 if true_id in ranked_ids[:k] else 0.0


def ndcg_at_k(ranked_ids, true_id, k):
    if true_id in ranked_ids[:k]:
        rank = ranked_ids.index(true_id) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0


def compute_global_pass_rates(module1_records):
    """문항별 전체 학생 평균 통과율 (by_pass_rate 베이스라인용).
    오프라인 평가 단순화를 위해 전체 로그의 집계 통계를 고정값으로 사용한다(README 한계점 참고)."""
    df = pd.DataFrame(module1_records)
    return {int(k): float(v) for k, v in df.groupby("problem_id")["test_pass_rate"].mean().items()}


def generate_eval_points(module1_records, min_history=2):
    """학생별 제출 이력을 timestep 순으로 정렬해, 매 시점마다 '지금까지의 이력으로 다음에
    실제로 푼 문제를 맞히는지' 평가할 수 있는 (히스토리, 정답 레이블) 포인트를 생성한다.

    학생이 6명뿐이라 마지막 제출 1건만 라벨로 쓰면 평가 포인트가 6개뿐이라 Recall@k가
    거의 의미가 없다. 그래서 최소 이력(min_history) 이상만 확보되면 매 시점을 하나의
    평가 포인트로 사용하는 슬라이딩 hold-out 방식을 쓴다 (지식추적/추천 시스템 오프라인
    평가에서 흔히 쓰는 방식 — "각 시점까지의 이력으로 다음 아이템을 예측").
    """
    df = pd.DataFrame(module1_records)
    points = []
    for sid, sub_df in df.groupby("subject_id"):
        sub_df = sub_df.sort_values("timestep").reset_index(drop=True)
        for cut in range(min_history, len(sub_df)):
            points.append(
                {
                    "subject_id": sid,
                    "history": sub_df.iloc[:cut].copy(),
                    "true_problem_id": int(sub_df.iloc[cut]["problem_id"]),
                }
            )
    return points


def evaluate_recommenders(module1_records, problem_meta_bundle, full_concept_map):
    """제안 시스템과 3개 베이스라인을 동일한 hold-out 평가셋에서 비교.

    full_concept_map: problem_id -> '조건문, 반복문, ...' (파일럿 12개뿐 아니라 실제
    17문항 전체). 학생 히스토리에는 파일럿에서 제외된 문항(조건문 단독 5개)의 제출도
    섞여 있으므로, concept_scores/weakest_concept 계산에는 이 전체 맵을 써야
    개념 신호를 놓치지 않는다 — 후보 풀(problem_meta, 추천 대상)만 파일럿으로 좁힌다.
    """
    problem_meta = problem_meta_bundle["problems"]
    concept_prerequisites = problem_meta_bundle["concept_prerequisites"]
    pilot_ids = {p["problem_id"] for p in problem_meta}

    global_pass_rates = compute_global_pass_rates(module1_records)
    points = generate_eval_points(module1_records)

    methods = ["proposed", "random", "by_difficulty", "by_pass_rate"]
    scores = {m: {"recall@1": [], "recall@3": [], "ndcg@3": []} for m in methods}
    n_skipped = 0

    for point in points:
        if point["true_problem_id"] not in pilot_ids:
            n_skipped += 1
            continue

        history = point["history"]
        history = history[history["problem_id"].isin(set(full_concept_map))]
        if history.empty:
            n_skipped += 1
            continue
        history = history.assign(
            개념군_매핑=history["problem_id"].map(full_concept_map),
            Score=history["test_pass_rate"],
        )

        concept_scores = compute_concept_scores(history)
        valid_scores = {k: v for k, v in concept_scores.items() if v is not None}
        weakest = max(valid_scores, key=valid_scores.get) if valid_scores else None
        recent_pass_rate = 1 - valid_scores[weakest] if weakest is not None else None

        solved = set(point["history"]["problem_id"])
        student_avg = float(point["history"]["test_pass_rate"].mean())
        # 파이썬 내장 hash()는 실행마다 값이 달라져(PYTHONHASHSEED) 재현이 안 되므로,
        # 결정적인 hashlib 기반 시드를 사용한다 (random 베이스라인도 매 실행 동일 결과가 나와야 함).
        seed_key = f"{point['subject_id']}_{len(point['history'])}".encode("utf-8")
        seed = int(hashlib.sha256(seed_key).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        ranked_by_method = {
            "proposed": [pid for pid, _, _ in rank_recommendations(
                weakest, solved, problem_meta, concept_prerequisites, recent_pass_rate)],
            "random": _rank_ids_random(problem_meta, solved, rng),
            "by_difficulty": _rank_ids_by_difficulty(problem_meta, solved),
            "by_pass_rate": _rank_ids_by_pass_rate(problem_meta, solved, global_pass_rates, student_avg),
        }

        for m in methods:
            ranked = ranked_by_method[m]
            scores[m]["recall@1"].append(recall_at_k(ranked, point["true_problem_id"], 1))
            scores[m]["recall@3"].append(recall_at_k(ranked, point["true_problem_id"], 3))
            scores[m]["ndcg@3"].append(ndcg_at_k(ranked, point["true_problem_id"], 3))

    n_eval = len(points) - n_skipped
    summary = {
        "n_total_transition_points": len(points),
        "n_skipped_true_label_outside_pilot_pool": n_skipped,
        "n_eval_points": n_eval,
        "methods": {},
    }
    for m in methods:
        if n_eval == 0:
            summary["methods"][m] = {"recall@1": None, "recall@3": None, "ndcg@3": None}
            continue
        summary["methods"][m] = {
            "recall@1": round(float(np.mean(scores[m]["recall@1"])), 4),
            "recall@3": round(float(np.mean(scores[m]["recall@3"])), 4),
            "ndcg@3": round(float(np.mean(scores[m]["ndcg@3"])), 4),
        }

    proposed_recall3 = summary["methods"]["proposed"]["recall@3"]
    summary["target_recall@3"] = 0.70
    summary["target_achieved"] = bool(proposed_recall3 is not None and proposed_recall3 >= 0.70)
    return summary


# ---------------------------------------------------------------------------
# 4. 실행
# ---------------------------------------------------------------------------

def main():
    print("[1/5] 문항 메타데이터 테이블 구축 중...")
    problem_meta_bundle = build_problem_meta()
    with open(PROBLEM_META_PATH, "w", encoding="utf-8") as f:
        json.dump(problem_meta_bundle, f, ensure_ascii=False, indent=2)
    print(f"  완료: 파일럿 문항 {len(problem_meta_bundle['problems'])}개, "
          f"선수개념 맵 {problem_meta_bundle['concept_prerequisites']}")

    print("[2/5] 모듈①②출력 로드 중...")
    with open(MODULE1_OUTPUT_PATH, encoding="utf-8") as f:
        module1_records = json.load(f)
    with open(MODULE2_OUTPUT_PATH, encoding="utf-8") as f:
        module2_records = json.load(f)
    print(f"  제출 이력 {len(module1_records)}건, 학생 {len(module2_records)}명")

    print("[3/5] 모듈 ③ 추천 실행 중...")
    module3_outputs = build_module3_output(
        module1_records, module2_records,
        problem_meta_bundle["problems"], problem_meta_bundle["concept_prerequisites"],
    )
    with open(MODULE3_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(module3_outputs, f, ensure_ascii=False, indent=2)
    print(f"  완료: 학생 {len(module3_outputs)}명 추천 생성")

    print("[4/5] hold-out 평가 실행 중 (제안 시스템 vs 베이스라인 3종)...")
    full_concept_map = build_problem_concept_map(load_problem_pool())
    evaluation = evaluate_recommenders(module1_records, problem_meta_bundle, full_concept_map)
    with open(MODULE3_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    print(f"  평가 포인트 {evaluation['n_eval_points']}개")
    for m, s in evaluation["methods"].items():
        print(f"  {m}: Recall@1={s['recall@1']} Recall@3={s['recall@3']} NDCG@3={s['ndcg@3']}")
    print(f"  목표(Recall@3>=0.70) 달성 여부: {evaluation['target_achieved']}")

    print("[5/5] 완료. data/problem_meta.json / results/module3_output.json / "
          "results/module3_evaluation.json 생성됨")


if __name__ == "__main__":
    main()
