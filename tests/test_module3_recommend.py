# -*- coding: utf-8 -*-
"""모듈 ③ 추천 로직 단위 테스트."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from module3_recommend import (
    derive_concept_prerequisites,
    ndcg_at_k,
    recall_at_k,
    recommend_by_difficulty,
    recommend_by_pass_rate,
    recommend_problem,
    recommend_random,
)


def make_problem(problem_id, concept_groups, n_kc, difficulty):
    return {
        "problem_id": problem_id,
        "concept_groups": concept_groups,
        "n_kc": n_kc,
        "difficulty": difficulty,
        "requirement": f"문제 {problem_id}",
        "prerequisite_concept": None,
    }


PROBLEM_META = [
    make_problem(1, ["조건문"], 3, "하"),
    make_problem(2, ["조건문"], 4, "하"),
    make_problem(3, ["조건문"], 5, "중"),
    make_problem(4, ["조건문", "반복문", "배열/문자열"], 6, "중"),
    make_problem(5, ["조건문", "반복문", "배열/문자열"], 8, "상"),
]
CONCEPT_PREREQ = {"조건문": None, "반복문": "조건문", "배열/문자열": "조건문"}


# ---------------------------------------------------------------------------
# recommend_problem
# ---------------------------------------------------------------------------

def test_recommend_problem_never_recommends_already_solved():
    solved = {1, 2, 3}
    result = recommend_problem("조건문", solved, PROBLEM_META, CONCEPT_PREREQ, recent_pass_rate=0.8)
    assert result is not None
    assert result["recommended_problem_id"] not in solved


def test_recommend_problem_returns_none_when_everything_solved():
    solved = {p["problem_id"] for p in PROBLEM_META}
    result = recommend_problem("조건문", solved, PROBLEM_META, CONCEPT_PREREQ, recent_pass_rate=0.8)
    assert result is None


def test_recommend_problem_handles_unknown_concept_without_error():
    """problem_meta에 아예 없는 개념이 weakest_concept으로 들어와도 에러 없이 처리되어야 한다."""
    result = recommend_problem("배열/문자열/존재하지않는하위개념", set(), PROBLEM_META, CONCEPT_PREREQ)
    assert result is not None
    assert result["recommendation_type"] == "fallback_any_unsolved"
    assert result["recommended_problem_id"] in {p["problem_id"] for p in PROBLEM_META}


def test_recommend_problem_handles_none_concept_without_error():
    """weakest_concept이 None(취약 개념 미확정)이어도 에러 없이 대체 추천되어야 한다."""
    result = recommend_problem(None, set(), PROBLEM_META, CONCEPT_PREREQ)
    assert result is not None
    assert result["recommendation_type"] == "fallback_any_unsolved"


def test_recommend_problem_low_pass_rate_prefers_prerequisite():
    """최근 통과율이 매우 낮으면(반복문 취약) 선수개념(조건문) 문제로 되돌아가야 한다."""
    result = recommend_problem("반복문", set(), PROBLEM_META, CONCEPT_PREREQ, recent_pass_rate=0.1)
    assert result["recommendation_type"] == "prerequisite_concept"
    chosen = next(p for p in PROBLEM_META if p["problem_id"] == result["recommended_problem_id"])
    assert chosen["concept_groups"] == ["조건문"]


def test_recommend_problem_high_pass_rate_prefers_similar_difficulty():
    """통과율이 양호하면 같은 개념의 유사(중/상) 난이도 문제를 우선 추천해야 한다."""
    result = recommend_problem("조건문", {1, 2}, PROBLEM_META, CONCEPT_PREREQ, recent_pass_rate=0.9)
    assert result["recommendation_type"] == "same_concept_similar"


def test_recommend_problem_missing_easy_falls_back_to_similar():
    """쉬운 문제를 모두 풀었으면(같은 개념 쉬운 문제 없음) 유사 난이도로 대체돼야 한다."""
    solved = {1, 2}  # 조건문 '하' 난이도 전부 해결
    result = recommend_problem("조건문", solved, PROBLEM_META, CONCEPT_PREREQ, recent_pass_rate=0.1)
    # 선수개념(조건문 자신)은 순환이라 없음(None) -> 같은 개념 쉬운 문제(고갈) -> 유사 문제로 대체
    assert result["recommended_problem_id"] not in solved


# ---------------------------------------------------------------------------
# 베이스라인 3종
# ---------------------------------------------------------------------------

def test_recommend_random_never_recommends_already_solved():
    solved = {1, 2, 3, 4}
    rng = np.random.default_rng(0)
    result = recommend_random("조건문", solved, PROBLEM_META, rng=rng)
    assert result["recommended_problem_id"] == 5


def test_recommend_random_returns_none_when_everything_solved():
    solved = {p["problem_id"] for p in PROBLEM_META}
    result = recommend_random("조건문", solved, PROBLEM_META, rng=np.random.default_rng(0))
    assert result is None


def test_recommend_by_difficulty_picks_easiest_unsolved_ignoring_concept():
    result = recommend_by_difficulty("배열/문자열", {1}, PROBLEM_META)
    assert result["recommended_problem_id"] == 2  # 다음으로 쉬운(하) 미해결 문항
    assert result["recommended_problem_id"] != 1


def test_recommend_by_pass_rate_never_recommends_already_solved():
    solved = {1}
    global_pass_rates = {1: 0.9, 2: 0.8, 3: 0.5, 4: 0.4, 5: 0.1}
    result = recommend_by_pass_rate("조건문", solved, PROBLEM_META, global_pass_rates, student_avg_pass_rate=0.85)
    assert result["recommended_problem_id"] not in solved
    assert result["recommended_problem_id"] == 2  # 0.8이 학생 평균(0.85)과 가장 가까움


# ---------------------------------------------------------------------------
# 문항 메타 생성 로직 — 선수개념 도출 규칙
# ---------------------------------------------------------------------------

def test_derive_concept_prerequisites_from_synthetic_rows():
    rows = [
        {"concept_groups": ["조건문"], "n_kc": 3},
        {"concept_groups": ["조건문"], "n_kc": 5},
        {"concept_groups": ["조건문", "반복문"], "n_kc": 7},
    ]
    prereq_map = derive_concept_prerequisites(rows)
    assert prereq_map["조건문"] is None  # 단독 문항이 있으므로 선수개념 없음(기초 개념)
    assert prereq_map["반복문"] == "조건문"  # 단독 문항이 없고 KC 평균이 더 높음


def test_derive_concept_prerequisites_no_prerequisite_when_all_standalone():
    rows = [
        {"concept_groups": ["조건문"], "n_kc": 3},
        {"concept_groups": ["반복문"], "n_kc": 4},
    ]
    prereq_map = derive_concept_prerequisites(rows)
    assert prereq_map["조건문"] is None
    assert prereq_map["반복문"] is None


# ---------------------------------------------------------------------------
# 평가 지표
# ---------------------------------------------------------------------------

def test_recall_at_k():
    assert recall_at_k([5, 2, 9], 2, 3) == 1.0
    assert recall_at_k([5, 2, 9], 2, 1) == 0.0
    assert recall_at_k([5, 2, 9], 7, 3) == 0.0


def test_ndcg_at_k_rewards_higher_rank():
    ndcg_rank1 = ndcg_at_k([2, 5, 9], 2, 3)
    ndcg_rank3 = ndcg_at_k([5, 9, 2], 2, 3)
    assert ndcg_rank1 == 1.0
    assert 0 < ndcg_rank3 < ndcg_rank1
    assert ndcg_at_k([5, 9, 1], 2, 3) == 0.0
