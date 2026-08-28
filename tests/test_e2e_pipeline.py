# -*- coding: utf-8 -*-
"""모듈 ①→②→③→④ 통합 파이프라인(run_pipeline) 테스트.

모듈④(Claude API)는 목(mock) client로 대체해 실제 네트워크를 타지 않는다. 모듈①은 실제
javac로 컴파일하므로 이 테스트는 javac가 설치돼 있어야 한다(저장소 전체가 이미 이 전제를
깔고 있음 — results/README.md 참고).
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from e2e_pipeline import run_pipeline


def make_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


PROBLEM_META = {
    "problems": [
        {"problem_id": 1, "concept_groups": ["조건문"], "n_kc": 4, "difficulty": "하",
         "requirement": "문제1", "prerequisite_concept": None},
        {"problem_id": 2, "concept_groups": ["조건문"], "n_kc": 5, "difficulty": "중",
         "requirement": "문제2", "prerequisite_concept": None},
        {"problem_id": 3, "concept_groups": ["조건문"], "n_kc": 6, "difficulty": "상",
         "requirement": "문제3", "prerequisite_concept": None},
    ],
    "concept_prerequisites": {"조건문": None},
}

CONCEPT_CARDS = {
    "조건문": {
        "concept": "조건문",
        "description": "if/else로 분기하는 개념군",
        "common_mistakes": ["경계값 혼동"],
        "pre_check_items": ["경계값을 표로 정리해보기"],
    }
}

# 컴파일이 성공하는 최소한의 Java 메서드 (check_compile이 class Solution { ... }으로 감쌈)
VALID_JAVA_CODE = "public boolean isPositive(int n) { return n > 0; }"


def test_run_pipeline_happy_path_returns_all_four_modules():
    """정상 케이스: 이력이 있어 취약도가 계산되고, 4개 모듈을 모두 통과해 실제 추천+피드백이 나온다."""
    client = MagicMock()
    client.messages.create.side_effect = [
        make_response("조건문을 다시 확인해보세요. 2번 문제를 풀어보세요."),
        make_response("판정: 아니오\n이유: 개념만 안내함"),
    ]

    submission_history = [
        {"subject_id": "s1", "problem_id": 1, "timestep": 0, "compile_ok": True,
         "test_pass_rate": 0.5, "n_failed": 1, "n_total": 2, "error_pattern": "partial_failure_logic"},
    ]
    new_submission = {
        "Code": VALID_JAVA_CODE,
        "binary_score": [1, 0],
        "ProblemID": 1,
        "timestep": 1,
    }

    result = run_pipeline("s1", new_submission, submission_history, PROBLEM_META, CONCEPT_CARDS, client=client)

    assert result["status"] == "ok"
    assert result["module1"]["compile_ok"] is True
    assert result["module1"]["problem_id"] == 1
    assert "조건문" in result["module2"]["concept_scores"]
    assert result["module2"]["weakest_concept"] == "조건문"
    assert result["module3"]["recommended_problem_id"] in (2, 3)
    assert result["module3"]["recommended_problem_id"] != 1  # 이미 푼 문제는 재추천 안 함
    assert result["module4"]["feedback_text"] == "조건문을 다시 확인해보세요. 2번 문제를 풀어보세요."
    assert result["module4"]["direct_answer_leak"] is False


def test_run_pipeline_records_error_without_crashing_when_module1_fails():
    """모듈①이 실패해도(필수 키 누락) 예외를 올리지 않고 에러를 기록하며, 이후 단계는 skip 처리된다."""
    client = MagicMock()
    broken_submission = {"Code": VALID_JAVA_CODE, "binary_score": [1]}  # ProblemID 누락 -> KeyError 유발

    result = run_pipeline("s2", broken_submission, [], PROBLEM_META, CONCEPT_CARDS, client=client)

    assert result["status"] == "partial_failure"
    assert "error" in result["module1"]
    assert result["module2"]["skipped"] is True
    assert result["module3"]["skipped"] is True
    assert result["module4"]["skipped"] is True
    client.messages.create.assert_not_called()


def test_run_pipeline_no_candidate_skips_llm_call():
    """모든 문제를 이미 푼 경우 모듈③이 no_candidate를 반환하고, 모듈④는 LLM 없이 정형 문구를 쓴다."""
    client = MagicMock()
    submission_history = [
        {"subject_id": "s3", "problem_id": pid, "timestep": i, "compile_ok": True,
         "test_pass_rate": 1.0, "n_failed": 0, "n_total": 1, "error_pattern": "all_passed"}
        for i, pid in enumerate([1, 2])
    ]
    new_submission = {"Code": VALID_JAVA_CODE, "binary_score": [1], "ProblemID": 3, "timestep": 2}

    result = run_pipeline("s3", new_submission, submission_history, PROBLEM_META, CONCEPT_CARDS, client=client)

    assert result["status"] == "ok"
    assert result["module3"]["recommendation_type"] == "no_candidate"
    assert result["module4"]["direct_answer_leak"] is False
    client.messages.create.assert_not_called()


def test_run_pipeline_accepts_precomputed_module1_record_without_recompiling():
    """원본 Code 없이(module1_output.json 스키마) 넘겨도 재컴파일 없이 동작한다 — 배치 재생 모드."""
    client = MagicMock()
    client.messages.create.side_effect = [
        make_response("피드백"),
        make_response("판정: 아니오\n이유: 안전함"),
    ]
    precomputed_submission = {
        "subject_id": "s4", "problem_id": 1, "timestep": 0, "compile_ok": True,
        "test_pass_rate": 0.5, "n_failed": 1, "n_total": 2, "error_pattern": "partial_failure_logic",
    }

    result = run_pipeline("s4", precomputed_submission, [], PROBLEM_META, CONCEPT_CARDS, client=client)

    assert result["status"] == "ok"
    assert result["module1"]["compile_ok"] is True
    assert result["module1"]["test_pass_rate"] == 0.5
