# -*- coding: utf-8 -*-
"""모듈 ④ 피드백 생성/평가 로직 단위 테스트.

실제 Claude API는 절대 호출하지 않는다 — call_claude()의 client 인자에 목(mock) 객체를
넘겨서, API 키 없이도 프롬프트 구성·분기 로직·규칙 기반 필터를 검증한다.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "feedback"))

from feedback import (
    NO_CANDIDATE_FEEDBACK,
    build_feedback_prompt,
    build_module4_output,
    evaluate_feedback,
    evaluate_module4,
    generate_baseline_feedback,
    generate_feedback,
    generate_problem_level_feedback,
    judge_direct_answer_leak,
    rule_based_leak_check,
)


def make_response(text: str):
    """client.messages.create()가 반환하는 Message 객체를 흉내낸다."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


CONCEPT_CARDS = {
    "조건문": {
        "concept": "조건문",
        "description": "if/else로 분기하는 개념군",
        "common_mistakes": ["경계값 혼동", "else if 순서 실수"],
        "pre_check_items": ["경계값을 표로 정리해보기"],
    },
    "배열/문자열": {
        "concept": "배열/문자열",
        "description": "문자열 인덱스를 다루는 개념군",
        "common_mistakes": ["substring end 인덱스 혼동"],
        "pre_check_items": ["시작/끝 인덱스를 손으로 적어보기"],
    },
}

PROBLEM_META_BY_ID = {
    5: {"problem_id": 5, "concept_groups": ["조건문"], "requirement": "전화를 받을지 판단하라"},
    34: {"problem_id": 34, "concept_groups": ["조건문", "반복문", "배열/문자열"], "requirement": "zipZap 문자열 처리"},
}


# ---------------------------------------------------------------------------
# 규칙 기반 leak 필터
# ---------------------------------------------------------------------------

def test_rule_based_leak_check_detects_code_fence():
    text = "이렇게 하면 돼요:\n```java\nreturn true;\n```"
    assert rule_based_leak_check(text) is True


def test_rule_based_leak_check_detects_if_block():
    text = "정답은 if (n > 0) { return true; } 이렇게 작성하면 됩니다."
    assert rule_based_leak_check(text) is True


def test_rule_based_leak_check_passes_clean_text():
    text = "조건문 개념에서 경계값 처리를 다시 확인해보세요. 다음은 5번 문제를 풀어볼 차례예요."
    assert rule_based_leak_check(text) is False


# ---------------------------------------------------------------------------
# 프롬프트 구성
# ---------------------------------------------------------------------------

def test_build_feedback_prompt_includes_key_fields():
    prompt = build_feedback_prompt(
        "조건문", {"조건문": 0.7}, 5, "동일 개념의 쉬운 문항", CONCEPT_CARDS["조건문"]
    )
    assert "조건문" in prompt
    assert "70%" in prompt
    assert "5" in prompt
    assert "동일 개념의 쉬운 문항" in prompt
    assert "경계값 혼동" in prompt


def test_build_feedback_prompt_handles_missing_concept_card_without_error():
    prompt = build_feedback_prompt("반복문", {}, 40, "이유", concept_card=None)
    assert "정보 없음" in prompt


# ---------------------------------------------------------------------------
# 피드백 생성 — call_claude를 목 client로 대체
# ---------------------------------------------------------------------------

def test_generate_feedback_uses_system_prompt_and_returns_text():
    client = MagicMock()
    client.messages.create.return_value = make_response("좋은 피드백입니다.")

    result = generate_feedback("조건문", {"조건문": 0.7}, 5, "이유", CONCEPT_CARDS["조건문"], client=client)

    assert result == "좋은 피드백입니다."
    _, kwargs = client.messages.create.call_args
    assert "정답 코드나 완성된 로직" in kwargs["system"]


def test_generate_baseline_feedback_uses_baseline_system_prompt():
    client = MagicMock()
    client.messages.create.return_value = make_response("일반 피드백입니다.")

    result = generate_baseline_feedback("문제 설명", client=client)

    assert result == "일반 피드백입니다."
    _, kwargs = client.messages.create.call_args
    assert "취약" not in kwargs["system"]  # 베이스라인은 취약 개념 맥락을 언급하지 않음


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

def test_judge_direct_answer_leak_parses_yes():
    client = MagicMock()
    client.messages.create.return_value = make_response("판정: 예\n이유: 완성된 코드가 그대로 제시됨")

    leak, reasoning = judge_direct_answer_leak("피드백", "문제", client=client)

    assert leak is True
    assert reasoning == "완성된 코드가 그대로 제시됨"


def test_judge_direct_answer_leak_parses_no():
    client = MagicMock()
    client.messages.create.return_value = make_response("판정: 아니오\n이유: 개념만 안내함")

    leak, reasoning = judge_direct_answer_leak("피드백", "문제", client=client)

    assert leak is False
    assert reasoning == "개념만 안내함"


# ---------------------------------------------------------------------------
# evaluate_feedback — 규칙 기반이 걸리면 judge 호출을 생략(비용 절약)
# ---------------------------------------------------------------------------

def test_evaluate_feedback_skips_judge_when_rule_based_leak_detected():
    client = MagicMock()
    leaking_text = "정답은 ```return true;``` 입니다."

    result = evaluate_feedback(leaking_text, "문제", client=client)

    assert result["direct_answer_leak"] is True
    client.messages.create.assert_not_called()


def test_evaluate_feedback_calls_judge_when_no_rule_based_leak():
    client = MagicMock()
    client.messages.create.return_value = make_response("판정: 아니오\n이유: 안전함")
    clean_text = "조건문 개념을 다시 확인해보세요."

    result = evaluate_feedback(clean_text, "문제", client=client)

    assert result["direct_answer_leak"] is False
    client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# build_module4_output — 학생별 출력
# ---------------------------------------------------------------------------

def test_build_module4_output_handles_no_candidate_without_calling_llm():
    """이미 파일럿 문항을 다 푼 학생(recommended_problem_id=None)은 LLM을 호출하지 않는다."""
    client = MagicMock()
    module2_records = [{"subject_id": "s1", "weakest_concept": "조건문", "concept_scores": {"조건문": 0.7}}]
    module3_records = [{"subject_id": "s1", "recommended_problem_id": None, "recommendation_type": "no_candidate",
                         "reason": "없음"}]

    results = build_module4_output(module2_records, module3_records, CONCEPT_CARDS, PROBLEM_META_BY_ID, client=client)

    assert results[0]["feedback_text"] == NO_CANDIDATE_FEEDBACK
    assert results[0]["direct_answer_leak"] is False
    client.messages.create.assert_not_called()


def test_build_module4_output_generates_feedback_for_valid_recommendation():
    client = MagicMock()
    client.messages.create.side_effect = [
        make_response("조건문을 다시 확인해보세요. 5번 문제를 풀어보세요."),
        make_response("판정: 아니오\n이유: 개념만 안내함"),
    ]
    module2_records = [{"subject_id": "s1", "weakest_concept": "조건문", "concept_scores": {"조건문": 0.7}}]
    module3_records = [{"subject_id": "s1", "recommended_problem_id": 5, "recommendation_type": "same_concept_easy",
                         "reason": "쉬운 문항"}]

    results = build_module4_output(module2_records, module3_records, CONCEPT_CARDS, PROBLEM_META_BY_ID, client=client)

    assert results[0]["feedback_text"] == "조건문을 다시 확인해보세요. 5번 문제를 풀어보세요."
    assert results[0]["direct_answer_leak"] is False
    assert client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# 문항 단위 평가 데이터셋 + 최종 평가
# ---------------------------------------------------------------------------

def test_generate_problem_level_feedback_uses_first_concept_group():
    client = MagicMock()
    client.messages.create.return_value = make_response("피드백")
    problem_meta_list = [PROBLEM_META_BY_ID[34]]

    records = generate_problem_level_feedback(problem_meta_list, CONCEPT_CARDS, client=client)

    assert records[0]["problem_id"] == 34
    assert records[0]["proposed_feedback"] == "피드백"
    assert records[0]["baseline_feedback"] == "피드백"


def test_evaluate_module4_computes_leak_rate_and_target():
    client = MagicMock()
    client.messages.create.return_value = make_response("판정: 아니오\n이유: 안전함")
    records = [
        {"problem_id": 5, "proposed_feedback": "조건문을 다시 확인해보세요.", "baseline_feedback": "일반 피드백"},
        {"problem_id": 34, "proposed_feedback": "정답은 ```return true;``` 입니다.", "baseline_feedback": "일반 피드백2"},
    ]

    evaluation = evaluate_module4(records, PROBLEM_META_BY_ID, client=client)

    assert evaluation["n_problems"] == 2
    assert evaluation["proposed"]["n_leaked"] == 1
    assert evaluation["proposed"]["direct_answer_leak_rate"] == 0.5
    assert evaluation["target_leak_rate"] == 0.10
    assert evaluation["target_achieved"] is False
