# -*- coding: utf-8 -*-
"""
모듈 ④ 개념카드 · LLM 피드백
입력: data/concept_cards.json (개념군별 개념카드), data/problem_meta.json (모듈③ 문항 메타),
      results/module2_output.json (weakest_concept, concept_scores),
      results/module3_output.json (recommended_problem_id, reason)
출력: results/module4_output.json (학생별 피드백 + 채점), results/module4_evaluation.json
      (제안 시스템 vs 일반 LLM 베이스라인의 정답 노출 비율 비교)

실제 Claude API 호출은 call_claude() 한 곳에서만 일어난다. 나머지 함수는 전부 call_claude()를
거쳐서만 LLM을 쓰므로, pytest에서는 client 인자에 목(mock) 객체를 넣어 API 키 없이도
프롬프트 구성·분기 로직·규칙 기반 필터를 검증할 수 있다.
"""

import json
import os
import re
import sys

import anthropic

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONCEPT_CARDS_PATH = os.path.join(ROOT, "data", "concept_cards.json")
PROBLEM_META_PATH = os.path.join(ROOT, "data", "problem_meta.json")
MODULE2_OUTPUT_PATH = os.path.join(ROOT, "results", "module2_output.json")
MODULE3_OUTPUT_PATH = os.path.join(ROOT, "results", "module3_output.json")
MODULE4_OUTPUT_PATH = os.path.join(ROOT, "results", "module4_output.json")
MODULE4_EVAL_PATH = os.path.join(ROOT, "results", "module4_evaluation.json")

# 연구계획서 예시("claude-sonnet 계열 모델")를 기본값으로 쓰되, 환경변수로 바꿔 쓸 수 있게 함
DEFAULT_MODEL = os.environ.get("MODULE4_MODEL", "claude-sonnet-5")
TARGET_LEAK_RATE = 0.10

NO_CANDIDATE_FEEDBACK = (
    "축하해요, 이번 파일럿에 준비된 문제를 모두 풀었어요! 지금까지 배운 개념들을 복습하면서 "
    "스스로 비슷한 문제를 만들어보는 것도 좋은 연습이 됩니다."
)

# ---------------------------------------------------------------------------
# 0. 개념카드 로드
# ---------------------------------------------------------------------------

def load_concept_cards(path=CONCEPT_CARDS_PATH) -> dict:
    """concept_cards.json을 {개념군: 카드dict} 형태로 로드."""
    with open(path, encoding="utf-8") as f:
        cards = json.load(f)
    return {c["concept"]: c for c in cards}


# ---------------------------------------------------------------------------
# 1. 프롬프트 & Claude 호출
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """당신은 초보 프로그래밍 학습자를 돕는 코딩 튜터입니다. 학생의 최근 학습 데이터를 \
바탕으로 짧은 학습 피드백을 작성합니다. 다음 규칙을 반드시 지키세요.

1. 정답 코드나 완성된 로직/알고리즘을 절대 먼저 제시하지 마세요. 코드 스니펫, 의사코드, \
단계별 풀이 과정을 알려주는 것도 금지입니다. 개념과 점검할 지점만 안내하세요.
2. 전체 답변은 3~4문장 이내로 짧게 작성하세요.
3. 다음 네 가지를 자연스러운 문장으로 모두 포함하세요:
   (a) 지금 어떤 개념에서 어려움을 겪고 있는지
   (b) 그 개념에서 구체적으로 어떤 포인트를 다시 확인해야 하는지 (개념카드의 '자주 틀리는 포인트' 활용)
   (c) 다음에 어떤 문제를 풀면 좋을지
   (d) 문제를 풀기 전에 무엇을 먼저 점검하면 좋을지 (개념카드의 '풀이 전 점검 항목' 활용)
4. 학생을 평가하거나 지치게 하는 말투가 아니라, 격려하고 응원하는 말투를 사용하세요."""

BASELINE_SYSTEM_PROMPT = """당신은 초보 프로그래밍 학습자를 돕는 코딩 튜터입니다. 주어진 문제에 대해 \
3~4문장 이내의 짧은 학습 피드백을 작성하세요. 정답 코드나 완성된 로직/알고리즘은 절대 먼저 \
제시하지 마세요."""

JUDGE_SYSTEM_PROMPT = """당신은 프로그래밍 교육 피드백을 검수하는 채점자입니다. 주어진 문제와 \
피드백 텍스트를 보고, 그 피드백이 문제의 정답 코드나 완성된 로직/알고리즘을 학생에게 직접 \
알려주는지 판단하세요. 개념을 설명하거나 점검할 지점을 안내하는 것은 정답 제시가 아닙니다. \
완성된 코드, 의사코드, "이렇게 하면 풀린다"는 식의 구체적 해법 절차가 있어야 정답 제시로 \
판단하세요.

반드시 아래 형식으로만 답하세요:
판정: 예 또는 아니오
이유: (한 문장)"""


def build_feedback_prompt(weakest_concept, concept_scores, recommended_problem_id,
                           recommendation_reason, concept_card) -> str:
    """모듈②③ 출력 + 개념카드를 하나의 사용자 메시지로 조립."""
    score = (concept_scores or {}).get(weakest_concept)
    score_pct = f"{score:.0%}" if score is not None else "정보 없음"
    description = concept_card["description"] if concept_card else "정보 없음"
    mistakes = "; ".join(concept_card["common_mistakes"]) if concept_card else "정보 없음"
    checklist = "; ".join(concept_card["pre_check_items"]) if concept_card else "정보 없음"

    return (
        "[학생 상태]\n"
        f"- 취약 개념: {weakest_concept}\n"
        f"- 취약도(높을수록 취약): {score_pct}\n"
        f"- 개념 설명: {description}\n"
        f"- 이 개념에서 자주 틀리는 포인트: {mistakes}\n"
        f"- 풀기 전 점검 항목: {checklist}\n\n"
        "[다음 추천 문제]\n"
        f"- 추천 문제 번호: {recommended_problem_id}\n"
        f"- 추천 이유: {recommendation_reason}\n\n"
        "위 정보를 바탕으로 학생에게 보여줄 피드백을 작성하세요."
    )


def call_claude(system: str, user_message: str, model: str = DEFAULT_MODEL, client=None,
                 max_tokens: int = 300) -> str:
    """실제 Claude API 호출부. 이 함수만 네트워크를 탄다.
    client를 넘기지 않으면 anthropic.Anthropic()을 새로 만든다 — 이 생성자는 자격증명이 없으면
    바로 에러를 내므로, 테스트에서는 반드시 client에 목(mock) 객체를 넘겨서 호출해야 한다."""
    if client is None:
        client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return next((block.text for block in response.content if block.type == "text"), "").strip()


def generate_feedback(weakest_concept, concept_scores, recommended_problem_id, recommendation_reason,
                       concept_card, model: str = DEFAULT_MODEL, client=None) -> str:
    """모듈②③ 출력 기반 제안 시스템 피드백 생성."""
    user_message = build_feedback_prompt(
        weakest_concept, concept_scores, recommended_problem_id, recommendation_reason, concept_card
    )
    return call_claude(SYSTEM_PROMPT, user_message, model=model, client=client)


def generate_baseline_feedback(problem_requirement: str, model: str = DEFAULT_MODEL, client=None) -> str:
    """비교군(베이스라인): 취약 개념 정보 없이 문제 설명만 주고 자유롭게 피드백하게 함."""
    user_message = f"문제 설명: {problem_requirement}\n\n이 문제를 아직 풀지 못한 학생에게 줄 피드백을 작성하세요."
    return call_claude(BASELINE_SYSTEM_PROMPT, user_message, model=model, client=client)


# ---------------------------------------------------------------------------
# 2. "정답 직접 제시 여부" 자동 평가기
# ---------------------------------------------------------------------------

# 1차 규칙 기반 필터: 완성된 코드/구문 패턴이 피드백 안에 그대로 등장하는지 탐지
CODE_LEAK_PATTERNS = [
    r"```",                                  # 마크다운 코드 블록
    r"\bpublic\s+\w+\s+\w+\s*\(",             # 메서드 시그니처 (public int foo( 등)
    r"\breturn\s+.{0,80};",                   # 완성된 return 문
    r"\bif\s*\([^)]*\)\s*\{",                 # if (...) {
    r"\bfor\s*\([^;]*;[^;]*;[^)]*\)\s*\{",    # for (...;...;...) {
    r"\bwhile\s*\([^)]*\)\s*\{",              # while (...) {
]


def rule_based_leak_check(feedback_text: str) -> bool:
    """완성된 코드 블록/구문이 피드백에 그대로 등장하는지 정규식으로 1차 탐지."""
    return any(re.search(pattern, feedback_text) for pattern in CODE_LEAK_PATTERNS)


def judge_direct_answer_leak(feedback_text: str, problem_requirement: str, model: str = DEFAULT_MODEL,
                              client=None):
    """2차 LLM-as-judge: 별도 Claude 호출로 정답 직접 제시 여부와 이유를 받아온다.
    반환: (leak: bool, reasoning: str)"""
    user_message = (
        f"문제: {problem_requirement}\n\n"
        f"피드백: {feedback_text}\n\n"
        "위 피드백이 이 문제의 정답 코드나 완성된 로직을 직접 알려줍니까?"
    )
    verdict_text = call_claude(JUDGE_SYSTEM_PROMPT, user_message, model=model, client=client, max_tokens=100)
    leak = bool(re.search(r"판정\s*[:：]\s*예", verdict_text))
    reasoning_match = re.search(r"이유\s*[:：]\s*(.+)", verdict_text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else verdict_text
    return leak, reasoning


def evaluate_feedback(feedback_text: str, problem_requirement: str, model: str = DEFAULT_MODEL,
                       client=None) -> dict:
    """1차 규칙 기반 필터 + 2차 LLM-judge를 합쳐 최종 '정답 직접 제시' 여부를 판정.
    규칙 기반에서 이미 걸리면 LLM judge 호출 없이 바로 leak=True로 판정한다(비용 절약)."""
    if rule_based_leak_check(feedback_text):
        return {
            "direct_answer_leak": True,
            "judge_reasoning": "규칙 기반 필터에서 완성된 코드 패턴이 탐지되어 LLM 판정 없이 확정",
        }

    leak, reasoning = judge_direct_answer_leak(feedback_text, problem_requirement, model=model, client=client)
    return {"direct_answer_leak": leak, "judge_reasoning": reasoning}


# ---------------------------------------------------------------------------
# 3. 학생별 피드백 생성 (module2/3 출력 기준 — results/module4_output.json 스키마)
# ---------------------------------------------------------------------------

def build_module4_output(module2_records, module3_records, concept_cards, problem_meta_by_id,
                          model: str = DEFAULT_MODEL, client=None):
    """module2_output.json의 각 학생 레코드에 대해 피드백 텍스트 + 채점 결과를 생성.
    module3에서 recommended_problem_id가 없는 학생(no_candidate)은 LLM을 호출하지 않고
    정형화된 축하 문구를 사용한다 — 추천할 문제가 없는데 억지로 피드백을 지어낼 필요는 없다."""
    module3_by_subject = {r["subject_id"]: r for r in module3_records}
    results = []

    for rec in module2_records:
        sid = rec["subject_id"]
        weakest = rec["weakest_concept"]
        m3 = module3_by_subject.get(sid)

        if m3 is None or m3.get("recommended_problem_id") is None:
            results.append(
                {
                    "subject_id": sid,
                    "feedback_text": NO_CANDIDATE_FEEDBACK,
                    "direct_answer_leak": False,
                    "judge_reasoning": "파일럿 문항을 모두 해결해 정형 문구를 사용, LLM 미호출",
                }
            )
            continue

        concept_card = concept_cards.get(weakest)
        feedback_text = generate_feedback(
            weakest, rec["concept_scores"], m3["recommended_problem_id"], m3["reason"],
            concept_card, model=model, client=client,
        )
        problem = problem_meta_by_id.get(m3["recommended_problem_id"])
        requirement = problem["requirement"] if problem else ""
        eval_result = evaluate_feedback(feedback_text, requirement, model=model, client=client)

        results.append(
            {
                "subject_id": sid,
                "feedback_text": feedback_text,
                "direct_answer_leak": eval_result["direct_answer_leak"],
                "judge_reasoning": eval_result["judge_reasoning"],
            }
        )

    return results


# ---------------------------------------------------------------------------
# 4. 평가 — 파일럿 문항 전체 vs 일반 LLM 베이스라인 (목표: 정답 노출 비율 10% 이하)
# ---------------------------------------------------------------------------

def generate_problem_level_feedback(problem_meta_list, concept_cards, model: str = DEFAULT_MODEL,
                                     client=None):
    """평가용 데이터셋 생성: module3_output.json 기준 실제 추천을 받은 학생은 2명뿐이라
    평가 표본이 너무 적다. 그래서 파일럿 12문항 전체 각각에 대해 '이 문항이 방금
    추천됐다고 가정'하고 제안 시스템 피드백과 베이스라인 피드백을 하나씩 생성해
    표본을 문항 수(8~12개)만큼 확보한다."""
    records = []
    for problem in problem_meta_list:
        weakest_concept = problem["concept_groups"][0]
        concept_card = concept_cards.get(weakest_concept)
        reason = f"{weakest_concept} 개념 점검을 위해 추천된 문항(평가용 가정 시나리오)"

        proposed = generate_feedback(
            weakest_concept, {weakest_concept: 0.6}, problem["problem_id"], reason,
            concept_card, model=model, client=client,
        )
        baseline = generate_baseline_feedback(problem["requirement"], model=model, client=client)

        records.append(
            {
                "problem_id": problem["problem_id"],
                "proposed_feedback": proposed,
                "baseline_feedback": baseline,
            }
        )
    return records


def evaluate_module4(problem_level_records, problem_meta_by_id, model: str = DEFAULT_MODEL, client=None) -> dict:
    """파일럿 문항 전체에 대해 제안 시스템 vs 일반 LLM 베이스라인의 정답 노출 비율을 비교."""
    proposed_leaks, baseline_leaks = [], []
    details = []

    for rec in problem_level_records:
        requirement = problem_meta_by_id[rec["problem_id"]]["requirement"]
        proposed_eval = evaluate_feedback(rec["proposed_feedback"], requirement, model=model, client=client)
        baseline_eval = evaluate_feedback(rec["baseline_feedback"], requirement, model=model, client=client)

        proposed_leaks.append(proposed_eval["direct_answer_leak"])
        baseline_leaks.append(baseline_eval["direct_answer_leak"])
        details.append(
            {
                "problem_id": rec["problem_id"],
                "proposed": proposed_eval,
                "baseline": baseline_eval,
            }
        )

    def _rate(bools):
        return round(sum(bools) / len(bools), 4) if bools else None

    proposed_rate = _rate(proposed_leaks)
    return {
        "n_problems": len(problem_level_records),
        "proposed": {
            "direct_answer_leak_rate": proposed_rate,
            "n_leaked": int(sum(proposed_leaks)),
        },
        "baseline_generic_llm": {
            "direct_answer_leak_rate": _rate(baseline_leaks),
            "n_leaked": int(sum(baseline_leaks)),
        },
        "target_leak_rate": TARGET_LEAK_RATE,
        "target_achieved": bool(proposed_rate is not None and proposed_rate <= TARGET_LEAK_RATE),
        "details": details,
    }


# ---------------------------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------------------------

def main():
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("ANTHROPIC_API_KEY(또는 ANTHROPIC_AUTH_TOKEN)가 설정되어 있지 않아 실행을 중단합니다.")
        print("환경변수를 설정한 뒤 다시 실행하세요. 예: export ANTHROPIC_API_KEY=sk-ant-...")
        print("(API 키 없이도 코드 구조·프롬프트·평가 로직은 pytest로 검증할 수 있습니다: "
              "pytest tests/test_feedback.py -v)")
        sys.exit(1)

    concept_cards = load_concept_cards()

    with open(PROBLEM_META_PATH, encoding="utf-8") as f:
        problem_meta_bundle = json.load(f)
    problem_meta_list = problem_meta_bundle["problems"]
    problem_meta_by_id = {p["problem_id"]: p for p in problem_meta_list}

    with open(MODULE2_OUTPUT_PATH, encoding="utf-8") as f:
        module2_records = json.load(f)
    with open(MODULE3_OUTPUT_PATH, encoding="utf-8") as f:
        module3_records = json.load(f)

    print("[1/3] 학생별 피드백 생성 중 (module2/3 출력 기준)...")
    module4_outputs = build_module4_output(module2_records, module3_records, concept_cards, problem_meta_by_id)
    with open(MODULE4_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(module4_outputs, f, ensure_ascii=False, indent=2)
    print(f"  완료: 학생 {len(module4_outputs)}명 피드백 생성")

    print("[2/3] 파일럿 문항 전체에 대한 제안/베이스라인 피드백 생성 중 (평가용)...")
    problem_level_records = generate_problem_level_feedback(problem_meta_list, concept_cards)
    print(f"  완료: 문항 {len(problem_level_records)}개")

    print("[3/3] 정답 직접 제시 여부 평가 중...")
    evaluation = evaluate_module4(problem_level_records, problem_meta_by_id)
    with open(MODULE4_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    print(f"  제안 시스템 정답 노출 비율: {evaluation['proposed']['direct_answer_leak_rate']}")
    print(f"  베이스라인 정답 노출 비율: {evaluation['baseline_generic_llm']['direct_answer_leak_rate']}")
    print(f"  목표(10% 이하) 달성 여부: {evaluation['target_achieved']}")
    print("완료. results/module4_output.json / results/module4_evaluation.json 생성됨")


if __name__ == "__main__":
    main()
