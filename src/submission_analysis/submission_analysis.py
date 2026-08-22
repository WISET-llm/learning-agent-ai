# -*- coding: utf-8 -*-
"""
모듈 ① 제출 분석 (컴파일 판정 + 실패 패턴 태깅)
입력: 모듈1_2용_데이터셋.xlsx (StudentSubmissions)
출력: module1_output.json에 쓸 레코드 리스트
"""

import ast
import os
import subprocess
import tempfile

import pandas as pd

DATA_PATH = "모듈1_2용_데이터셋.xlsx"


def load_data(path=DATA_PATH):
    df = pd.read_excel(path, sheet_name="StudentSubmissions")
    df["binary_score"] = df["binary_score"].apply(ast.literal_eval)
    df["개념군_매핑"] = df["개념군_매핑"].astype(str)
    return df


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
