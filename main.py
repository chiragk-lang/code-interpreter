import os
import re
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int] = Field(default_factory=list)


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    captured_output = StringIO()

    try:
        sys.stdout = captured_output
        exec(code, {"__name__": "__main__"})
        return {
            "success": True,
            "output": captured_output.getvalue()
        }
    except Exception:
        sys.stdout = old_stdout
        return {
            "success": False,
            "output": traceback.format_exc()
        }
    finally:
        sys.stdout = old_stdout


def fallback_error_lines(code: str, error_traceback: str) -> List[int]:
    """Used only if the Gemini call fails: read line numbers from the traceback."""
    total_lines = len(code.splitlines())
    lines = [int(n) for n in re.findall(r'File "<string>", line (\d+)', error_traceback)]
    lines = [n for n in lines if 1 <= n <= total_lines]
    return lines[-1:] if lines else []


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Analyze this Python code and traceback.
Identify the exact line number(s) in the submitted CODE where the error occurred.

Rules:
- Return only line numbers that exist in the submitted CODE.
- Do not invent line numbers.
- If multiple lines are relevant, return all relevant line numbers.

CODE:
{code}

TRACEBACK:
{error_traceback}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ErrorAnalysis,
        ),
    )

    analysis = ErrorAnalysis.model_validate_json(response.text)
    return analysis.error_lines


@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest) -> CodeResponse:
    execution = execute_python_code(request.code)

    if execution["success"]:
        return CodeResponse(
            error=[],
            result=execution["output"]
        )

    try:
        error_lines = analyze_error_with_ai(
            code=request.code,
            error_traceback=execution["output"]
        )
    except Exception:
        error_lines = fallback_error_lines(request.code, execution["output"])

    return CodeResponse(
        error=error_lines,
        result=execution["output"]
    )
