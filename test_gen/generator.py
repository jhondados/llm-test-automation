"""LLM-based test generator."""
from langchain_google_vertexai import ChatVertexAI
import ast
from pathlib import Path
from typing import List

class LLMTestGenerator:
    SYSTEM = """You are a senior test engineer. Generate comprehensive pytest unit tests.
Include: happy path, edge cases (None, empty, negative), boundary values, exceptions.
Use parametrize for multiple inputs. Add docstrings. Import only standard libraries + the module under test."""

    def __init__(self):
        self.llm = ChatVertexAI(model_name="gemini-1.5-pro-002", temperature=0.1)

    def generate_tests_for_file(self, source_file: str) -> str:
        source = Path(source_file).read_text()
        module_name = Path(source_file).stem
        prompt = f"""{self.SYSTEM}

Source code from {module_name}.py:
```python
{source}
```

Generate complete pytest test file. Include imports, fixtures and all test functions."""
        tests = self.llm.invoke(prompt).content
        # Clean up markdown code blocks
        if "```python" in tests:
            tests = tests.split("```python")[1].split("```")[0]
        return tests

    def find_test_gaps(self, source_file: str, existing_tests: str) -> List[str]:
        source = Path(source_file).read_text()
        prompt = f"""Analyze this source code and existing tests. List what is NOT tested:
Source: {source[:3000]}
Existing tests: {existing_tests[:2000]}
Return a list of missing test scenarios."""
        response = self.llm.invoke(prompt).content
        return [line.strip("- ") for line in response.split("\n") if line.strip().startswith("-")]
