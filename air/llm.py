"""Pluggable LLM backends.  Each backend implements ``complete(system, user) -> str``.

* ``gemini``  : Google AI Studio REST API (no SDK). Key from env ``GEMINI_API_KEY``.
* ``gemini-cli``: shells out to Google's ``gemini`` CLI (uses a logged-in Google account's quota).
* ``claude-cli``: shells out to ``claude -p`` (Claude Code CLI) if installed.
* ``file``    : returns a hand-written response from a file (used to disclose /
                reproduce a manual intervention, or to replay a logged response).
* ``mock``    : deterministic fake proposals, for testing the pipeline without an API key.
"""
import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class LLMError(RuntimeError):
    pass


class Backend:
    name = "base"
    model = ""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model}


class GeminiBackend(Backend):
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str, temperature: float = 0.7, max_retries: int = 5, timeout: int = 600):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise LLMError("GEMINI_API_KEY is not set (export it in your shell; never commit it)")

    def complete(self, system: str, user: str) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": 32768},
        }
        data = json.dumps(body).encode()
        last = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.URL.format(model=self.model), data=data, method="POST",
                headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                cands = resp.get("candidates") or []
                if not cands:
                    raise LLMError(f"no candidates in response: {json.dumps(resp)[:500]}")
                parts = cands[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                if not text.strip():
                    raise LLMError(f"empty text, finishReason={cands[0].get('finishReason')}")
                return text
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                try:
                    msg = json.loads(body)["error"]["message"]
                except Exception:  # noqa: BLE001
                    msg = body
                last = f"HTTP {e.code}: {msg[:1500]}"
                # 'limit: 0' = this model has no quota on this key (e.g. free tier); retrying cannot help
                if e.code not in (429, 500, 502, 503, 504) or "limit: 0" in msg:
                    raise LLMError(last) from e
            except (urllib.error.URLError, TimeoutError, LLMError) as e:
                last = str(e)
            wait = min(60, 5 * 2 ** attempt)
            print(f"[llm] gemini call failed ({last}); retry in {wait}s", flush=True)
            time.sleep(wait)
        raise LLMError(f"gemini failed after {self.max_retries} attempts: {last}")


class ClaudeCLIBackend(Backend):
    name = "claude-cli"

    def __init__(self, model: str = "", timeout: int = 900):
        self.model = model or "default"
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        cmd = ["claude", "-p", "--append-system-prompt", system]
        if self.model != "default":
            cmd += ["--model", self.model]
        r = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise LLMError(f"claude -p failed: {r.stderr[-500:]}")
        return r.stdout


class GeminiCLIBackend(Backend):
    """Google's `gemini` CLI (npm @google/gemini-cli). When logged in with a Google
    account it uses that account's Gemini subscription quota instead of an API key.
    Runs in an empty temp directory so the CLI's own file tools see nothing."""
    name = "gemini-cli"

    def __init__(self, model: str = "", timeout: int = 900):
        self.model = model or "default"
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        import tempfile
        prompt = f"{system}\n\n{user}"
        cmd = ["gemini", "-p", prompt]
        if self.model != "default":
            cmd += ["-m", self.model]
        env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        with tempfile.TemporaryDirectory() as cwd:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=cwd,
                               stdin=subprocess.DEVNULL, env=env)
        if r.returncode != 0:
            raise LLMError(f"gemini CLI failed: {r.stderr[-1500:]}")
        return r.stdout


class FileBackend(Backend):
    """Return a fixed response from a file (manual intervention / replay)."""
    name = "file"

    def __init__(self, path: str):
        self.path = Path(path)
        self.model = f"file:{self.path.name}"

    def complete(self, system: str, user: str) -> str:
        return self.path.read_text()


class MockBackend(Backend):
    """Fake LLM for pipeline tests: tweaks a hyperparameter of the current best code."""
    name = "mock"

    def __init__(self, seed: int = 0):
        self.model = "mock"
        self.seed = seed

    def complete(self, system: str, user: str) -> str:
        m = re.search(r"<current_best_algorithm>\n```python\n(.*?)```", user, re.S)
        code = m.group(1)
        new_lambda = random.Random(len(user) + self.seed).choice(["2e-3", "1e-2", "3.9e-3"])
        code = re.sub(r"'lambda_param': [0-9.e-]+", f"'lambda_param': {new_lambda}", code)
        return ("```json\n" + json.dumps({
            "hypothesis": f"Changing the Barlow Twins off-diagonal weight to {new_lambda} improves accuracy.",
            "rationale": "mock backend: random hyperparameter perturbation for pipeline testing",
            "expected_effect": "+0.5% val acc", "change_summary": f"lambda_param -> {new_lambda}",
        }, indent=2) + "\n```\n\n```python\n" + code + "```\n")


def make_backend(name: str, model: str = "", response_file: str = "") -> Backend:
    if name == "gemini":
        return GeminiBackend(model or os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    if name == "gemini-cli":
        return GeminiCLIBackend(model)
    if name == "claude-cli":
        return ClaudeCLIBackend(model)
    if name == "file":
        return FileBackend(response_file)
    if name == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend {name}")


def list_gemini_models():
    key = os.environ["GEMINI_API_KEY"]
    req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                 headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        for m in json.loads(r.read()).get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].removeprefix("models/"))


if __name__ == "__main__":
    list_gemini_models()
