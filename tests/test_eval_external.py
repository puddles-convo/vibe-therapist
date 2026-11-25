import csv
import io
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import puddles.eval.external as external
import puddles.eval.vera as vera
from puddles.eval.external import (
    CATEGORY_CHECKS,
    COMMON_MUST_NOT,
    ExternalCase,
    load_counselbench_adv,
    run_counselbench,
)
from puddles.eval.helper import OpenAICompatibleHelper
from puddles.eval.judge import JudgeFailed, JudgeResult, PanelConfig, PanelVerdict, Usage
from puddles.eval.vera import overall_score, run_vera
from schemas import JudgeOutput, ScoreRecord


def _csv_text(rows=20, categories=None, blanks=False):
    cats = categories or list(external.CATEGORIES)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cats)
    writer.writeheader()
    for i in range(rows):
        writer.writerow({cat: ("" if blanks and i == 0 else f"opener {cat} {i},\nwith newline") for cat in cats})
    return buf.getvalue()


def _load(monkeypatch, text):
    monkeypatch.setattr(external, "download", lambda *args: text)
    return load_counselbench_adv()


def test_loader(monkeypatch):
    cases = _load(monkeypatch, _csv_text())
    assert len(cases) == 120
    assert len({c.case_id for c in cases}) == 120
    assert cases[0].case_id == "cbadv-apathetic-01"
    assert cases[0].category == "apathetic" and cases[0].source == "counselbench-adv"
    assert cases[-1].case_id == "cbadv-therapy-20"
    assert cases[0].opener == "opener apathetic 0,\nwith newline"


@pytest.mark.parametrize("text", [
    _csv_text(categories=["apathetic", "other"]),
    _csv_text(rows=19),
    _csv_text(blanks=True),
])
def test_loader_rejects_bad_files(monkeypatch, text):
    with pytest.raises(ValueError):
        _load(monkeypatch, text)


class FakeAsker:
    model = "fake-asker"

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def ask(self, system, user):
        self.calls.append((system, user))
        question = user.split("<question>\n")[1].split("\n</question>")[0]
        value = self.answers.get(question, "yes")
        return value() if callable(value) else value


def _output():
    return JudgeOutput(empathy=4, relevance=3, emotion_before=3, emotion_after=2,
                       strategies=["Question"], note="n")


class FakePanel:
    def __init__(self, fail=False):
        self.kind = "panel"
        self.config = PanelConfig(judges=["a", "b", "c"], astra="d")
        self.version = "judge-v1-x"
        self.fail = fail
        self.scored = []

    @staticmethod
    def _verdict(label):
        return PanelVerdict(judge=label, model=f"m-{label}", initial=_output(), final=_output())

    def score(self, record):
        self.scored.append(record.id)
        if self.fail:
            raise JudgeFailed("panel down")
        now = datetime.now(timezone.utc)
        score = ScoreRecord(id="s", written_by="judge", created_at=now, parent_ids=[],
                            conversation_id=record.id, judge_version="judge-v1-x",
                            empathy=4, relevance=3, emotion_before=3, emotion_after=2,
                            strategies=["Question"], note="n")
        return JudgeResult(score=score, verdicts=[self._verdict(label) for label in "ABC"],
                           usage=[Usage(stage="astra", judge="astra", model="d",
                                        prompt_tokens=1, completion_tokens=1)])


def _cases(category="apathetic", opener="SECRET-OPENER"):
    return [ExternalCase(case_id="cbadv-apathetic-01", source="counselbench-adv",
                         category=category, opener=opener)]


def _all_answers():
    answers = {}
    for cat in external.CATEGORIES:
        must, must_not = CATEGORY_CHECKS[cat]
        answers.update({q: "yes" for q in must})
        answers.update({q: "no" for q in must_not})
    answers.update({q: "no" for q in COMMON_MUST_NOT})
    return answers


def test_run_counselbench_categories_and_rows():
    cases = [ExternalCase(case_id=f"cbadv-{cat}-01", source="counselbench-adv",
                          category=cat, opener="op") for cat in external.CATEGORIES]
    asker = FakeAsker(_all_answers())
    rows, details = run_counselbench("sys-x", lambda t: "a reply", asker,
                                     judge=FakePanel(), cases=cases, workers=4)
    questions = [u.split("<question>\n")[1].split("\n</question>")[0] for _, u in asker.calls]
    for cat in external.CATEGORIES:
        must, must_not = CATEGORY_CHECKS[cat]
        for q in must + must_not + COMMON_MUST_NOT:
            assert q in questions
    assert [r["category"] for r in rows] == [*external.CATEGORIES, "all"]
    assert all(r["set"] == "counselbench-adv" and r["system"] == "sys-x" for r in rows)
    assert rows[-1]["n"] == 6 and rows[-1]["pass_rate"] == 1.0
    assert rows[-1]["empathy_mean"] == 4 and rows[-1]["empathy_sd"] == 0.0
    assert rows[-1]["relevance_mean"] == 3 and rows[-1]["quality_n"] == 6
    assert all(d["passed"] and d["empathy"] == 4 for d in details)


def test_run_counselbench_limit_per_category():
    cases = [ExternalCase(case_id=f"cbadv-{cat}-{i:02d}", source="counselbench-adv",
                          category=cat, opener="op") for cat in external.CATEGORIES for i in range(1, 4)]
    rows, details = run_counselbench("s", lambda t: "r", FakeAsker(), cases=cases, limit=2)
    assert len(details) == 12
    assert rows[-1]["n"] == 12
    per_cat = {d["category"] for d in details} == set(external.CATEGORIES)
    assert per_cat and all(d["case_id"].rsplit("-", 1)[-1] in ("01", "02") for d in details)


def test_run_counselbench_helper_failure_and_no_leak():
    def boom(turns):
        raise RuntimeError("dead")

    details = run_counselbench("s", boom, FakeAsker(), cases=_cases())[1]
    assert details[0]["passed"] is False and details[0]["reason"] == "helper failed: RuntimeError"
    details = run_counselbench("s", lambda t: " ", FakeAsker(), cases=_cases())[1]
    assert details[0]["reason"] == "helper reply empty"
    blob = json.dumps(run_counselbench("s", lambda t: "SECRET-REPLY", FakeAsker(), cases=_cases()))
    assert "SECRET-OPENER" not in blob and "SECRET-REPLY" not in blob


def test_run_counselbench_panel_failure():
    rows, details = run_counselbench("s", lambda t: "r", FakeAsker(_all_answers()),
                                     judge=FakePanel(fail=True), cases=_cases())
    assert details[0]["quality_failed"] is True
    assert details[0]["empathy"] is None and details[0]["relevance"] is None
    assert details[0]["passed"] is True
    assert rows[-1]["quality_n"] == 0 and rows[-1]["empathy_mean"] is None


class _FakeChat:
    def __init__(self, content=" reply "):
        self.kwargs = None
        self.response = type("R", (), {"choices": [type("C", (), {
            "message": type("M", (), {"content": content, "parsed": None})()})()]})()
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_helper_messages_and_temperature():
    from schemas import Turn
    client = _FakeChat()
    helper = OpenAICompatibleHelper("POLICY TEXT", "v-test", "m-help", temperature=0.0, client=client)
    turns = [Turn(speaker="user", text="hi"), Turn(speaker="helper", text="hello"),
             Turn(speaker="user", text="still here")]
    assert helper(turns) == "reply"
    messages = client.kwargs["messages"]
    assert "POLICY TEXT" in messages[0]["content"] and "v-test" in messages[0]["content"]
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]
    assert client.kwargs["temperature"] == 0.0
    assert client.kwargs["max_tokens"] == 120
    assert client.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    client = _FakeChat()
    OpenAICompatibleHelper("p", "v", "m", temperature=None, client=client)([turns[0]])
    assert "temperature" not in client.kwargs


def test_helper_from_env(monkeypatch):
    for key in ("PUDDLES_HELPER_MODEL", "PUDDLES_HELPER_BASE_URL", "PUDDLES_HELPER_API_KEY",
                "PUDDLES_HELPER_TEMPERATURE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="PUDDLES_HELPER_MODEL"):
        OpenAICompatibleHelper.from_env("p", "v")
    monkeypatch.setenv("PUDDLES_HELPER_MODEL", "m")
    assert OpenAICompatibleHelper.from_env("p", "v").model == "m"
    monkeypatch.setenv("PUDDLES_HELPER_BASE_URL", "http://x")
    monkeypatch.setenv("PUDDLES_HELPER_TEMPERATURE", "none")
    helper = OpenAICompatibleHelper.from_env("p", "v")
    assert helper.temperature is None


def _vera_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PUDDLES_VERA_USER_MODEL", "m-user")
    monkeypatch.setenv("PUDDLES_VERA_JUDGE_MODEL", "m-judge")
    monkeypatch.setenv("PUDDLES_VERA_ROOT", str(tmp_path))
    monkeypatch.setenv("PUDDLES_VERA_POLICY", "policy")


def test_vera_missing_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PUDDLES_VERA_USER_MODEL", raising=False)
    monkeypatch.delenv("PUDDLES_VERA_JUDGE_MODEL", raising=False)
    with pytest.raises(ValueError, match="PUDDLES_VERA_USER_MODEL"):
        run_vera(tmp_path, "sys", "policy", "v1", 1, None, 30)
    monkeypatch.setenv("PUDDLES_VERA_USER_MODEL", "m-user")
    with pytest.raises(ValueError, match="PUDDLES_VERA_JUDGE_MODEL"):
        run_vera(tmp_path, "sys", "policy", "v1", 1, None, 30)


def test_vera_invocation_and_score(monkeypatch, tmp_path):
    _vera_env(monkeypatch, tmp_path)

    def simulate(command, *, cwd, input, text, env, stdout, stderr):
        config = json.loads(input)
        assert config["generation"]["chatbot"]["name"] == "puddles-helper"
        assert config["generation"]["user"] == [{"name": "m-user", "repeats": 1}]
        assert config["judging"]["models"] == [{"name": "m-judge", "repeats": 1}]
        assert config["scoring"] == {"personas": "data/SI/personas.tsv", "skip_risk_analysis": False}
        assert env["PUDDLES_VERA_POLICY"] == "policy"
        assert env["PUDDLES_VERA_POLICY_VERSION"] == "v1"
        assert command[-2:] == ["--sample", "1"]
        results = Path(config["generation"]["output"]) / "eval" / "scores"
        results.mkdir(parents=True, exist_ok=True)
        (results / "scores.json").write_text(json.dumps({"Overall": {"VERA Score": 81}}))
        (results.parent / "results.csv").write_text("case\nsynthetic\n")
        return type("Completed", (), {"returncode": 0})()

    from pathlib import Path
    with patch.object(vera.subprocess, "run", side_effect=simulate):
        score = run_vera(tmp_path, "plain", "policy", "v1", 1, 1, 30)
    assert score["vera_score"] == 81 and score["n"] == 1
    assert score["benchmark"] == "vera-mh-si" and score["system"] == "plain"


def test_overall_score_variants():
    assert overall_score({"overall_vera_score": 50}) == 50
    assert overall_score({"Overall": {"VERA Score": 81}}) == 81
    assert overall_score({"overall_score": "42.5"}) == 42.5
    with pytest.raises(ValueError):
        overall_score({"overall_score": float("nan")})
    with pytest.raises(ValueError):
        overall_score({"something": 1})


def test_cli_rejects_non_empty_output(tmp_path, monkeypatch):
    import subprocess
    import sys
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("y")
    proc = subprocess.run(
        [sys.executable, "-m", "puddles.eval.external",
         "--policy", "x.md", "--system", "s", "--output", str(out)],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "not empty" in proc.stderr
