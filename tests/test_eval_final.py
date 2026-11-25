import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import puddles.eval.final as final_module
from puddles.eval.final import render_results_markdown, run_final, select_scenarios
from puddles.eval.helper import HELPER_SYSTEM, NO_NOTES, policy_version_of, render_helper_system
from puddles.eval.simulate import (
    BEGIN,
    END,
    SIM_SYSTEM,
    OpenAICompatibleUser,
    run_conversation,
    scenario_seed,
)
from schemas import SafetyCaseResult, SafetyResult, Scenario, ScoreRecord, Turn

REPO = Path(__file__).parents[1]


def _doc_block(path, marker):
    doc = (REPO / path).read_text()
    return doc.split(marker, 1)[1].split("```", 2)[1].strip("\n")


def test_helper_system_matches_doc():
    block = _doc_block("docs/agents/t2-i3-helper.md", "### System message")
    block = block.replace('{user_notes, or "Nothing yet. This is your first conversation."}',
                          "{user_notes}")
    assert HELPER_SYSTEM == block


def test_render_helper_system():
    rendered = render_helper_system("C1. Listen first.", "policy-v9")
    assert '<policy version="policy-v9">' in rendered and "C1. Listen first." in rendered
    assert NO_NOTES in rendered
    rendered = render_helper_system("p", "v", ["note one", "note two"])
    assert "note one\nnote two" in rendered and NO_NOTES not in rendered


def test_policy_version_of():
    assert policy_version_of("# policy-v9\nrules", "fb") == "policy-v9"
    assert policy_version_of("no heading", "fb") == "fb"


def test_sim_system_matches_doc():
    assert SIM_SYSTEM == _doc_block("docs/agents/t1-simulated-user.md", "### System message")


class _FakeChat:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": self})()
        self.kwargs = None
        self.response = type("R", (), {"choices": [type("C", (), {
            "message": type("M", (), {"content": content})()})()], "usage": None})()

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _scenario(sid="esconv-test-0001"):
    return Scenario(scenario_id=sid, split="test", emotion_type="grief",
                    problem_type="loss", situation="SIT", initial_intensity=4)


def test_user_next_turn_roles_and_seed():
    client = _FakeChat("hello")
    user = OpenAICompatibleUser("m-user", client=client)
    scenario = _scenario()
    assert user.next_turn(scenario, []) == "hello"
    system, messages = client.kwargs["messages"][0], client.kwargs["messages"]
    assert system["content"].startswith("You are playing a person")
    assert "SIT" in system["content"] and "grief" in system["content"] and "loss" in system["content"]
    assert messages[1:] == [{"role": "user", "content": BEGIN}]
    assert client.kwargs["seed"] == scenario_seed("esconv-test-0001")
    assert client.kwargs["temperature"] == 0.7
    assert client.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    turns = [Turn(speaker="user", text="sad"), Turn(speaker="helper", text="i hear you")]
    user.next_turn(scenario, turns)
    assert [m["role"] for m in client.kwargs["messages"][1:]] == ["assistant", "user"]


def test_user_from_env_errors(monkeypatch):
    monkeypatch.delenv("PUDDLES_USER_MODEL", raising=False)
    with pytest.raises(ValueError, match="PUDDLES_USER_MODEL"):
        OpenAICompatibleUser.from_env()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    monkeypatch.setenv("PUDDLES_USER_MODEL", "m")
    monkeypatch.setenv("PUDDLES_USER_TEMPERATURE", "none")
    assert OpenAICompatibleUser.from_env().temperature is None


class FakeUser:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def next_turn(self, scenario, turns):
        self.seen.append(scenario.scenario_id)
        return self.replies.pop(0) if self.replies else END


def test_run_conversation_end_variants():
    # END alone -> no trailing turn
    user = FakeUser(["first words", END])
    def helper(turns):
        return "a reply"
    record = run_conversation(_scenario(), helper, user, "conv-x", "pv")
    assert [(t.speaker, t.text) for t in record.turns] == [("user", "first words"), ("helper", "a reply")]
    assert record.source == "simulated" and record.scenario_id == "esconv-test-0001"
    assert record.split == "test" and record.policy_version == "pv" and record.written_by == "final-eval"

    # text then END -> keeps the text
    user = FakeUser(["u1", "last words " + END])
    record = run_conversation(_scenario(), helper, user, "conv-x", "pv")
    assert [t.text for t in record.turns] == ["u1", "a reply", "last words"]

    # cap on helper turns
    user = FakeUser(["u"] * 20)
    record = run_conversation(_scenario(), helper, user, "conv-x", "pv", max_helper_turns=3)
    assert sum(1 for t in record.turns if t.speaker == "helper") == 3

    # empty helper reply -> ValueError
    with pytest.raises(ValueError, match="empty"):
        run_conversation(_scenario(), lambda t: " ", FakeUser(["u"]), "conv-x", "pv")


class FakeJudge:
    kind = "panel"
    config = None

    def __init__(self, scores=None, fail_on=()):
        self.version = "judge-v1-test"
        self.scores = scores or {}
        self.fail_on = fail_on
        self.scored = []

    def score(self, record):
        self.scored.append(record.id)
        if record.scenario_id in self.fail_on:
            raise RuntimeError("judge down")
        empathy, relevance = self.scores.get(record.scenario_id, (3, 3))
        return type("R", (), {"score": ScoreRecord(
            id=f"score-{record.id}", written_by="judge",
            created_at=datetime.now(timezone.utc), parent_ids=[record.id],
            conversation_id=record.id, judge_version=self.version,
            empathy=empathy, relevance=relevance, emotion_before=4, emotion_after=2,
            strategies=["Question"], note="n")})()


class FakeAsker:
    model = "a"


def _safety(passed):
    return SafetyResult(policy_version="pv", passed=passed, cases=[
        SafetyCaseResult(case_id="c1", passed=passed),
        SafetyCaseResult(case_id="c2", passed=True)])


def _systems():
    return {"plain": ("text-p", "policy-plain"), "handwritten": ("text-h", "policy-hw"),
            "final": ("text-f", "policy-final")}


def _make_helper(text, version):
    return lambda turns: f"reply-{version}"


def _scenarios(n=2):
    return [_scenario(f"esconv-test-{i:04d}") for i in range(n)]


def test_run_final_all_systems_same_scenarios(tmp_path, monkeypatch):
    monkeypatch.setattr(final_module, "run_safety", lambda v, h, a: _safety(True))
    judge = FakeJudge(scores={"esconv-test-0000": (4, 2), "esconv-test-0001": (2, 4)})
    user = FakeUser(["u", "u" + END] * 10)
    rows, summary = run_final(_systems(), _scenarios(), _make_helper, user, judge,
                              FakeAsker(), tmp_path, turns=5, workers=2)
    esconv = {r["system"]: r for r in rows if r["set"] == "esconv-test"}
    assert set(esconv) == {"plain", "handwritten", "final"}
    for name, row in esconv.items():
        assert row["n"] == 2 and row["failed"] == 0
        assert row["empathy"] == 3 and row["relevance"] == 3
        assert row["quality"] == 3 and row["quality_sd"] == 0.0
        assert row["empathy_sd"] == 1.0
        assert row["emotion_change"] == 2 and row["emotion_change_sd"] == 0.0
        assert row["safety_pass_rate"] == 1.0 and row["safety_passed"] is True
        assert row["judge_version"] == "judge-v1-test" and row["policy_version"]
        convs = (tmp_path / "esconv-test" / name / "conversations.jsonl").read_text().strip().split("\n")
        ids = [json.loads(line)["id"] for line in convs]
        assert ids == [f"conv-final-{name}-esconv-test-{i:04d}" for i in range(2)]
        assert json.loads((tmp_path / "safety" / f"{name}.json").read_text())["passed"] is True
    assert len(judge.scored) == 6


def test_run_final_failures_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(final_module, "run_safety", lambda v, h, a: _safety(False))
    scenarios = _scenarios(3)

    def bad_helper(text, version):
        def reply(turns):
            if turns and turns[-1].text == "boom":
                raise RuntimeError("helper dead")
            return "ok"
        return reply

    user = FakeUser(["boom" if i == 1 else "u" for i in range(3)] + [END] * 10)
    judge = FakeJudge(fail_on={"esconv-test-0002"})
    rows, summary = run_final({"final": ("t", "pv")}, scenarios, bad_helper, user, judge,
                              FakeAsker(), tmp_path, turns=5, workers=2)
    row = rows[0]
    assert row["n"] == 1 and row["failed"] == 2
    failed = json.loads((tmp_path / "esconv-test" / "final" / "failures.json").read_text())
    stages = {f["scenario_id"]: f["stage"] for f in failed}
    # queue: 0000 gets "u" then "boom" (helper dies), 0001 gets "u" then END, 0002 gets END
    # then the judge fails -> one scored, two failures
    assert stages == {"esconv-test-0000": "conversation", "esconv-test-0002": "judge"}
    assert row["safety_pass_rate"] == 0.5 and row["safety_passed"] is False


def test_results_markdown(tmp_path):
    run = {"judge_version": "judge-v1-x", "judge_kind": "panel", "n_scenarios": 195,
           "n_scenarios_total": 195, "sample": None, "seed": 0}
    rows = [
        {"set": "esconv-test", "system": s, "policy_version": "v", "n": 195, "failed": 0,
         "empathy": e, "empathy_sd": 0.1, "relevance": 3.0, "relevance_sd": 0.1,
         "quality": e, "quality_sd": 0.1, "emotion_change": 1.0, "emotion_change_sd": 0.2,
         "safety_pass_rate": 1.0, "safety_passed": True, "judge_version": "judge-v1-x"}
        for s, e in (("plain", 3.0), ("handwritten", 3.5), ("final", 4.0))
    ]
    md = render_results_markdown(run, rows, {"plain": [], "handwritten": [], "final": []})
    assert "# Final evaluation" in md and "## ESConv test" in md
    assert "Change versus baselines" in md
    assert "| empathy | 1.00 | 0.50 |" in md
    assert "spread across scenarios should not be read as improvements" in md
    assert "## Failures" in md
    run2 = {**run, "sample": 50, "n_scenarios": 50}
    rows2 = rows + [{"set": "counselbench-adv", "system": "final", "category": "all", "n": 120,
                     "pass_rate": 1.0, "empathy_mean": None, "empathy_sd": None,
                     "relevance_mean": None, "relevance_sd": None, "quality_n": 0},
                    {"set": "vera-mh-si", "system": "final", "n": 100, "vera_score": 81.0}]
    md = render_results_markdown(run2, rows2, {"final": [{"scenario_id": "x", "stage": "judge", "error": "e"}]})
    assert "sampled 50 of 195 with seed 0" in md
    assert "## CounselBench-Adv" in md and "## VERA-MH" in md and "81.00" in md
    assert "—" in md and "final: 1 failed (stages: judge)" in md


def test_cli_rejects_non_empty_output(tmp_path):
    import subprocess
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("y")
    proc = subprocess.run(
        [".venv/bin/python", "-m", "puddles.eval.final",
         "--plain", "a", "--handwritten", "b", "--final", "c", "--output", str(out)],
        capture_output=True, text=True)
    assert proc.returncode != 0 and "not empty" in proc.stderr


def test_select_scenarios_deterministic_and_sorted():
    scenarios = _scenarios(6)
    a = select_scenarios(scenarios, 3, 7)
    b = select_scenarios(scenarios, 3, 7)
    ids = [s.scenario_id for s in a]
    assert ids == [s.scenario_id for s in b] and ids == sorted(ids) and len(ids) == 3
    assert select_scenarios(scenarios, None, 0) == scenarios
