# ruff: noqa: E501
"""The final evaluation. The ESConv test split is used here once, after Team 2's final policy is frozen, and never before."""
import argparse
import hashlib
import json
import logging
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from puddles.eval.calibrate import check_frozen
from puddles.eval.data import ESCONV_REVISION, load_scenarios
from puddles.eval.external import CBADV_REVISION, run_counselbench
from puddles.eval.helper import OpenAICompatibleHelper, policy_version_of
from puddles.eval.judge import judge_from_env
from puddles.eval.safety import OpenAIYesNo, run_safety
from puddles.eval.simulate import OpenAICompatibleUser, run_conversation
from puddles.eval.vera import VERA_REVISION, run_vera
from schemas import Scenario

logger = logging.getLogger(__name__)


def _stats(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.pstdev(values)


def run_final(systems: dict[str, tuple[str, str]], scenarios: list[Scenario],
              make_helper: Callable[[str, str], Callable], user, judge, asker, output: Path,
              turns: int, workers: int,
              run_external: Optional[Callable[[str, Callable], list[dict]]] = None,
              ) -> tuple[list[dict], dict]:
    rows = []
    failures: dict[str, list[dict]] = {}
    for name, (policy_text, policy_version) in systems.items():
        helper = make_helper(policy_text, policy_version)
        sys_name, sys_version = name, policy_version
        system_dir = output / "esconv-test" / name
        system_dir.mkdir(parents=True, exist_ok=True)
        failed: list[dict] = []
        records = {}
        scores = {}

        def work(scenario: Scenario, name=sys_name, fn=helper, version=sys_version):
            record_id = f"conv-final-{name}-{scenario.scenario_id}"
            try:
                record = run_conversation(scenario, fn, user, record_id, version,
                                          max_helper_turns=turns)
            except Exception as e:
                return scenario.scenario_id, "conversation", e, None, None
            try:
                return scenario.scenario_id, None, None, record, judge.score(record).score
            except Exception as e:
                return scenario.scenario_id, "judge", e, record, None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(work, scenarios))
        for scenario_id, stage, error, record, score in outcomes:
            if record is not None:
                records[scenario_id] = record
            if score is not None:
                scores[scenario_id] = score
            if error is not None:
                entry = {"scenario_id": scenario_id, "stage": stage,
                         "error": f"{type(error).__name__}: {error}"}
                failed.append(entry)
                logger.warning("%s %s failed at %s: %s", name, scenario_id, stage, error)
        (system_dir / "conversations.jsonl").write_text(
            "".join(records[sid].model_dump_json() + "\n" for sid in records), encoding="utf-8")
        (system_dir / "scores.jsonl").write_text(
            "".join(scores[sid].model_dump_json() + "\n" for sid in scores), encoding="utf-8")
        (system_dir / "failures.json").write_text(json.dumps(failed, indent=2) + "\n", encoding="utf-8")
        failures[name] = failed

        safety = run_safety(policy_version, helper, asker)
        safety_dir = output / "safety"
        safety_dir.mkdir(parents=True, exist_ok=True)
        (safety_dir / f"{name}.json").write_text(safety.model_dump_json(indent=2), encoding="utf-8")

        scored = [scores[sid] for sid in scores]
        empathy_mean, empathy_sd = _stats([s.empathy for s in scored])
        relevance_mean, relevance_sd = _stats([s.relevance for s in scored])
        quality_mean, quality_sd = _stats([(s.empathy + s.relevance) / 2 for s in scored])
        change_mean, change_sd = _stats([s.emotion_before - s.emotion_after for s in scored])
        rows.append({
            "set": "esconv-test", "system": name, "policy_version": policy_version,
            "n": len(scored), "failed": len(failed),
            "empathy": empathy_mean, "empathy_sd": empathy_sd,
            "relevance": relevance_mean, "relevance_sd": relevance_sd,
            "quality": quality_mean, "quality_sd": quality_sd,
            "emotion_change": change_mean, "emotion_change_sd": change_sd,
            "safety_pass_rate": (sum(1 for c in safety.cases if c.passed) / len(safety.cases)
                                 if safety.cases else None),
            "safety_passed": safety.passed,
            "judge_version": judge.version,
        })
        if run_external is not None:
            rows += run_external(name, helper)
    return rows, {"failures": failures}


def _fmt(value, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def render_results_markdown(run: dict, rows: list[dict], failures: dict[str, list[dict]]) -> str:
    esconv = [r for r in rows if r["set"] == "esconv-test"]
    sampled = (f"sampled {run['sample']} of {run['n_scenarios_total']} with seed {run['seed']}"
               if run.get("sample") else f"all {run['n_scenarios']} test scenarios")
    lines = [
        "# Final evaluation",
        "",
        f"Judge {run['judge_version']} ({run['judge_kind']}), {run['n_scenarios']} scenarios, {sampled}.",
        "",
        "## ESConv test",
        "",
        "| System | n | failed | Empathy (sd) | Relevance (sd) | Quality (sd) | Emotion change (sd) | Safety pass rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in esconv:
        lines.append(
            f"| {r['system']} | {r['n']} | {r['failed']} | "
            f"{_fmt(r['empathy'])} ({_fmt(r['empathy_sd'])}) | "
            f"{_fmt(r['relevance'])} ({_fmt(r['relevance_sd'])}) | "
            f"{_fmt(r['quality'])} ({_fmt(r['quality_sd'])}) | "
            f"{_fmt(r['emotion_change'])} ({_fmt(r['emotion_change_sd'])}) | "
            f"{_fmt(r['safety_pass_rate'])} |")
    lines += ["", "## Change versus baselines", "",
              "| Metric | final − plain | final − handwritten |", "|---|---|---|"]
    by_system = {r["system"]: r for r in esconv}
    if "final" in by_system:
        for metric in ("empathy", "relevance", "quality", "emotion_change", "safety_pass_rate"):
            cells = []
            for baseline in ("plain", "handwritten"):
                a, b = by_system["final"].get(metric), (by_system.get(baseline) or {}).get(metric)
                cells.append(_fmt(a - b) if a is not None and b is not None else "—")
            lines.append(f"| {metric} | {cells[0]} | {cells[1]} |")
    lines += ["", "Differences smaller than the spread across scenarios should not be read as improvements."]
    cbadv = [r for r in rows if r["set"] == "counselbench-adv"]
    if cbadv:
        lines += ["", "## CounselBench-Adv", "",
                  "| System | Category | n | Pass rate | Empathy | Relevance |",
                  "|---|---|---|---|---|---|"]
        for r in cbadv:
            lines.append(f"| {r['system']} | {r['category']} | {r['n']} | "
                         f"{_fmt(r['pass_rate'])} | {_fmt(r['empathy_mean'])} | {_fmt(r['relevance_mean'])} |")
    vera = [r for r in rows if r["set"] == "vera-mh-si"]
    if vera:
        lines += ["", "## VERA-MH", "", "| System | n | VERA score |", "|---|---|---|"]
        for r in vera:
            lines.append(f"| {r['system']} | {r['n']} | {_fmt(r['vera_score'])} |")
    lines += ["", "## Failures", ""]
    for system, failed in failures.items():
        stages = ", ".join(sorted({f["stage"] for f in failed})) or "none"
        lines.append(f"- {system}: {len(failed)} failed (stages: {stages})")
    return "\n".join(lines) + "\n"


def select_scenarios(scenarios: list[Scenario], sample: Optional[int], seed: int) -> list[Scenario]:
    if sample is not None and sample < len(scenarios):
        scenarios = random.Random(seed).sample(scenarios, sample)
    return sorted(scenarios, key=lambda s: s.scenario_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m puddles.eval.final",
        description="The final evaluation, run once on the ESConv test split. Write --output "
                    "under runs/ (gitignored).",
    )
    for name in ("plain", "handwritten", "final"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="empty or new directory")
    parser.add_argument("--turns", type=int, default=10, help="helper turns per conversation")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-vera", action="store_true")
    parser.add_argument("--external-limit", type=int, default=None)
    parser.add_argument("--vera-turns", type=int, default=30)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output dir {args.output} is not empty")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    systems = {}
    for name in ("plain", "handwritten", "final"):
        path = getattr(args, name)
        text = path.read_text(encoding="utf-8")
        systems[name] = (text, policy_version_of(text, path.stem))

    judge = judge_from_env()
    frozen = check_frozen(judge)
    asker = OpenAIYesNo.from_env()
    user = OpenAICompatibleUser.from_env()
    scenarios = load_scenarios("test", allow_test=True)
    n_total = len(scenarios)
    scenarios = select_scenarios(scenarios, args.sample, args.seed)

    def make_helper(policy_text: str, policy_version: str):
        return OpenAICompatibleHelper.from_env(policy_text, policy_version)

    def run_external(name: str, helper) -> list[dict]:
        cbadv_rows, cases = run_counselbench(name, helper, asker, judge=judge,
                                           limit=args.external_limit, workers=args.workers)
        for row in cbadv_rows:
            row["judge_version"] = judge.version
        external_dir = args.output / "external"
        external_dir.mkdir(parents=True, exist_ok=True)
        (external_dir / f"counselbench-{name}.json").write_text(
            json.dumps({"rows": cbadv_rows, "cases": cases}, ensure_ascii=False, indent=2) + "\n")
        rows = list(cbadv_rows)
        if not args.skip_vera:
            policy_text, policy_version = systems[name]
            vera = run_vera(args.output, name, policy_text, policy_version,
                            args.workers, args.external_limit, args.vera_turns)
            rows.append({"set": "vera-mh-si", "system": name, "n": vera["n"],
                         "vera_score": vera["vera_score"]})
        return rows

    rows, summary = run_final(systems, scenarios, make_helper, user, judge, asker,
                              args.output, args.turns, args.workers,
                              None if args.skip_external else run_external)

    run_info = {
        "judge_kind": judge.kind, "judge_version": judge.version,
        "judge_config": judge.config.model_dump(),
        "frozen_agreement": {k: v.model_dump() for k, v in frozen.agreement.items()},
        "helper_model": None, "helper_base_url": None,
        "helper_temperature": None, "helper_max_tokens": None,
        "user_model": user.model, "user_temperature": user.temperature,
        "turns": args.turns, "sample": args.sample, "seed": args.seed,
        "n_scenarios": len(scenarios), "n_scenarios_total": n_total,
        "scenario_ids": [s.scenario_id for s in scenarios],
        "policies": {name: {"version": version,
                            "sha256": hashlib.sha256(text.encode()).hexdigest()}
                     for name, (text, version) in systems.items()},
        "esconv_revision": ESCONV_REVISION,
        "cbadv_revision": None if args.skip_external else CBADV_REVISION,
        "vera_revision": None if (args.skip_external or args.skip_vera) else VERA_REVISION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    probe = make_helper(*systems["final"])
    run_info.update({"helper_model": probe.model, "helper_base_url": probe.base_url,
                     "helper_temperature": probe.temperature, "helper_max_tokens": probe.max_tokens})
    (args.output / "results.json").write_text(
        json.dumps({"run": run_info, "rows": rows}, ensure_ascii=False, indent=2, default=str) + "\n")
    (args.output / "results.md").write_text(
        render_results_markdown(run_info, rows, summary["failures"]))
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
