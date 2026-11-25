"""Invoke the pinned official VERA-MH pipeline and collect its score artifact."""
import csv
import json
import math
import os
import subprocess
from pathlib import Path

VERA_REVISION = "2ea4823cade663da3143e15dd7f63641c175502d"
SI_PERSONAS = 100  # data/SI/personas.tsv at VERA_REVISION


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def run_vera(output: Path, system: str, policy: str, policy_version: str,
             workers: int, limit: int | None, turns: int) -> dict:
    vera_root = Path(os.getenv("PUDDLES_VERA_ROOT", "/opt/vera"))
    user_model = _required("PUDDLES_VERA_USER_MODEL")
    judge_model = _required("PUDDLES_VERA_JUDGE_MODEL")
    generation_dir = (Path(output) / "vera" / system).resolve()
    config = {
        "target": "SI",
        "generation": {
            "chatbot": {"name": "puddles-helper", "repeats": 1},
            "user": [{"name": user_model, "repeats": 1}],
            "turns": turns, "output": str(generation_dir),
            "max_concurrent": workers, "persona_speaks_first": True,
        },
        "judging": {"models": [{"name": judge_model, "repeats": 1}],
                    "max_concurrent": workers, "per_judge": False},
        "scoring": {"personas": "data/SI/personas.tsv", "skip_risk_analysis": False},
    }
    command = [str(vera_root / ".venv/bin/python"), "-m", "puddles.eval.vera_bridge",
               "pipeline", "--config", "-"]
    if limit is not None:
        command.extend(["--sample", str(limit)])
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PUDDLES_VERA_POLICY": policy,
           "PUDDLES_VERA_POLICY_VERSION": policy_version, "MPLBACKEND": "Agg",
           "PYTHONPATH": os.pathsep.join((str(vera_root), str(repo_root)))}
    env.pop("VERA_RUN_CONFIG", None)
    generation_dir.mkdir(parents=True, exist_ok=True)
    with (generation_dir / "pipeline.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=vera_root, input=json.dumps(config),
                                   text=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"VERA-MH exited {completed.returncode}; see {generation_dir}/pipeline.log")
    files = list(generation_dir.rglob("scores.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one VERA-MH scores.json under {generation_dir}; found {len(files)}")
    score_path = files[0]
    results_path = score_path.parent.parent / "results.csv"
    with results_path.open(newline="", encoding="utf-8") as f:
        count = sum(1 for _ in csv.DictReader(f))
    expected = SI_PERSONAS if limit is None else min(limit, SI_PERSONAS)
    if count != expected:
        raise ValueError(f"VERA-MH scored {count} cases for {system}; expected {expected}")
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    return {"benchmark": "vera-mh-si", "system": system, "n": count,
            "vera_score": overall_score(scores),
            "details": str(score_path.relative_to(Path(output).resolve()))}


def overall_score(scores: dict) -> float:
    """Read the official overall field, allowing capitalization across release outputs."""
    def lookup(data, key):
        return next((value for name, value in data.items()
                     if name.casefold().replace(" ", "_") == key), None)

    score = lookup(scores, "overall_vera_score")
    if score is None:
        score = lookup(scores, "vera_mh_score")
    if score is None:
        score = lookup(scores, "overall_score")
    if score is None:
        overall = lookup(scores, "overall")
        score = lookup(overall, "vera_score") if isinstance(overall, dict) else overall
        if score is None and isinstance(overall, dict):
            score = lookup(overall, "vera_mh_score")
        if score is None and isinstance(overall, dict):
            score = lookup(overall, "score")
    if score is None:
        raise ValueError("VERA-MH scores.json has no recognized overall score field")
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("VERA-MH overall score falls outside the 0-100 scale")
    return score
