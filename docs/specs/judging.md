# Build spec: the judge in the graph

Covers item 1 of [docs/plan/prompts/stages-2-3-4.md](../plan/prompts/stages-2-3-4.md): the round's `judge` node runs the judge in `puddles/eval/judge.py` (#33, #38) through an adapter. It replaces lane A of stage 2, which built a second judge.

This file is the contract. Where it and the stage file disagree, this file is right.

## Ground rules

- Nothing under `puddles/eval/` or `safety/` is created or edited, and no `tests/test_eval_*.py`. His judge is called through `score(record)` and `version`, and built with `puddles.eval.judge.judge_from_env()`. No model, prompt or temperature is chosen here.
- `schemas.py` is not changed. No existing test file is changed.
- No network and no API keys in any test.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 python -m pytest -q
```

## Layout

```
puddles/judging.py                    new: the adapter
puddles/graph.py                      changed: RoundDeps.judge_panel, the judge node, one metric
puddles/round.py                      changed: RoundResult.unscored; finish_round counts it in the reason
scripts/run_graph_round.py            changed: builds the judge
tests/test_judging.py                 new: the implementer's tests
tests/acceptance/test_acc_judging.py  new: written from this file alone, by someone else
```

## 1. `puddles/judging.py`

```python
CRITIC_CONVERSATIONS = 10

class Judge(Protocol):
    version: str
    def score(self, record: ConversationRecord) -> Any: ...
    # the returned object has `.score`, a ScoreRecord, and `.usage`, a list of objects
    # each with `.model`, `.prompt_tokens` and `.completion_tokens` (his JudgeResult)

@dataclass
class JudgeRun:
    scores: list[ScoreRecord]            # in the order of the records, the unscored ones left out
    unscored: list[str]                  # conversation IDs, in the order of the records
    reasons: dict[str, str]              # conversation ID -> why it was left unscored

def score_conversations(judge: Judge, records: list[ConversationRecord]) -> JudgeRun
def record_usage(usage: Iterable[Any]) -> None
def to_scored(record: ConversationRecord, score: ScoreRecord) -> ScoredConversation
def scored_conversations(records: list[ConversationRecord], scores: list[ScoreRecord]) -> list[ScoredConversation]
def select_for_critic(scored: list[ScoredConversation], n: int = CRITIC_CONVERSATIONS) -> list[ScoredConversation]
def train_scores_path(root: Path, round_: int) -> Path          # <root>/scores/round-NN-train.jsonl
def save_scores(scores: list[ScoreRecord], path: Path) -> None
def judge_models(judge: Any) -> list[str]

class MeteredJudge:
    def __init__(self, judge: Judge) -> None
    version: str                          # the wrapped judge's
    def score(self, record: ConversationRecord) -> Any
```

- `score_conversations` calls `judge.score(record)` once for each record, in order. His `score` already asks each judge model again once when an answer fails its checks, and raises when the second answer fails too; so **any exception from `score` is a conversation his judge failed on twice: it is left unscored, with the exception's type and text as the reason, and the rest are still scored.** A result whose `score.conversation_id` is not the record's ID is unscored too. After each successful call it passes the result's `usage` to `record_usage`.
- `record_usage` writes each entry to the usage log through the stage 1 functions, under the meter and agent that are current (`puddles.usage.current()`): one call per entry with the entry's model and its two token counts, as counts the model reported, not estimates. With no current meter it does nothing. It reads the entries; it does not change his client. A model that has no price is handled the way the stage 1 meter handles it; `record_usage` does not catch that.
- **What is and is not counted.** The usage of a result that came back is recorded even when the result is then left unscored for a wrong conversation ID: the tokens were spent. A conversation whose `score` raised has no usage to record, because his exception carries none; those tokens are missing from the usage log, and the cost of a round with unscored conversations is understated by that much. This is a stated limit, not something to work around in his client. An exception from `record_usage` itself (a model with no price) is not caught by `score_conversations`: it propagates, because a round whose cost cannot be counted must not go on quietly.
- `to_scored`: `conversation_id`, `empathy`, `relevance` and `note` from the score, `turns` from the record.
- `scored_conversations` pairs each score with its record by conversation ID, in the order of the scores. A score with no record raises `ValueError`.
- `select_for_critic` returns the `n` lowest by `(empathy + relevance) / 2`, lowest first, ties broken by conversation ID in ascending order; all of them, in that order, when there are fewer than `n`. `n` below 1 raises `ValueError`.
- `save_scores` writes one `ScoreRecord` per line and creates the folder. An empty list writes an empty file. A file that is already there is replaced.
- `judge_models` returns the model names a judge will call, for the price check: for his panel the three judges and Astra (`judge.config.judges` and `judge.config.astra`), for his fallback judge its one model (`judge.config.model`), and an empty list for anything that has no `config`.
- `MeteredJudge.score` calls the wrapped judge's `score`, passes the result's `usage` to `record_usage`, and returns the result unchanged. An exception passes through. It is what the gate is given, so the gate's judge calls are counted too.

## 2. `puddles/graph.py`

- `RoundDeps` gets `judge_panel: Optional[Judge] = None` and `critic_conversations: int = CRITIC_CONVERSATIONS`. `RoundState` gets `unscored: list[str]`.
- With `judge_panel` `None`, the `judge` node behaves exactly as before: it calls `deps.judge`.
- With `judge_panel` set, `deps.judge` is not called. Inside the node's `metering(meter, "judge")` block it calls `score_conversations(deps.judge_panel, records)`, saves the scores with `save_scores(run.scores, train_scores_path(root, round))`, and sets `scored = select_for_critic(scored_conversations(records, run.scores), deps.critic_conversations)` and `unscored = run.unscored`. The rest of the node is as before: when `scored` is empty the round goes to `finish` with the reasons graph.md gives.
- The round goes on whatever was left unscored. The unscored IDs are put on the `RoundResult` (`result.unscored`) by whichever node makes or holds it, before `finish` runs.
- `run_round` adds `judge_models(deps.judge_panel)` to the models whose prices are checked before the round starts, and logs the metric `conversations_unscored`.

## 3. `puddles/round.py`

- `RoundResult` gets `unscored: list[str]`, empty by default.
- When it is not empty, `finish_round` appends one sentence to the round event's reason: `The judge left N conversation(s) unscored: <ids>.` with the IDs separated by a comma and a space, and `conversation` or `conversations` as the count needs. Its signature does not change.

## 4. `scripts/run_graph_round.py`

- Without `--fake`, the judge is `puddles.eval.judge.judge_from_env()`, passed as `judge_panel`. Its `ValueError` for a missing variable is missing configuration: one line on standard error, exit code 2, nothing in the log.
- With `--fake`, `judge_panel` is a scripted judge defined in the script: it reads no conversation, gives fixed scores, reports no usage, and its `version` is `judge-fake`. So a fake run goes through the same adapter and writes `scores/round-NN-train.jsonl`. If the script already has a scripted judge for `--gate real`, it is the same one.
- With `--fake --gate real`, the gate's judge is `MeteredJudge` around that same scripted judge, so the gate's scoring goes through the same judge as the round and is counted under the agent `gatekeeper`.
- The script reaches the factory as `puddles.eval.judge.judge_from_env`, looked up when `main` runs, so a test can replace it there.
- `tests/acceptance/test_acc_graph_script.py` passes as it is, with one exception. `test_without_fake_constructs_documented_dependencies` runs real mode with every constructor replaced and no judge configuration; real mode now needs a judge, so its author adds `puddles.eval.judge.judge_from_env` to the constructors it replaces and checks the judge reaches `RoundDeps.judge_panel`. Nothing else in that file changes.

## 5. Tests

`tests/test_judging.py`, with a fake judge:

1. A record his judge scores becomes a `ScoredConversation` with the score's numbers and the record's turns.
2. A judge that raises on one conversation leaves it unscored and counted, and the others are scored; the round reaches `finish`, and the event's reason names the unscored conversation.
3. `select_for_critic` returns the lowest `n` in the fixed order, and all of them when there are fewer.
4. The scores file has one valid `ScoreRecord` per line.
5. Token counts the judge reports reach `<root>/log/usage.jsonl` under the agent `judge`, and through `MeteredJudge` under the agent that is current.
6. One test builds his real `JudgePanel` with three fake `JudgeLLM`s and a fake Astra (`puddles.eval.judge.JudgePanel(config, judges, astra)`) and runs it through `score_conversations`, to show the adapter fits his code as it is.

The acceptance tests are written from this file alone, by someone who has not seen the code.
