"""The hand-written pages of the site. Every number quoted here is checked against the file it comes from."""
# ruff: noqa: E501  (page text: the lines are sentences)
from __future__ import annotations

import html
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = "https://github.com/puddles-convo/puddles-stack"

LOGO = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="38" height="38" role="img" aria-label="Puddles">'
        '<ellipse cx="60" cy="109" rx="42" ry="7" fill="#cfeaf8"/>'
        '<path d="M60 2C60 2 49 14 49 21a11 11 0 0 0 22 0C71 14 60 2 60 2z" fill="#7ccdf0"/>'
        '<rect x="15" y="21" width="90" height="82" rx="19" fill="#fffdf8" stroke="#e2d6c3" stroke-width="2.5"/>'
        '<rect x="24" y="30" width="72" height="57" rx="12" fill="#2d2b35"/>'
        '<rect x="29" y="35" width="62" height="47" rx="8" fill="#a6dcf5"/>'
        '<circle cx="38" cy="66" r="4.5" fill="#f79a94" opacity=".65"/><circle cx="82" cy="66" r="4.5" fill="#f79a94" opacity=".65"/>'
        '<ellipse cx="46" cy="54" rx="4.6" ry="5.8" fill="#2d2b35"/><ellipse cx="74" cy="54" rx="4.6" ry="5.8" fill="#2d2b35"/>'
        '<path d="M53 65q7 6.5 14 0" stroke="#2d2b35" stroke-width="2.8" fill="none" stroke-linecap="round"/></svg>')

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Vibe Therapist</title>
<meta name="description" content="A support-conversation helper that improves its own behaviour. A research prototype: not therapy and not a crisis service.">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Vibe Therapist">
<meta property="og:title" content="{title} · Vibe Therapist">
<meta property="og:description" content="A support-conversation helper that improves its own behaviour. A research prototype: not therapy and not a crisis service.">
<meta property="og:url" content="{public}/{page}">
<meta property="og:image" content="{public}/docs/media/puddles-robot.png">
<meta name="twitter:card" content="summary">
<link rel="icon" type="image/svg+xml" href="{root}/static/favicon.svg">
<link rel="stylesheet" href="{root}/static/site.css">
</head>
<body>
<header class="top"><a class="brand" href="{root}/index.html">{logo}<span><b>Vibe Therapist</b><small>Puddles, a helper that improves its own behaviour</small></span></a>
<nav>{nav}<a href="{repo}">GitHub</a></nav></header>
<p class="notice"><b>A research prototype.</b> Not a medical device, not therapy and not a crisis service. Every person in these records is scripted or simulated. If you are in danger, call 911; in the US you can call or text 988.</p>
<main class="{main}">
{body}
</main>
<footer>Built at the SF AI Research Hackathon, 14 to 15 November 2026. This site is generated from the <a href="{repo}">repository</a>'s own documents and records by <code>python -m website.build</code>.</footer>
<script src="{root}/static/site.js"></script>
</body>
</html>
"""


def source_line(source: str, repo: str) -> str:
    return f'<p class="source">From <a href="{repo}/blob/main/{source}"><code>{html.escape(source)}</code></a> in the repository</p>'


def claim(text: str, source: str) -> str:
    """`text` as quoted on the site, after checking that the file it is quoted from still says it."""
    if text not in (ROOT / source).read_text(encoding="utf-8"):
        raise ValueError(f"{source} no longer says {text!r}; update website/pages.py")
    return html.escape(text)


def stat(value: str, label: str, source: str) -> str:
    return f'<div class="stat"><b>{claim(value, source)}</b><span>{label}</span></div>'


RUN, FIND, REPRO = "results/training-2026-11-15/README.md", "results/FINDINGS-2026-11-15.md", "results/reproduction-2026-11-15/README.md"
RUNS = (
    ("The 13-round training run", RUN,
     "The loop started from the hand-written starting policy and ran unattended until it stopped itself. Each round it read its weakest conversations, researched the failure, proposed one change and kept it only if the gate said it was better and still safe.",
     (("13 rounds", "3 changes kept, 7 reverted, 3 rounds with no candidate", "README.md"), ("+0.57", "final over starting policy, 40 scenarios, final judge (+0.36 to +0.78)", "README.md"),
      ("$6.67", "for 4,832 model calls", "README.md")), (("Replay it round by round", "site:demo.html"),)),
    ("Measured again: does the result hold?", REPRO,
     "The final and starting policies were measured again on ten fresh sets of the 20 held-out scenarios that no gatekeeper used.",
     (("+0.63", "final over starting policy (+0.53 to +0.73)", REPRO), ("10 of 10", "sets in which the final policy was higher", REPRO), ("$9.94", "for 600 conversations", REPRO)), ()),
    ("The proposer's toolbox, and our own Qwen helper", FIND,
     "A proposer that may also add skill cards and examples was run against the rules-only proposer under the same conditions. The loop was also run end to end with a Qwen3.5-9B helper served on Modal.",
     (("+0.02", "toolbox over rules-only (-0.15 to +0.19): no measurable difference", FIND), ("8 of 11", "toolbox rounds with a testable idea, against 5 of 11", FIND)),
     (("The run journal, step by step", "results/toolbox-2026-11-15/RUNLOG.md"),)),
    ("The live text path on real models", "results/inference-smoke/README.md",
     "One 11-turn conversation through the text chat: keyword check, crisis checker and helper in parallel, then the release step. It ends in a direct crisis statement, which got the fixed response with 988.",
     (("1435 ms", "median total per turn", "README.md"), ("3046 ms", "the one late turn, which got the holding line", "README.md")),
     (("Replay the conversation", "site:demo.html"), ("The note store on past.dev", "results/inference-smoke/pastdev.md"))),
    ("The same path over HTTP, for the robot", "results/http-replay/README.md",
     "The turn service the robot's brain will call, driven with the same messages, two people at once, and requests that should be refused.",
     (("62 turns", "over HTTP", "README.md"), ("1202 ms", "median service time per turn", "README.md")), (("The hand-over page for the robot team", "docs/robot-handover.md"),)),
    ("Crisis routing on labelled messages", "results/crisis-pilot/summary.md",
     "36 labelled messages, each sent twice: direct and indirect statements, danger from others, messages about someone else, retractions, idioms and everyday distress.",
     (("44 of the 46", "turns that expected the fixed crisis response got it", "README.md"), ("24 of 26", "turns that expected the helper's own reply got it", "README.md")),
     (("Try the messages in the replay", "site:demo.html"),)),
    ("Memory across sessions", "results/persona-pilot/README.md",
     "Three synthetic people, three sessions each, with notes and without. Notes were added, replaced and deleted on request, and loaded into the next session.",
     (("18 sessions", "completed and scored", "README.md"), ("12", "notes kept after session 3 with notes on", "README.md")), (("The full report", "results/persona-pilot/report.md"),)),
    ("A run from datasets, with no model", "results/esconv-dataset-run/README.md",
     "A stand-in made when a live training run failed: the same record and log formats, driven by rated ESConv conversations instead of model calls.", (), ()),
    ("Policy-v2 on a small local model", "results/inference-run/README.md",
     "Two scripted conversations through the live turn code with a CPU stand-in model, because the robot's own endpoint was not available.", (), ()),
)
VISION_NUMBERS = (("3.19", "README.md"), ("3.80", "README.md"), ("2.67", FIND), ("3.50", FIND), ("Qwen3.5-9B", FIND), ("gpt-6-luna", RUN))
EXPLORE = (("site:demo.html", "Replay the records", "Recorded conversations, the 13 rounds step by step, and policy-v1 against policy-v4."),
           ("site:how-it-works.html", "How it works", "The training loop and the live conversation path, with the animated figures."),
           ("site:results.html", "Results", "What was measured, what broke, and what is still wrong."),
           ("site:runs.html", "The runs", "Every recorded run with its numbers, its cost and its report."),
           ("site:design.html", "The design in depth", "Agents, models, memory, the gate and how a change reaches the policy."),
           ("site:run-it.html", "Run it yourself", "Install it, run the tests and one text conversation, all offline with no key."),
           ("site:vision.html", "Vision", "What Puddles is for, what it is not, and what stands between it and real use."))
DOC_GROUPS = (("The agents", ("docs/agents.md", "docs/agent-prompts.md", "docs/agents/")), ("Memory", ("docs/memory.md", "docs/memory-summary.md", "docs/research/memory/")),
              ("Specifications", ("docs/specs/",)), ("The robot", ("docs/robot-handover.md",)), ("The build plan", ("docs/plan/",)),
              ("Research notes", ("docs/research/reusable", "docs/research/self-")), ("Figures", ("figures/",)), ("Policies", ("policies/",)))


def charts(training: dict) -> str:
    """The 13 rounds at the gate, and the three policies as measured after the run. Drawn from the run's records."""
    rounds = training["rounds"]
    gated = [r for r in rounds if r["gate"]]
    values = [r["gate"][k]["quality"] for r in gated for k in ("before", "after")]
    lo, hi = math.floor(min(values) * 4) / 4, math.ceil(max(values) * 4) / 4
    W, H, L, R, T, B = 720, 250, 40, 12, 12, 30

    def x(n: int) -> float:
        return L + (W - L - R) * (n - .5) / len(rounds)

    def y(q: float) -> float:
        return T + (H - T - B) * (hi - q) / (hi - lo)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Quality of each round\'s candidate against the policy in force">']
    for i in range(round((hi - lo) * 4) + 1):
        q = lo + i / 4
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(q):.1f}" y2="{y(q):.1f}" stroke="#eadfd0"/><text x="{L - 6}" y="{y(q) + 4:.1f}" text-anchor="end">{q:.2f}</text>')
    for r in rounds:
        n, gate = r["round"], r["gate"]
        out.append(f'<text x="{x(n):.1f}" y="{H - 10}" text-anchor="middle">{n}</text>')
        if not gate:
            out.append(f'<text x="{x(n):.1f}" y="{y(lo) - 8:.1f}" text-anchor="middle" fill="#b8a99d">none</text>')
            continue
        before, after, kept = gate["before"]["quality"], gate["after"]["quality"], gate["decision"] == "kept"
        colour = "#3f8f5f" if kept else "#b8a99d"
        out.append(f'<g><title>Round {n}: {html.escape(gate["reason"])}</title>'
                   f'<line x1="{x(n) - 13:.1f}" x2="{x(n) + 13:.1f}" y1="{y(before):.1f}" y2="{y(before):.1f}" stroke="#3b2f2a" stroke-width="3" stroke-linecap="round"/>'
                   f'<line x1="{x(n):.1f}" x2="{x(n):.1f}" y1="{y(before):.1f}" y2="{y(after):.1f}" stroke="{colour}" stroke-width="2"/>'
                   f'<circle cx="{x(n):.1f}" cy="{y(after):.1f}" r="6.5" fill="{colour if kept else "#fffdf9"}" stroke="{colour}" stroke-width="2.5"/></g>')
    out.append("</svg>")
    final = next(m for m in training["measurements"] if m["scenarios"] == "validation" and m["judge"] == "final")
    names = {"start": "policy-v1, the starting policy", "control": "Safety rules only, no conversation rules", "final": "policy-v4, the final policy"}
    bars = "".join(f'<div class="bar{" on" if p["name"] == "final" else ""}"><span>{names[p["name"]]}</span><i style="width:{(p["quality"] - 1) / 4:.1%}"></i><b>{p["quality"]:.2f}</b></div>'
                   for p in final["policies"])
    return f"""<section class="charts"><div><h2>The 13 rounds at the gate</h2>
<p>Each round tests one proposed change against the policy in force (the dark bar), on the same validation scenarios. A change is kept only if it scores higher by more than the noise and every safety case still passes.</p>
<div class="figure">{"".join(out)}</div>
<p class="key"><span class="kept"></span> kept ({training["stats"]["kept"]}) <span class="rev"></span> reverted ({training["stats"]["reverted"]}) <span class="none">none</span> no change proposed ({training["stats"]["no_candidate"]}). Quality, 1 to 5, under the loop's judge. <a href="site:demo.html">Replay each round</a></p></div>
<div><h2>Measured after the run</h2>
<p>Quality under a judge that took no part in the loop, on {final["n"]} validation scenarios, {final["sets"]} sets each.</p>
<div class="bars">{bars}</div>
<p class="key">The people are simulated and the judge is a model, not calibrated against human ratings. <a href="site:results.html">All results</a></p></div></section>"""


def home(data: dict) -> str:
    t = data["training"]["stats"]
    assert (t["rounds"], t["kept"], t["reverted"]) == (13, 3, 7), "the training record changed; update the home page"
    tools = re.findall(r'<a href="([^"]+)"><img alt="([^":]+): ([^"]+)"', data["readme"].split("How it learns")[0])
    tools = [x for x in tools if x[1] not in ("python", "tests", "helper")]
    assert len(tools) == 11, "the README's badges changed; check the home page's Built with"
    tools = [("docs/specs/provider-settings.md", "helper, the measured runs", "gpt-6-luna"),
             ("docs/agents/t2-i3-helper.md", "helper, open weights", "Qwen3.5-9B")] + tools
    built = "".join(f'<{"span" if h.startswith(PRIVATE) else "a"} href="{h}"><small>{html.escape(html.unescape(role))}</small><b>{html.escape(html.unescape(tool))}</b></{"span" if h.startswith(PRIVATE) else "a"}>'
                    for h, role, tool in tools).replace(f'<span href="{PRIVATE}">', '<span title="In a private repository">')
    cards = "".join(f'<a class="card" href="{h}"><b>{t_}</b><span>{d}</span></a>' for h, t_, d in EXPLORE)
    return f"""<section class="hero"><div><h1>A support-conversation helper that <em>improves its own behaviour</em>.</h1>
<p>{claim("An offline loop reads the helper's weakest conversations, researches how counsellors handle the failure, proposes one change, and keeps it only if a gate fixed before the run says it is better and still safe.", "README.md")}</p>
<p>{claim("The helper listens, reflects, asks questions, suggests coping techniques and points people to professional help. It does not diagnose or treat.", "README.md")}</p>
<p class="actions"><a class="btn primary" href="site:demo.html">Watch the replay</a><a class="btn" href="site:how-it-works.html">How it works</a><a class="btn" href="{data['repo']}">Read the code</a></p></div>
<img src="docs/media/puddles-robot.png" alt="The Vibe Therapist robot: a StackChan on an M5Stack CoreS3" width="300" height="300"></section>
<section class="stats">{stat("13 rounds", "run unattended: 3 changes kept, 7 reverted, 3 with no candidate", "README.md")}
{stat("+0.57", "quality on a 1 to 5 scale, final over starting policy, under a judge that took no part in the loop (95% interval +0.36 to +0.78)", "README.md")}
{stat("+0.63", "when measured again on ten fresh sets; higher in 10 of 10", REPRO)}
{stat("19 safety cases", "all passed by the final policy, whose safety rules the loop may not edit", "README.md")}
{stat("1202 ms", "median live text turn over HTTP, before speech", "README.md")}
{stat("44 of the 46", "crisis turns got the fixed response with 988", "README.md")}</section>
{charts(data["training"])}
<section class="three"><div><h2>It learns one change at a time</h2><p>Seven steps turn the weakest conversations into at most one change to the policy. A gate written in code, with no model, tests the change on validation conversations and safety cases, and restores the previous policy if it is not better.</p><p><a href="site:how-it-works.html">The training loop</a></p></div>
<div><h2>Nothing is spoken until the crisis check has answered</h2><p>Each message goes to a keyword check, a crisis checker and the helper at once. On a crisis the helper's draft is discarded and a fixed response with 988 is spoken instead.</p><p><a href="site:how-it-works.html#the-live-conversation">The live conversation</a></p></div>
<div><h2>It remembers only with consent</h2><p>The policy is the same for everyone. What is personal is a short set of notes, kept only if the person agreed, loaded at the start of the next session and removed from active memory on request.</p><p><a href="site:results.html#memory-across-sessions">Memory across sessions</a></p></div></section>
<section class="three"><div><h2>Why safety comes first</h2><p>People already bring their worst moments to chatbots, and lawsuits allege it has ended in harm. So here the safety rules are closed to the loop, and a crisis gets a fixed response, not the model's draft. None of it is validated yet. <a href="site:vision.html#why-safety-comes-first">Read the context</a>.</p></div>
<div><h2>Any model, and an open-weight one to tune</h2><p>The helper can be any model behind an OpenAI-compatible endpoint. <code>gpt-6-luna</code> gave the measured results and worked well. Qwen3.5-9B, open-weight and tunable, runs in the loop but starts lower. <a href="site:vision.html#which-model-any-and-why-an-open-weight-one">Read the reasoning</a>.</p></div>
<div><h2>Fine-tuning, behind the same kind of gate</h2><p>When rule changes stop helping, the loop can train the helper's weights with DPO or GRPO and keep the adapter only if it is better and still safe. Built and tested offline; no full run yet. <a href="docs/specs/weight-training.md">Read the specification</a>.</p></div></section>
<section><h2>Explore</h2><div class="cards">{cards}</div></section>
<section><h2>Built with</h2><p class="lead small">The helper can be any model behind an OpenAI-compatible chat endpoint, chosen by two settings. Two were run: {claim("gpt-6-luna", "docs/specs/provider-settings.md")}, which every measured result here comes from, and {claim("Qwen3.5-9B", FIND)}, an open-weight model that can be fine-tuned.</p><div class="tools">{built}</div></section>"""


def runs(data: dict) -> str:
    out = ["<h1>The runs</h1><p class=\"lead\">Every run below was recorded on 14 to 15 November 2026 and is committed with its report. The people in all of them are scripted or simulated.</p>"]
    for title, page, text, facts, more in RUNS:
        stats = "".join(stat(v, label, src) for v, label, src in facts)
        links = "".join(f'<a href="{h}">{label}</a>' for label, h in ((("Read the report", page),) + more))
        out.append(f'<article class="run"><h2>{title}</h2><p>{text}</p>{f"<div class=stats>{stats}</div>" if stats else ""}<p class="more">{links}</p></article>')
    return "".join(out)


def docs(data: dict, section: Callable, render: Callable) -> str:
    out = ["<h1>Documentation</h1><p class=\"lead\">The repository's design documents, rendered as they are. Where a page and the code differ, the code is right.</p>"]
    for name, prefixes in DOC_GROUPS:
        items = [(s, t) for s, t in data["titles"].items() if s.startswith(prefixes)]
        out.append(f"<h2>{name}</h2><ul class=\"index\">" + "".join(f'<li><a href="{s}">{html.escape(t)}</a> <code>{s}</code></li>' for s, t in items) + "</ul>")
    return "".join(out) + render(section(data["readme"], "Read more"))[0]


def written(data: dict[str, Any], section: Callable, render: Callable) -> Iterator[tuple[str, str, str, str, bool]]:
    readme = data["readme"]
    yield "index.html", "Home", "SITE", home(data), True
    text = (ROOT / "website" / "content" / "vision.md").read_text(encoding="utf-8")
    for number, source in VISION_NUMBERS:  # the draft quotes these; each must still be in the file it comes from
        assert number in text, f"website/content/vision.md no longer quotes {number}; update VISION_NUMBERS"
        claim(number, source)
    vision = render(text)[0]
    yield "vision.html", "Vision", "SITE", vision, False
    yield "runs.html", "The runs", "SITE", runs(data), True
    yield "docs.html", "Documentation", "README.md", docs(data, section, render), False
    limits = "".join(section(readme, h, level) for h, level in (("Limits", 2), ("Limits", 3), ("Known problems, not fixed", 3)))
    yield "limits.html", "Limits", "README.md", "<h1>Limits and known problems</h1>" + render(limits.replace("## Limits\n", "## What the project cannot claim\n", 1).replace("### Limits\n", "### Limits of the recorded runs\n"))[0], False
    demo = ('<h1>The replay</h1><p class="lead">Recorded conversations with what happened behind each reply, the 13 rounds step by step, and policy-v1 against policy-v4. '
            'Nothing here calls a model. <a href="demo/index.html">Open it full screen</a></p><iframe class="demo" src="demo/index.html" title="Replay of the recorded runs"></iframe>')
    yield "demo.html", "Replay", "SITE", demo, True
