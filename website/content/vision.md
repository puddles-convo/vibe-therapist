# Vision

<p class="draft"><b>Draft.</b> Written from the repository's README and design documents for the team to edit. It states intentions, not results; what was measured is under <a href="site:results.html">Results</a>.</p>

## The idea

Many people have no one to talk to at the moment they need it. Puddles is a small desk robot meant to be that first listener: it listens, reflects, asks questions, suggests coping techniques and points people to professional help. It does not diagnose or treat, and it says plainly that it is a robot, not a person and not a therapist.

The harder question is how such a helper gets better without getting less safe. A model that is tuned by hand improves only as fast as someone reads its transcripts, and a model that tunes itself can drift. Vibe Therapist is an attempt at the middle: a helper that improves its own behaviour, one small written change at a time, with every change tested before it is kept and every decision kept on record.

## Why safety comes first

People already bring their worst moments to chatbots, and it has gone wrong in public. In October 2024 a Florida mother sued a companion-chatbot company, alleging that its product presented itself to her 14-year-old son as a real person and a licensed psychotherapist, and that he had told it about his thoughts of suicide before he died ([Reuters](https://www.reuters.com/legal/mother-sues-ai-chatbot-company-characterai-google-sued-over-sons-suicide-2024-10-23/)). More families filed similar suits, and in January 2026 the companies agreed to settle several of them ([AP](https://apnews.com/article/ai-chatbot-lawsuits-character-google-fbca4e105b0adc5f3e5ea096851437de)). These are allegations in lawsuits, not findings of a court, but they name the failures a support helper must not have: claiming to be a person or a therapist, and carrying on when someone says they want to die.

That is why safety is built into the structure here and not left to the helper's reply. The policy's protected Safety section tells the helper not to claim to be a therapist or a person, and no edit by the loop may change that section. Every message goes to a keyword list in code and to a separate crisis checker, nothing is spoken until the checker has answered, and on a crisis the helper's draft is discarded for a fixed response with the 988 number. None of this has been validated as safe: the checker is itself a model, it has made recorded mistakes, and no clinician has reviewed it. [Limits](site:limits.html).

## Which model: any, and why an open-weight one

The loop and the live path do not depend on one model. The helper is whatever model answers at an OpenAI-compatible chat endpoint, chosen by two settings (`HELPER_URL` and `HELPER_MODEL`). Two have been run:

- **`gpt-6-luna`, a hosted model.** The 13-round run and every measured result on this site used it, and it worked well: under the judge that took no part in the loop, quality rose from 3.19 with the starting policy to 3.80 with the final one. [Results](site:results.html).
- **Qwen3.5-9B, an open-weight model that can be fine-tuned.** Served on Modal, it ran in the loop end to end in two smoke rounds. In the one that completed it started lower (quality 2.67 against 3.50 for `gpt-6-luna` on the same gate), and the proposed change was reverted: its gain was below the noise and its reply to a medication safety case gave dose advice. Whether Qwen fails that case under the starting policy too has not been checked. [The findings](results/FINDINGS-2026-11-15.md).

So a hosted model is the quickest way to a good helper today. Why also work on an open-weight one, tuned for this job? This part is the team's reasoning and has not been measured.

- **The behaviour can be owned.** A hosted frontier model changes when its provider updates it, and its training cannot be inspected. With open weights the model that was tested is the model that runs, and each change to it can go through the gate.
- **One narrow job.** A support helper does not need to write code or pass exams. It needs to listen well and to stop when it should. A small model can be trained on exactly that and tested on exactly that.
- **It can be trained, not only instructed.** Rules are all the loop can change in a hosted model. In an open one it can also change the weights, which is the next section.
- **Privacy and cost.** What people say to Puddles is sensitive. Open weights can be run on hardware the team controls, and a small model is cheaper to serve. This is not true yet: today the Qwen helper is served on Modal, and a trained adapter is served by Tinker by default.

## The fine-tuning

Rules can only do so much for a small model, so the loop can also train the helper's weights between rounds.

- **When.** After three rounds in which no rule change was kept, the loop starts a training run, up to three runs in all.
- **How.** The helper writes several replies to the same moment in a conversation and the judge scores them. DPO trains the model to prefer the best reply over the worst; GRPO, the fallback, weights every reply by how far it is above or below the group. Training is a LoRA adapter on Tinker.
- **The same kind of gate.** The policy stays fixed and only the weights change. A new adapter is kept only if quality rises by more than the noise, every safety case passes, and no fewer pass than before.

What exists today: the code, with offline tests, and one smoke run on Tinker (Qwen3.5-9B, 15 November) that took a single DPO step and a single GRPO step and got a reply from the saved weights. That run ranked replies by length, not with the judge. No full loop with the gate has trained an adapter, no safety data is mixed into training, and the loop's own judge is the reward, so the model could learn to please the judge. [The specification](docs/specs/weight-training.md).

## What we are building

**A helper whose behaviour is a readable file.** What Puddles does is set by a policy: a short list of numbered rules that anyone can read, question and diff. Improvement means a change to that file, not an opaque change to a model. [See the policy the run ended with](policies/training-2026-11-15/policy-v4.md).

**A loop that proposes, and a gate that decides.** Each round the loop reads its weakest conversations, researches how counsellors handle the failure, and proposes one change. A gate written in code, fixed before the run, keeps the change only if quality rises by more than the noise and every safety case still passes. Most proposals are reverted, and that is the design working. [How it learns](site:how-it-works.html).

**Safety that the loop cannot touch.** The safety rules are locked outside the search. In a live conversation nothing is spoken until a crisis checker has answered, and on a crisis the helper's draft is discarded for a fixed response that gives the 988 number. [The live conversation](site:how-it-works.html#the-live-conversation).

**Memory that belongs to the person.** The policy is the same for everyone. What is personal is a short set of notes, kept only with consent, bounded in number, and removed from active memory when the person asks. [The memory](docs/memory.md).

**A body.** The helper is meant to live in a StackChan robot on an M5Stack CoreS3, speaking and listening. The text path behind it runs today; speech is in a separate repository and is not connected through it yet. [The hand-over page for the robot](docs/robot-handover.md).

## What it is not

- Not therapy, not a medical device and not a crisis service.
- Not a replacement for a counsellor, a doctor or the people in someone's life. A good outcome is often that Puddles points a person towards one of them.
- Not validated. Everything measured so far used scripted or simulated people and model judges. See [Limits](site:limits.html).

## Where it stands

A hackathon research prototype, built on 14 and 15 November 2026. The training loop ran unattended for 13 rounds and produced a policy that scored higher than the one it started from, and the result held when measured again. The live text path ran end to end on real models. The honest list of what broke and what is still wrong is under [Results](site:results.html).

## What stands between this and real use

1. **Evidence with people.** Clinician review of the fixed crisis responses, the crisis labels and a sample of replies; a judge calibrated against human ratings; and, eventually, a study with real users. None of this has happened.
2. **The known crisis-path problems.** For example, a keyword match currently overrides the checker even on an explicit denial, and the crisis test set has no hard negatives. These are listed, not fixed.
3. **The voice.** Routing the robot's speech through the checked text path, and meeting the latency a spoken conversation needs.
4. **Operations and privacy.** A deployable service with per-client access, retention rules, and a forget request that reaches every stored copy, not only active memory.
5. **A model of our own.** The loop already runs with a Qwen helper served on Modal, and weight training between rounds exists but has only been tested offline.

Until the first of these is done, Puddles stays a research prototype.
