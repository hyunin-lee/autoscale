# autoscale

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoscale/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoscale/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `metric.md` — monitoring metric notebook, if it already exists. It records custom metrics added to `train.py`, their definitions, and what was learned from them.
4. **Verify data exists**: Check that `~/.cache/autoscale/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Initialize metric.md**: If `metric.md` does not exist, create it with sections for metric definitions, per-run observations, and candidate next metrics. Leave it untracked by git, like `results.tsv`, so it survives discarded experiment commits.
   Use this skeleton:

   ```markdown
   # Monitoring metrics

   Persistent notebook for custom monitoring metrics added to `train.py`.
   Goal of every metric: help find the lowest `val_bpb`. Untracked by git.

   ## Active metrics
   <!-- For each metric: Name | Definition (formula/code) | Location in train.py | Why it helps reach lower val_bpb -->

   ## Per-run observations
   <!-- One short entry per experiment: commit, val_bpb, status, metrics tried this iteration, what they showed, what to try next -->

   ## Candidate / retired metrics
   <!-- Ideas not yet tried, and metrics removed for being noisy/useless (with the reason) -->
   ```
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only code file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, monitoring metric computations/prints, etc.
- Modify `metric.md` — this is the persistent monitoring notebook. Use it to define custom monitoring metrics, explain why they were added, record what they showed in `run.log`, and decide what metric or training-code change to try next. Do not commit `metric.md`.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Custom monitoring metrics

In addition to optimizing the final `val_bpb`, you should actively improve your ability to understand training dynamics by creating custom monitoring metrics inside `train.py`. The purpose of these metrics is not to replace `val_bpb`; the goal is still to get the lowest final `val_bpb`. Add or keep a monitoring metric only when it gives a useful perspective on the learning trajectory that can guide better future edits to `train.py` and help lower final `val_bpb`.

**Actively evolve your metric set — do not set it once and freeze it.** The metric set is a living dashboard that must track whatever question the current experiments are actually about, not a fixed pair of numbers you define on iteration one and then watch passively forever. Early iterations might care about loss slope/noise; once those are understood, the interesting question moves on (matrix-LR overshoot, embedding-vs-matrix LR balance, depth/width capacity, gradient/update scale, logit saturation, throughput), and your metrics should move with it. A metric whose question has already been answered, or that has been flat/uninformative for several runs, is dead weight — retire it and replace it with one that probes the *current* open question. Keep the active set small and focused (a handful of metrics at a time) so that adding a new metric usually means retiring an old one: the set rotates to follow the bottleneck rather than growing without bound. If your last few runs all show the same metrics doing nothing to change your decisions, that is a signal your monitoring has stagnated and you must design new metrics.

You may edit the existing training-loop print line, add lightweight metric computations, and print additional compact diagnostic lines to `run.log`. You can use existing variables such as `loss`, `train_loss_f`, `debiased_smooth_loss`, `lrm`, `dt`, `tok_per_sec`, `mfu`, `progress`, `step`, `epoch`, optimizer param groups, model parameters, gradients, activations, logits, or any other values available in `train.py`. You may also define derived metrics, e.g. ratios, slopes, moving averages, normalized quantities, or products such as `val1**2 / (val2 + eps) * val3`.

When adding metrics:

- Keep them cheap. Do not materially reduce the number of training steps within the fixed 5-minute budget unless the metric is clearly worth the cost.
- Prefer scalar metrics, moving averages, or periodic diagnostics over huge logs.
- Use `torch.no_grad()` and `.detach()` where appropriate.
- Avoid changing the validation/evaluation harness. The final decision metric remains `evaluate_bpb` from `prepare.py`.
- Keep the final summary lines unchanged, especially `val_bpb:` and `peak_vram_mb:`, because the experiment loop parses them.
- Make log output easy to search. Prefer appending cheap scalar metrics to the right side of the existing progress print after `remaining: {remaining:.0f}s`, e.g. `| grad_norm: ...`, or use a stable prefix such as `mon |` for separate custom monitoring lines.
- If a metric is expensive, compute it every N steps instead of every step.

Examples of useful metric families:

- **Loss dynamics**: recent loss slope, loss curvature, loss noise, EMA-vs-current loss gap.
- **Optimization dynamics**: gradient norm, parameter norm, update-to-weight ratio, per-param-group gradient scale, effective LR.
- **Stability diagnostics**: NaN/Inf counts, max absolute logits, logit softcap saturation fraction, unusually large activations.
- **Architecture diagnostics**: per-layer residual scalar values, `x0_lambdas`, attention/value-embedding gate behavior.
- **Efficiency diagnostics**: tokens/sec trend, MFU trend, time per step, memory use, compile/startup overhead.

For every new metric you create, immediately update `metric.md` with:

1. **Name** — short stable name used in `run.log`.
2. **Definition** — exact formula or code-level description.
3. **Location** — where it is computed/printed in `train.py`.
4. **Purpose** — why this metric could help find lower `val_bpb`.
5. **Cost/risk** — expected overhead and any risk of perturbing training.
6. **Interpretation plan** — what high/low/increasing/decreasing values suggest you might try next.

After each experiment, read `run.log` and `metric.md`, then append one short per-experiment observation to `metric.md`. Include `metrics tried:` listing every metric added, retired, or redefined this iteration, and — for any metric you deliberately kept unchanged — a one-line reason it is still actively informing your decisions. Do NOT use `metrics tried: none` as a routine default: keeping the entire metric set identical is an exception that needs justification, and you may not carry the exact same active metric set for more than two consecutive iterations without adding, retiring, or redefining at least one metric. Then note which metrics were informative, which were noisy/useless, and what next code or hyperparameter change they suggest for lowering final `val_bpb`. When a metric has served its purpose or proven uninformative, retire it — move it to the "Candidate / retired metrics" section with the reason — and bring in a new one aimed at the current open question.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoscale/mar5` or `autoscale/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Analyze the previous run: read the last `run.log` (final `val_bpb` plus your custom monitoring output) together with `metric.md` and the `results.tsv` history. Ask, per active metric: did it actually inform a decision this time, or was it flat / redundant / a question that is now answered?
3. Curate your monitoring metrics — this is a required active step, not optional. Based on that analysis, retire metrics that are uninformative or already answered, and add at least one new metric probing the current open question / bottleneck. Keep the active set small so new metrics usually replace old ones. You may keep the set unchanged only with an explicit reason it is still the most informative possible, and never for more than two iterations in a row (see "Custom monitoring metrics"). Note: a discard in the final step reverts in-code metric edits along with `train.py`, so treat `metric.md` as the durable source of truth and re-establish your curated metric set on top of whatever commit you are building from.
4. Decide and apply the next `train.py` experimental change — the hyperparameter / architecture / optimizer / batch / model-size idea — by directly modifying the code. Every non-baseline iteration must include a real change to `train.py` beyond the monitoring edits.
5. Update `metric.md` to match the code before running: record definitions for any added or redefined metrics, and move retired ones to the "Candidate / retired metrics" section with the reason. Do not commit `metric.md`.
6. git commit the `train.py` experiment (the monitoring edits ride along in the same commit; `metric.md` stays untracked).
7. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
8. Read out the final results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
9. Read the custom monitoring output, e.g. `grep "mon |" run.log` if you used the recommended prefix, plus any relevant nearby training logs.
10. If the final-result grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
11. Record the result in `results.tsv` (NOTE: do not commit `results.tsv`; leave it untracked by git).
12. Append a short post-run note to `metric.md`: final `val_bpb`, status, `metrics tried:` for this exact iteration (metrics added / retired / redefined, plus a one-line reason for any deliberately kept unchanged), important metric observations, whether the metrics helped understand the learning trajectory, and what they suggest for the next `train.py` edit toward lower final `val_bpb`. Do not replace this with a bulk sweep log; `metric.md` should have one note per experiment. Do not commit `metric.md`.
13. If `val_bpb` improved (lower), you "advance" the branch, keeping the git commit.
14. If `val_bpb` is equal or worse, you git reset back to where you started. `metric.md` and `results.tsv` should remain as untracked research notes so you still remember what happened and can use the monitoring observations next.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
