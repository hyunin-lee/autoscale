# autoscale

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `metric.md` — monitoring metric notebook, if it already exists. It records custom metrics added to `train.py`, their definitions, and what was learned from them.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
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
   <!-- One short entry per experiment: commit, val_bpb, status, what the metrics showed, what to try next -->

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

In addition to optimizing the final `val_bpb`, you should actively improve your ability to understand training dynamics by creating custom monitoring metrics inside `train.py`. The purpose of these metrics is not to replace `val_bpb`; the goal is still to get the lowest final `val_bpb`. Monitoring metrics are diagnostic signals that help you choose better future edits to `train.py`.

You may edit the existing training-loop print line, add lightweight metric computations, and print additional compact diagnostic lines to `run.log`. You can use existing variables such as `loss`, `train_loss_f`, `debiased_smooth_loss`, `lrm`, `dt`, `tok_per_sec`, `mfu`, `progress`, `step`, `epoch`, optimizer param groups, model parameters, gradients, activations, logits, or any other values available in `train.py`. You may also define derived metrics, e.g. ratios, slopes, moving averages, normalized quantities, or products such as `val1**2 / (val2 + eps) * val3`.

When adding metrics:

- Keep them cheap. Do not materially reduce the number of training steps within the fixed 5-minute budget unless the metric is clearly worth the cost.
- Prefer scalar metrics, moving averages, or periodic diagnostics over huge logs.
- Use `torch.no_grad()` and `.detach()` where appropriate.
- Avoid changing the validation/evaluation harness. The final decision metric remains `evaluate_bpb` from `prepare.py`.
- Keep the final summary lines unchanged, especially `val_bpb:` and `peak_vram_mb:`, because the experiment loop parses them.
- Make log output easy to search. Prefer a stable prefix such as `mon |` for custom monitoring lines, or append compact fields to the existing progress print.
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

After each experiment, read `run.log` and `metric.md`, then append a short observation to `metric.md`: which metrics were informative, which were noisy/useless, and what next code or hyperparameter change they suggest. You may remove or simplify useless metrics in later iterations, but record that decision in `metric.md`.

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

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Read `metric.md` and the previous `results.tsv` entries. Decide whether the next iteration should change training behavior, add/remove/refine monitoring metrics, or do both.
3. Tune `train.py` with an experimental idea by directly hacking the code. This can include adding custom monitoring metrics and print output, as long as the final `val_bpb` optimization goal and fixed evaluation remain unchanged.
4. If you add, remove, or redefine monitoring metrics, update `metric.md` before running so the definitions match the code. Do not commit `metric.md`.
5. git commit the `train.py` experiment
6. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
7. Read out the final results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
8. Read the custom monitoring output, e.g. `grep "mon |" run.log` if you used the recommended prefix, plus any relevant nearby training logs.
9. If the final-result grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
10. Record the result in `results.tsv` (NOTE: do not commit `results.tsv`; leave it untracked by git)
11. Append a short post-run note to `metric.md`: final `val_bpb`, status, important metric observations, whether the metrics helped, and what they suggest for the next iteration. Do not commit `metric.md`.
12. If `val_bpb` improved (lower), you "advance" the branch, keeping the git commit
13. If `val_bpb` is equal or worse, you git reset back to where you started. `metric.md` and `results.tsv` should remain as untracked research notes so you still remember what happened and can use the monitoring observations next.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
