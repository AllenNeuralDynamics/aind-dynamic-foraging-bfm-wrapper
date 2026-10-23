# Post-Training Evaluation Guide

> **Living document.** This is the single reference for what the post-training
> *evaluation* codebase includes and how to use it. **When you add or change an
> evaluation feature, update this file** (and add a dated entry to the
> [Changelog](#changelog) at the bottom). For the *training* pipeline see
> [TRAINING.md](TRAINING.md); for per-config-key docs see the inline comments in
> `configs/*.yaml`.

Scope: everything under `code/` that **analyzes an already-trained model** — i.e.
runs *after* `run_capsule.py` has produced a saved run directory. This is designed
to be **independent of training**: importing the analysis code or running an
analysis never imports `model_trainers` (with two documented exceptions, see
[§7](#7-training-adjacent-exceptions)).

---

## 1. Overview

A post-training analysis takes a **saved trained-run directory** and produces
metrics, dataframes, and plots — without retraining:

```
saved run dir ──► resolve_model_run(model_dir, split, checkpoint_policy)
  (inputs.yaml      │   → ResolvedModelRun (params path, config, checkpoint, metadata)
   + outputs/)      │
                    └─► analysis fn  ──► metrics / dataframes / figures / summary JSON
                          (generative, likelihood comparison, embedding space, …)
```

- **`run_analysis.py`** — the single command-line entry point. One sub-command per
  analysis type; loads everything from the saved run. Start here.
- **`post_training_analysis/`** — the analysis implementations (importable Python
  API), exposed through a lazy gateway in `post_training_analysis/__init__.py`.
- **`evaluation/`** — shared *evaluation primitives* used by **both** training-time
  held-out evaluation and standalone post-training analysis (network-free inference
  helpers, held-out config, plotting).
- **`resolve_model_run`** (`post_training_analysis/generative_analysis.py`) — the
  canonical loader that turns a run directory into a `ResolvedModelRun`.

Everything is **JAX/Haiku** on the upstream `disentangled_rnns` library and the
in-repo `data_loaders.disrnn_dataset` builder (vendored from the retired
`aind_disrnn_utils` package).

---

## 2. Repository layout

| Path | What it is |
|------|------------|
| `run_analysis.py` | **Main entry point** — unified CLI; one sub-command per analysis ([§4](#4-running-analyses-run_analysispy)) |
| `post_training_analysis/__init__.py` | Lazy public-API gateway (import functions from here) |
| `post_training_analysis/generative_analysis.py` | `resolve_model_run` / `ResolvedModelRun`; generative rollouts; switch & history-dependent switch statistics vs animal |
| `post_training_analysis/likelihood_comparison.py` | Cross-model next-trial prediction likelihood (GRU / disRNN / baseline-RL) |
| `post_training_analysis/likelihood_advantage_analysis.py` | Log-likelihood advantage + RNN / Q state-space visualizations |
| `post_training_analysis/embedding_space_analysis.py` | Subject-embedding-space visualization (multisubject runs) |
| `post_training_analysis/baseline_rl_analysis.py` | Baseline-RL analysis from fitted per-session parameters |
| `post_training_analysis/heldout_finetuning.py` | Held-out subject embedding fine-tuning (**training-adjacent**, [§7](#7-training-adjacent-exceptions)) |
| `post_training_analysis/gru_trial_states.py` | `extract_gru_trial_states`: one forward pass of a saved 2-action GRU over a training-bundle split → flat per-trial table (hidden states, logits, task-state columns) in the `state-space-*` schema, with likelihood and readout-alignment checks |
| `post_training_analysis/gru_counterfactual.py` | Rebuild a GRU run's network + model-input tensor for a split and run it on edited inputs (`reward_flip_effects`: Δlogit after flipping one trial's reward) |
| `post_training_analysis/gru_step.py` | `GruStep`: the trained multisubject GRU's update with the context as a free input (`step`, `logit_diff`, `jacobian`, `run`) plus `subject_context` (the network's own embedding + session-delta module) `find_fixed_points` (forward iteration + damped Newton) and `run_ablated` (mean-replacement ablation of a subspace at every step, global or per-input means; full or memory-only); parity-tested against Haiku |
| `post_training_analysis/baseline_rl_trial_rollout.py` | `rollout_baseline_rl`: per-trial P(right) and Q of per-subject baseline-RL fits on any trial table; Q taken from the agent itself (exact; the shared recovery can be wrong for 2-learning-rate agents) |
| `post_training_analysis/*.ipynb` | Exploratory notebooks (`p_switch_*`, `simulate_sessions`) — see [§8](#8-known-gaps--backlog) |
| `evaluation/heldout_eval_config.py` | `HeldoutEvalConfig` — config→eval bridge (shared by training & eval) |
| `evaluation/common.py` | Model-agnostic helpers (logit→prob math, identifier/filename normalization, subject/session grouping, param loading) |
| `evaluation/disrnn_evaluation.py` | disRNN held-out evaluation + example plots |
| `evaluation/gru_evaluation.py` | GRU held-out evaluation + example plots |
| `evaluation/baseline_rl_evaluation.py` | Baseline-RL held-out evaluation |
| `evaluation/plotting.py` | Framework-free latent/trajectory plotting |
| `utils/{disrnn,gru,baseline_rl}_evaluation.py`, `utils/disrnn_plotting.py` | **Deprecated re-export shims** → `evaluation/*` (back-compat for trainers/`run_capsule.py`; prefer `evaluation.*` in new code) |
| `run_embedding_space_analysis.py`, `run_heldout_subject_finetuning.py` | Older single-purpose argparse CLIs (still work; `run_analysis.py` supersedes them) |
| `tests/` | `unittest` suites (see [§9](#9-testing)) |

---

## 3. The trained-run contract

Every model-dir-based analysis loads a run via
`resolve_model_run(model_dir, split=..., checkpoint_policy=...)`, which expects:

```
<model_dir>/
├── inputs.yaml                 # the fully-resolved run config (written by training)
└── outputs/
    ├── output_summary.json     # metrics, best-checkpoint pointers (optional)
    ├── {disrnn,gru}_config.json / baseline_rl_output.json
    └── checkpoints/
        ├── index.json
        └── step_<N>/params.json
```

- **`split`** — `train` | `eval` | `combined` | `heldout` (which sessions to score/rollout).
- **`checkpoint_policy`** — `best_eval` | `best_heldout` | `final` (which checkpoint to load).
- Returns a `ResolvedModelRun` carrying `params_path`, `config_path`, `model_type`,
  `multisubject`, `ignore_policy`, resolved subject/session IDs, etc.

No Hydra runtime is required: standalone eval reads the saved `inputs.yaml`
directly (it satisfies the same duck-typed interface the live config does during
training). `resolve_model_run` imports **no** `model_trainers`.

---

## 4. Running analyses (`run_analysis.py`)

```bash
python run_analysis.py <subcommand> [options]
python run_analysis.py <subcommand> --help    # full options for any sub-command
```

| Sub-command | Wraps | Key arguments |
|-------------|-------|---------------|
| `generative` | `run_post_training_analysis` | `--model-dir` · `--split` · `--checkpoint-policy` · `--rollout-mode` · `--n-rollouts-per-session` · `--window-size` · `--session-partitions` |
| `from-histories` | `run_post_training_analysis_from_saved_histories` | `--simulated-history` · `--animal-history` · `--resolved-run` · `--window-size` (no model load) |
| `likelihood-comparison` | `run_prediction_likelihood_comparison` | `--model-dirs A B …` · `--model-labels` · `--checkpoint-policy` · `--include-heldout/--no-include-heldout` · `--precomputed-session-metrics` |
| `likelihood-advantage` | `run_likelihood_advantage_analysis` | `--model1-dir` · `--model2-dir` · `--split` · `--include-rnn-state-space/--no-…` · `--include-baseline-q-space/--no-…` |
| `state-space-condition` | `run_rnn_state_space_condition_analysis` | `--trial-advantage` · `--condition-columns` · `--condition-values` · `--probability-color-column` (no model load) |
| `state-space-subject` | `run_rnn_state_space_subject_analysis` | `--trial-advantage` · `--probability-column` · `--subject-ids` · `--subject-embeddings` (no model load) |
| `state-space-overview` | `run_rnn_state_space_overview_analysis` | `--trial-advantage` · `--color-columns` (no model load) |
| `q-space-condition` | `run_baseline_q_space_condition_analysis` | `--trial-advantage` · `--condition-columns` · `--condition-values` (no model load) |
| `q-space-subject` | `run_baseline_q_space_subject_analysis` | `--trial-advantage` · `--probability-column` · `--subject-ids` (no model load) |
| `embedding-params` | `run_subject_embedding_baseline_parameter_analysis` | `--model1-dir` (RNN) · `--model2-dir` (baseline-RL) · `--parameter-names` |
| `embedding` | `run_embedding_space_analysis` | `--model-dir` · `--checkpoint-policy` · `--task-column` · `--room-column` · … |
| `baseline-rl` | `run_baseline_rl_post_training_analysis` | `--resolved-run` · `--fitting-df` · `--model-aliases` · `--n-rollouts-per-session` |
| `finetune` | `run_heldout_subject_finetuning_from_config` | `--config <yaml>` · `--output-root` (**training-adjacent**, [§7](#7-training-adjacent-exceptions)) |

### The two-stage state-space / Q-space flow

`likelihood-advantage` is the **expensive** stage: it loads both models, evaluates them,
rolls out the baseline-RL agent, extracts RNN hidden states, and writes a canonical
trial-level dataframe to `<output-dir>/trial_advantage.pkl` (it also runs the projections
inline by default — `--include-rnn-state-space` / `--include-baseline-q-space`).

The five `state-space-*` / `q-space-*` sub-commands are the **cheap** stage: they read that
pickle and re-render figures **without loading any model**, so a projection, condition split,
or colouring can be changed without repeating the evaluation.

```bash
# Stage 1 (heavy, needs both runs) — writes OUT/trial_advantage.pkl
python run_analysis.py likelihood-advantage \
    --model1-dir /data/<RNN_RUN> --model2-dir /data/<BASELINE_RUN> --output-dir OUT

# Stage 2 (cheap, re-plot only) — no model is loaded
python run_analysis.py state-space-subject --trial-advantage OUT/trial_advantage.pkl \
    --subject-ids 700123 700456 --output-dir OUT
python run_analysis.py q-space-condition --trial-advantage OUT/trial_advantage.pkl \
    --condition-values '{"session_stage": ["early", "late"]}' --output-dir OUT
```

`embedding-params` stands apart: it needs both run directories (not the pickle) and colours
model1's subject embeddings by model2's fitted baseline-RL parameters.

All sub-commands accept `--output-dir` and print a JSON summary. Examples:

```bash
# Generative rollout + switch statistics vs animal, for a saved run
python run_analysis.py generative --model-dir /data/<RUN> \
    --split train --checkpoint-policy best_eval --output-dir /results \
    --session-partitions train eval combined

# Compare next-trial prediction likelihood across several models
python run_analysis.py likelihood-comparison \
    --model-dirs /data/<GRU_RUN> /data/<DISRNN_RUN> /data/<RL_RUN>/1 \
    --model-labels GRU disRNN Bari --no-include-heldout --output-dir /results

# Subject-embedding-space visualization for a multisubject run
python run_analysis.py embedding --model-dir /data/<MULTISUBJECT_RUN> --output-dir /results
```

### Python API

The same functions are importable from the lazy gateway (heavy/training-adjacent
modules load only when the function is first used):

```python
from post_training_analysis import (
    resolve_model_run, run_post_training_analysis,
    run_prediction_likelihood_comparison, run_likelihood_advantage_analysis,
    run_embedding_space_analysis, run_baseline_rl_post_training_analysis,
    compute_switch_stats, compute_history_dependent_switch_stats,
    simulate_model_sessions, load_animal_session_history,
)
result = run_post_training_analysis("/data/<RUN>", split="train", checkpoint_policy="best_eval")
```

State-space sub-analyses not yet exposed on the CLI (e.g.
`run_rnn_state_space_subject_analysis`, `run_baseline_q_space_subject_analysis`)
are available via the Python API.

---

## 5. Analysis types

| Analysis | Produces |
|----------|----------|
| **Generative** | Model rollouts vs animal: switch rate around switches, reward/run-length-conditioned switching, history-dependent switch patterns; per-session/subject points |
| **Likelihood comparison** | Per-trial next-action prediction likelihood for each model on train/eval/combined/heldout splits; per-session & per-subject aggregates |
| **Likelihood advantage** | Trial-level Δ log-likelihood between two models, variable-contribution ranking, and RNN-hidden / Q-value state-space plots |
| **Embedding space** | 2-D projections of learned subject embeddings colored by behavioral metadata; session-context trajectories (multisubject) |
| **Baseline-RL** | Same generative/statistics flow but sourced from fitted RL agent parameters instead of network weights |
| **Held-out fine-tuning** | Fine-tuned embeddings for unseen subjects + checkpoint loss/likelihood curves (**re-trains**, [§7](#7-training-adjacent-exceptions)) |

---

## 6. The shared evaluation layer (`evaluation/`)

`evaluation/` holds primitives used by **both** training-time held-out evaluation
(called from the trainers and `run_capsule.py`) **and** standalone post-training
analysis, so neither side duplicates them:

- `heldout_eval_config.py` — `HeldoutEvalConfig` (+`from_data_cfg`,
  `_resolve_heldout_eval_config`, `should_run_heldout_eval`).
- `common.py` — model-agnostic helpers (`_prob_from_logits`, `_load_saved_params`,
  `_normalize_identifier`, `_iter_subject_session_groups`,
  `_aligned_action_probabilities_from_output_df`, …).
- `{disrnn,gru,baseline_rl}_evaluation.py` — `evaluate_*_on_heldout_subjects`,
  `load_*_heldout_subject_data`, `plot_*_examples_for_split`, `add_gru_model_results`.
- `plotting.py` — `plot_latents_over_trials`, `plot_latents_in_space`, `save_figure`.

> **Back-compat shims:** the old `utils/{disrnn,gru,baseline_rl}_evaluation.py` and
> `utils/disrnn_plotting.py` now re-export everything from `evaluation/*`, so the
> trainers and `run_capsule.py` are unchanged. New code should import from
> `evaluation.*`.

---

## 7. Training-adjacent exceptions

Post-training analysis is import-independent of training — **except** these two,
which import the trainers **lazily** (never at package import time) and are invoked
explicitly:

1. **`heldout_finetuning.run_heldout_subject_finetuning_from_config`** — genuinely
   *re-trains* held-out subject embeddings, so it legitimately uses
   `DisrnnTrainer`/`GruTrainer` and the training utilities. It also accepts a
   canonical external target through `target_data` or `target_bundle`, freezes
   the recurrent core, and uses a fixed-final-step policy so target-test data do
   not select adaptation hyperparameters or checkpoints.
2. **`likelihood_comparison._evaluate_disrnn_dataset`** — for the **multisubject
   disRNN** case only, it borrows the trainer's network-construction helpers
   (single-subject eval uses `evaluation.disrnn_evaluation`'s own builder).

> **Planned clean fix** (deferred to avoid touching training code): extract disRNN
> network construction into `models/disrnn_network.py` (symmetric with
> `models/gru_network.py`) and have both the trainer and eval call it.

---

## 8. Known gaps & backlog

Tracked here so the doc reflects reality. From the evaluation-code audit:

- **All-ignored-session alignment** is handled in two places
  (`evaluation/disrnn_evaluation.load_disrnn_heldout_subject_data` vs
  `evaluation/gru_evaluation.add_gru_model_results`); should be unified.
- **`run_capsule.py` final held-out eval** duplicates near-identical disRNN/GRU
  blocks; should collapse to one helper (also fixes the disRNN-vs-GRU `wandb_step`
  warmup-offset inconsistency).
- **Config-drift:** `output_dir` defaults silently to `/results/outputs`;
  GRU/disRNN config keys diverge; `HeldoutEvalConfig` reads `type`→`data_type`.
- **Notebook duplication:** `p_switch_*.ipynb` reimplement
  `compute_switch_stats` / `compute_history_dependent_switch_stats`.
- **Dead test fixture:** `tests/test_post_training_analysis.py` references a
  nonexistent `ex_model_dir-…` directory (4 erroring tests).

These are behavior-changing and intentionally **not** part of the reorg; see the
plan's "Phase B".

---

## 9. Testing

```bash
# pytest is not installed — use unittest
python -m unittest tests.test_post_training_analysis tests.test_likelihood_comparison \
    tests.test_likelihood_advantage_analysis tests.test_embedding_space_analysis \
    tests.test_baseline_rl_post_training_analysis
```

> Run with `python -m unittest`; **do not** pipe through `tail` (it masks the exit
> code). The suite currently has ~25 *pre-existing* failures unrelated to the
> evaluation reorg (mostly `test_disrnn_trainer` + the dead fixture above); compare
> against that known set rather than expecting all-green.

---

## 10. Extending the evaluation codebase

- **New analysis:** add the implementation under `post_training_analysis/`, export
  it via the lazy map in `post_training_analysis/__init__.py`, and add a
  `run_analysis.py` sub-command (lazy-import the function inside the handler so
  `import run_analysis` stays training-free). Add a `tests/test_<analysis>.py`.
- **New shared eval primitive:** put model-agnostic helpers in `evaluation/common.py`
  and config logic in `evaluation/heldout_eval_config.py` so training-time and
  post-training callers share one implementation.
- **Loading a trained run:** always go through `resolve_model_run`; don't
  re-implement run discovery or read checkpoints directly.
- **Keep it training-independent:** do not import `model_trainers` at module load;
  if an analysis truly needs training, lazy-import and document it in
  [§7](#7-training-adjacent-exceptions).
- **Always:** add/extend a test, run the affected suites, and **update this file +
  the Changelog**.

---

## Changelog

> Add a dated entry (newest first) whenever you add or change an evaluation feature.

### 2026-10-16
- **Counterfactual GRU runs and per-trial baseline-RL rollouts.** `gru_counterfactual` and
  `baseline_rl_trial_rollout` (above). `likelihood_comparison._build_gru_network` is split
  out of `_evaluate_gru_dataset` so both use one network construction.
- **Free-context step.** `gru_step` (above) for flow fields, fixed points and Jacobians.
- **Known issue (not changed):** `likelihood_advantage_analysis`'s manual ForagerQLearning Q
  fallback assumes one learning rate; for Hattori L2F1 fits its reproduction check can pass
  with wrong Q values. `baseline_rl_trial_rollout` avoids that path.
- **Per-trial GRU state table.** `gru_trial_states.extract_gru_trial_states` returns every
  choice trial of a split with `rnn_state_*`, logits/probabilities and optional raw task
  columns (reward probabilities, bait state), aligned so row *t*'s state produced row *t*'s
  prediction; it verifies that by reconstructing each logit from the readout.
  `likelihood_comparison._load_training_bundle_for_run` gains an optional
  `cols_to_retain` override for those raw columns (model inputs unchanged).

### 2026-09-04
- **External target-transfer evaluation.** Held-out embedding fine-tuning accepts
  canonical external bandit bundles, applies predeclared zero/few-shot budgets,
  fixes model selection before reading target test performance, and exports
  `test_trial_predictions.csv` plus `test_metrics.json`. The shared
  `evaluation.target_transfer` contract keeps GRU/disRNN and Q-learning trial
  keys and binary prediction metrics directly comparable.

### 2026-06-18
- **Evaluation made training-independent + this guide created.** Introduced the
  `evaluation/` package (moved `{disrnn,gru,baseline_rl}_evaluation` + `plotting`
  out of `utils/`; split shared `HeldoutEvalConfig` → `evaluation/heldout_eval_config.py`
  and model-agnostic helpers → `evaluation/common.py`, breaking the GRU→disRNN
  private-helper coupling). Added the unified **`run_analysis.py`** CLI; deprecated the
  seven `run_capsule-test_*.py` scratch scripts to stubs pointing at it. Transparent
  re-export shims left at `utils/*` so training is unchanged. Documented the two
  training-adjacent exceptions. No training-code changes; no test regressions.
  (commit `73fd4c2`)
- **Naming.** Renamed this guide `EVALUATION.md` → `POST_TRAINING_ANALYSIS.md` and the
  CLI `run_eval.py` → `run_analysis.py`, to avoid confusion with training-time held-out
  *evaluation*. The shared `evaluation/` package (called by the trainers during training)
  intentionally keeps its name — it is the held-out-evaluation primitive layer, not
  post-training-specific.
