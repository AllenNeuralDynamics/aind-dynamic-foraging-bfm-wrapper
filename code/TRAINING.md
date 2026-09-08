# Training Codebase Guide

> **Living document.** This is the single reference for what the training codebase
> includes and how to use it. **When you add or change a feature, update this file**
> (and add a dated entry to the [Changelog](#changelog) at the bottom). Keep the
> inline comments in `configs/*.yaml` as the authoritative per-key reference; this
> guide explains the concepts, workflows, and how the pieces fit together.

Scope: everything under `code/` that trains GRU / disRNN / baseline-RL models on
mouse foraging behavior (and synthetic agents), evaluates them, and runs
post-training analysis. For the post-training *analysis* entrypoints specifically,
see also the repo-root [README.md](../README.md).

---

## 1. Overview

The pipeline is a small, composable stack wired together by Hydra config:

```
config.yaml ──► run_capsule.main()
                  │
                  ├─ instantiate(data)   ──► DatasetLoader.load() ──► DatasetBundle
                  │                                                    (raw df, train_set, eval_set, metadata)
                  ├─ instantiate(model)  ──► ModelTrainer.fit(bundle) ──► trained params + metrics
                  │                                                       (+ checkpoints, plots, W&B)
                  └─ held-out evaluation (single-subject) / auto fine-tuning (multisubject)
```

- **Data loaders** (`data_loaders/`) turn a data source (foraging database, docDB,
  or synthetic agents) into a `DatasetBundle` with train/eval splits.
- **Model trainers** (`model_trainers/`) fit a model and own the training loop,
  checkpointing, evaluation, plotting, and W&B logging.
- **Models** (`models/`) are the JAX/Haiku network definitions (GRU, multisubject
  disRNN, session-conditioning, subject embeddings).
- **`run_capsule.py`** is the orchestrator: load config → load data → train →
  evaluate held-out → log to Weights & Biases.
- **`post_training_analysis/`** holds analysis run *after* training
  (generative rollouts, embedding space, likelihood comparisons, held-out
  fine-tuning).

Everything is **JAX/Haiku** built on the upstream `disentangled_rnns` library and
the in-repo `data_loaders.disrnn_dataset` builder (vendored from the retired
`aind_disrnn_utils` package).

---

## 1.5 Run lifecycle & key switches (read first)

> Two independent sessions once misread the log line *"Skipping held-out evaluation
> for multisubject …"* as *"no held-out numbers at all."* It only means the
> **per-checkpoint** held-out eval is off — the **end-of-training held-out
> fine-tune still runs**. This section exists so that never happens again. If a
> log message and this section disagree, trust the code path named here.

### The four phases of a run (in order)
1. **Warmup** — `n_warmup_steps` of penalty ramp-up (disRNN) / session-curriculum
   warmup. Loss is high and bottlenecks are held open here; this is *not* the model
   failing. **The W&B `_step` axis includes the warmup offset**, so main-phase
   progress is `(_step − n_warmup_steps) / n_steps`, not `_step / n_steps`.
2. **Main training** — the `while steps_completed < n_steps` loop. Checkpoints are
   written every `checkpoint_every_n_steps` (chunked path only — see resumability
   below).
3. **Artifact upload** — after the main loop, the *entire* `output_dir` (params +
   `train_state.pkl` + plots) is uploaded to W&B as `<mtype>-output-<run_id>`
   (type `training-output`). **This happens once, at the end — not per checkpoint.**
   A run that hasn't finished has no restorable artifact yet.
4. **Held-out fine-tune** (multisubject GRU/disRNN, on by default) — fine-tune a
   fresh subject embedding on each reserved held-out mouse, predict its other
   sessions, log `heldout/*`. This keeps the job in `running`/`idle` briefly after
   training + upload are already done.

### The two DIFFERENT held-out switches (the source of the confusion)
| Switch | What it controls | Default | Skipping it means |
|---|---|---|---|
| `model.training.checkpoint_run_heldout_eval` | **per-checkpoint** held-out eval *during* training (expensive) | often `false` | only the mid-training held-out curve is off |
| `model.training.auto_heldout_finetune.enabled` | the **end-of-training** held-out fine-tune+test (the real figure of merit) | **`true`** | **NO `heldout/*` metrics at all** |

These are **not the same knob.** The *"Skipping held-out evaluation … seen-subject
personalization only"* log line refers to the **first** (per-checkpoint eval); the
second still runs and produces `heldout/final/eval_likelihood`. Full detail in §8.

- **Held-out-mouse likelihood is the correct figure of merit** for generalization
  questions; within-subject `checkpoint/eval_likelihood` **saturates** (~0.72–0.75
  across model sizes) and cannot discriminate. Read `heldout/final/eval_likelihood`.

### Checkpoints, resumability, and extendability (three distinct things)
- **Checkpointing** — only the *chunked* training path writes resumable state, so
  it requires `checkpoint_every_n_steps > 0`. Each checkpoint writes params +
  `train_state.pkl` (optimizer state, `steps_completed`, rng) under
  `output_dir/checkpoints/step_*`.
- **Resumability = WITHIN one experiment (preemption recovery).** A preempted task
  restarts as the *same* task with the *same* `/results` dataset (anchored by
  `BFM_RESUMABLE_OUTPUT_DIR=/results/run`). On start, `find_latest_resumable_state`
  loads the newest checkpoint and the loop continues from `steps_completed`,
  **skipping warmup** (warmup is already folded into the checkpointed params).
  Gated by `auto_resume` (default `true`) + `checkpoint_every_n_steps > 0`; W&B
  continuity via a deterministic `WANDB_RUN_ID` + `WANDB_RESUME=allow`. Automatic,
  no flags needed.
- **Extendability = ACROSS experiments (staged horizons).** A *new* experiment gets
  a *fresh, empty* `/results`, so it does **not** see the old run's checkpoints and
  would restart from scratch — **unless** you set
  `model.training.restore_from_run_id=<source W&B run name>` (or env
  `BFM_RESTORE_FROM_RUN_ID`). That downloads the source run's
  `<mtype>-output-<run_id>:latest` artifact into the new run's `outputs/` so resume
  finds the checkpoint and continues (skipping warmup). Set the new `n_steps`
  **larger** than the source's. Prerequisite: the source run must have **finished**
  (phase 3 above) so its artifact is `COMMITTED`. Trainer-agnostic (GRU + disRNN
  upload the same artifact shape). Helper:
  `run_helpers.maybe_restore_checkpoint_from_wandb`, called from `run_hpc.py`.

### Other switches that surprise people
- **`model.training.freeze_gru_core`** (GRU only, default `false`) turns the
  multisubject GRU into a frozen random reservoir for a controlled ablation. The
  native seeded initialization is unchanged; all parameters except
  `subject_embeddings` and the final `readout` are masked to zero updates. The
  end-of-training held-out path is unchanged and still adapts only each new
  subject embedding. This switch requires multisubject mode.
- **`length_bucketing`** (+ `length_bucket_grid`, default 128): trims each
  `random`-mode batch's RNN unroll to the batch's own session length instead of the
  global `T_max`. Big speedup (~1.86× measured on 100-mice disRNN). Requires
  `batch_mode=random`; no-op under `single`/full-batch. Runs in the **training**
  loop (both trainers), *not* rollout/generation. The disRNN/GRU configs must
  **declare** the key (Hydra struct mode rejects overriding an absent key).
- **disRNN has NO early stopping.** Only `gru_trainer` has opt-in loss-based early
  stopping. A disRNN sweep must **omit** `early_stopping` keys or Hydra errors.
- **Bottleneck sigma convention:** small σ = **OPEN** (info flows), σ→1 = **CLOSED**.
  So openness *decreases* as σ→1 (sparser). At init all bottlenecks are open.
- **Reading the sparsity metrics — mind the threshold.** A single hard-threshold
  `frac_open` is **misleading**: `bottlenecks/<fam>_frac_open` (σ<0.1) *saturates* to
  0 while channels sit at σ≈0.85 (mostly-but-not-fully closed), whereas the dashboard
  `plot_bottlenecks` figure calls a latent "open" at the far more permissive σ<0.97
  (it logs `open latents by bottleneck > 0.03`, i.e. 1−σ>0.03) — so the same run can
  read `frac_open=0.0` in the scalar and "almost all open" in the figure. They are not
  contradictory; they are different cutoffs on the same σ distribution. **Prefer the
  threshold-free readouts** logged alongside: `bottlenecks/<fam>_n_eff_open_frac` (the
  normalized participation ratio of channel openness `1−σ`, in [1/n,1], **comparable
  across families of different size** — 5-element latent vs the matrix-shaped
  update-net bottleneck), plus `sigma_median` / `sigma_p10` / `sigma_p90` (catch a
  bimodal open/closed split) and `total_openness` (=Σ(1−σ), raw capacity). `frac_open`
  is kept as a multi-threshold curve (`_frac_open_s0p03/0p1/0p5/0p9/0p97`) so both the
  strict and figure conventions are visible.

---

## 2. Repository layout

| Path | What it is |
|------|------------|
| `run_capsule.py` | **Main entry point** — train + evaluate one run from a config |
| `run` | Code Ocean launcher (`python -u run_capsule.py "$@"`) |
| `configs/` | Hydra config templates (`config_gru.yaml`, `config_disrnn.yaml`, `config_baseline_rl.yaml`, `config_heldout_subject_finetuning.yaml`) — every key is documented inline |
| `base/` | `interfaces.py` (`DatasetLoader`, `ModelTrainer` ABCs), `types.py` (`DatasetBundle`) |
| `data_loaders/` | `mice.py` (database + docDB loaders), `synthetic.py` (synthetic agents / task-trained RNN) |
| `models/` | `gru_network.py`, `multisubject_disrnn.py`, `session_conditioning.py`, `subject_embedding_initialization.py` |
| `model_trainers/` | `base_multisubject_trainer.py` (shared base), `gru_trainer.py`, `disrnn_trainer.py`, `baseline_rl_trainer.py` |
| `utils/` | `run_helpers.py`, `multisubject.py`, `session_regularized_training.py`, `load_mice_database.py` (the `*_evaluation.py` / `disrnn_plotting.py` here are now deprecated re-export shims → `evaluation/`) |
| `evaluation/` | Shared evaluation primitives used by **both** training-time held-out eval and post-training analysis: `heldout_eval_config.py`, `common.py`, `{disrnn,gru,baseline_rl}_evaluation.py`, `plotting.py` |
| `post_training_analysis/` | `generative_analysis.py`, `embedding_space_analysis.py`, `likelihood_comparison.py`, `likelihood_advantage_analysis.py`, `baseline_rl_analysis.py`, `heldout_finetuning.py` — see **[POST_TRAINING_ANALYSIS.md](POST_TRAINING_ANALYSIS.md)** |
| `run_analysis.py` | **Unified post-training-analysis CLI** (see [POST_TRAINING_ANALYSIS.md](POST_TRAINING_ANALYSIS.md)) |
| `run_heldout_subject_finetuning.py`, `run_embedding_space_analysis.py` | Older single-purpose analysis CLIs (superseded by `run_analysis.py`) |
| `tests/` | `unittest` suites (see [§9 Testing](#9-testing)) |
| `load_mice_data.py` | Pull/cache mouse data snapshots from the database |

---

## 3. Running a training job

### In Code Ocean (capsule)
`run` executes `python -u run_capsule.py`. At runtime `run_capsule` finds the
active config via `utils.run_helpers.find_hydra_config()`, which globs
`/data/jobs/**/config.yaml` (injected by the capsule/pipeline). The files in
`configs/` are the **maintained templates** you copy/select from; pick one model
type and one data source by editing the active blocks.

Outputs are written to `/results/` (see [§7 Outputs](#7-outputs--weights--biases)).

### Locally / ad hoc
Point `run_capsule` at a config by placing it under `/data/jobs/<name>/config.yaml`,
or import and drive the trainers directly (see how `tests/` construct
`GruTrainer`/`DisrnnTrainer` and call `.fit(bundle)`).

### Selecting what runs
The `run` script has commented alternates (baseline-RL, fine-tuning, analysis,
data loading). Uncomment the desired line, or call `run_analysis.py <sub-command>`
(see **[POST_TRAINING_ANALYSIS.md](POST_TRAINING_ANALYSIS.md)**).

---

## 4. Configuration

A config has four top-level sections; **`configs/config_gru.yaml` and
`configs/config_disrnn.yaml` document every key inline** — treat them as the
reference. High level:

```yaml
data:    # which dataset + how to split it      (type, _target_, selection, eval_every_n, ...)
model:   # architecture + training schedule      (type, _target_, architecture, training, ...)
seed: 43
wandb:   # entity / project / dir / name
job_id: 0
```

- `${...}` are OmegaConf interpolations resolved at load time (e.g. `${seed}`,
  `${.architecture.hidden_size}`, `${.penalties.beta}`).
- Exactly **one** `data:` block and **one** `model:` block are active; the others
  are kept as commented, ready-to-use presets.
- disRNN penalty multipliers (`<name>_multiplier`) are resolved into effective
  penalties before training (`utils.run_helpers.resolve_disrnn_penalties`).
- Multisubject mode auto-appends `_multisubject` to the W&B run name.

### External binary two-arm bandit files

`data_loaders.external_bandit.ExternalBanditDatasetLoader` is the file boundary
for behavioral datasets collected outside AIND. It always returns a multisubject
`DatasetBundle`; its `train_set` is the adaptation partition and its
`eval_set` is the untouched test partition.

The canonical Parquet table has one row per trial and requires:

| Column | Contract |
|--------|----------|
| `subject_id` | Stable subject identifier (string or integer) |
| `ses_idx` | Stable session identifier within the subject |
| `trial` | Integer-valued within-session trial index |
| `animal_response` | Chosen arm, encoded `0` or `1` |
| `rewarded` | Binary outcome consumed by the GRU, encoded `0` or `1` |
| `earned_reward` | Binary outcome, encoded `0` or `1` |

Rows must be unique by `(subject_id, ses_idx, trial)`. Optional scalar
`dataset_id` and `species` columns may be included for provenance; when the
same fields are supplied in the config or manifest, all values must agree. The
two reward aliases must agree row by row.

Splits are versioned JSON rather than recomputed from row order:

```json
{
  "schema_version": 1,
  "dataset_id": "example-rats",
  "species": "rat",
  "subjects": [
    {
      "subject_id": "rat-01",
      "adapt_session_ids": ["s01", "s03"],
      "test_session_ids": ["s02", "s04"]
    }
  ]
}
```

Every retained subject needs at least one adaptation and one test session. The
two lists must be disjoint and together cover every retained session exactly
once; missing, duplicated, overlapping, and unknown session IDs fail before
training. Subject indexing follows the manifest's subject-list order, and the
manifest list order is preserved in each split.

Single-session datasets use schema version 2 with a prefix/suffix boundary:

    {
      "schema_version": 2,
      "dataset_id": "example-humans",
      "species": "human",
      "split_strategy": "within_session_prefix_suffix",
      "subjects": [
        {
          "subject_id": "person-01",
          "session_id": "main",
          "adapt_prefix_trials": 150,
          "total_trials": 300
        }
      ]
    }

The adaptation dataset ends at the boundary. The evaluation dataset retains
the full input sequence so the prefix establishes the recurrent initial
condition at the suffix boundary, but prefix targets are masked from scoring.
The raw table is annotated with `external_split_partition=adapt|test`, and the
bundle metadata records that partition column so downstream evaluation aligns
the adaptation rows to `train_set`, runs the full sequence for `eval_set`, and
scores only the test suffix. This avoids inventing pseudo-sessions or resetting
hidden state. Baseline-RL evaluation follows the same contract: fitted
parameters stay fixed while the full sequence updates the agent's Q-state,
and likelihood metrics include only suffix choices. Start from
`configs/data/external_bandit.yaml`; `subject_ids` may select a manifest
subset, while `null` requires exact agreement between table and manifest
subjects.

Session-level few-shot budgets are applied at this shared loader boundary, so
GRU/disRNN and subject-level Q-learning consume exactly the same adaptation
sessions while the test partition remains immutable. For schema version 1,
`adapt_sessions_per_subject: K` means the first K entries in each subject's
**manifest adaptation list** (for the dispatcher-generated odd/even split, the
first K odd-positioned adaptation sessions), not the first K sessions before the
split. `K=0` is the GRU/disRNN zero-shot condition initialized from the
source-subject embedding mean; a fitted subject-level Q-learning model needs at
least one adaptation session and therefore has no K=0 result. Schema version 2
does not use K: it always fits on the complete manifest-declared prefix and
scores the complete suffix.

---

## 5. Models & trainers

### Model types (`model.type`)
| `type` | `_target_` | Notes |
|--------|-----------|-------|
| `gru` | `model_trainers.gru_trainer.GruTrainer` | GRU baseline; `num_layers` must be 1 |
| `disrnn` | `model_trainers.disrnn_trainer.DisrnnTrainer` | Disentangled RNN with information-bottleneck penalties + warmup phase |
| `baseline_rl` | `model_trainers.baseline_rl_trainer.BaselineRLTrainer` | Classic RL agents (e.g. `ForagerQLearning`) fit by differential evolution |

### Shared base class
`GruTrainer` and `DisrnnTrainer` both subclass **`BaseMultisubjectTrainer`**
(`model_trainers/base_multisubject_trainer.py`), which owns the logic that used to
be duplicated between them: the checkpoint/initialization snapshot pipeline
(`_evaluate_initialization_snapshot`), per-split example plotting
(`_generate_split_examples`), subject-embedding / session-context state-space
plots, media cleanup, W&B held-out logging, and loss plotting. Subclasses set
`_MODEL_LABEL` / `_TRAINER_CONTEXT_NAME` (for log text) and override small hooks
(`_resolve_n_action_logits`, `_add_model_results`, `_plot_examples_for_split`,
`_plot_model_specific_diagnostics`, `_evaluate_heldout_subjects`,
`_snapshot_extra_summary_fields`). **When adding shared behavior, put it on the
base; only model-specific bits belong in the hooks.** `BaselineRLTrainer` is
independent (different fit/analysis).

### Multisubject personalization
With `architecture.multisubject: true`, the model learns a per-subject embedding
(a learned table, `models/subject_embedding_initialization.py`) concatenated into
the network input. disRNN can additionally apply an information bottleneck to the
subject embedding (`use_global_subject_bottleneck`) and per-network subject
penalties. The data loader prepends `subject_index` as input feature 0.

### Session conditioning (multisubject only)
A learned, session-indexed perturbation (a "session-delta" MLP) is added on top of
each subject embedding, gated by a curriculum schedule (0 during pretrain, linear
ramp over warmup, 1 after). Controlled by `architecture.session_encoding_type`
(`none|scalar|fourier`), `session_integration_type`, `session_fourier_k`,
`session_delta_n_layers/hidden_size`, and `session_n_pretrain_steps/n_warmup_steps`
(null → ~30%/~20% of total steps). Optional zero-mean regularization via
`training.lambda_reg_session`. Core math: `models/session_conditioning.py`. When
enabled, input feature 1 is `session_index`.

### disRNN specifics
- **Penalties** (`model.penalties`): KL information-bottleneck weights on the
  latent state, choice-net / update-net inputs, and (multisubject) subject context.
  `beta` is the default for any unset penalty.
- **Warmup**: `training.n_warmup_steps` trains a *noiseless* copy of the network
  with all penalties zeroed (penalty-free warmup) before the penalized phase.
- **Distillation** (`model.distillation`): optional GRU→disRNN knowledge
  distillation — train the disRNN student toward the temperature-softened mean of
  one or more trained GRU teachers. v1: `aggregation=mean_probs`, `loss=kl`.

### baseline_rl specifics
Classic RL agents (`agent_class`, e.g. `ForagerQLearning`) fit by differential
evolution — **one agent fit per subject**, independently (no shared model/embedding).
- **Database (`mice_snapshot`) loading is supported** for both single- and
  multi-subject runs. Multisubject fits one agent per training subject
  (parallelized by `multisubject_subject_workers`) and reports per-subject +
  pooled train/eval-split likelihood (`train_likelihood`/`eval_likelihood`, also
  surfaced as `checkpoint/train_likelihood`/`checkpoint/eval_likelihood` for
  cross-model W&B parity).
- **Held-out subjects** are *re-fit*, not transferred: when
  `model.heldout_refit.enabled` (default on for `config_baseline_rl.yaml`) and
  held-out eval applies, `run_capsule` loads the reserved held-out subjects as a
  multisubject bundle and `BaselineRLTrainer.fit_heldout` fits a fresh agent per
  held-out subject (train/eval split), logging aggregate
  `heldout/train_likelihood` + `heldout/eval_likelihood` into the same W&B run.
  Set `model.heldout_refit.skip_train_fit=true` for held-out-only RL baselines
  that should not run the main training-subject fit.
  (parity with the NN models) and writing artifacts under `outputs/heldout_test/`.
  This replaces the older transfer-based held-out eval (apply trained params to
  held-out), which is retired from the run path.

---

## 6. Data loaders & splits

### Sources (`data.type`)
| `type` | Class | Source |
|--------|-------|--------|
| `mice_snapshot` | `MiceSnapshotDatasetLoader` | Foraging **database** snapshot (the standard path) |
| `mice` | `MiceDatasetLoader` | docDB per-subject query |
| `synthetic_task_trained_rnn` | `TaskTrainedRNNDatasetLoader` | Synthetic task-trained RNN logs |
| (tests) | `SyntheticCognitiveAgents` | Synthetic RL agents — used by the test suite |

### Subject selection (`mice_snapshot`)
Subjects with ≥ `min_sessions` sessions are assigned their most-common task as a
curriculum and ranked per curriculum by session count. A fixed ~20% **held-out**
test set is reserved (every `heldout_every_n`-th in rank order); the training set
is a seeded `subject_ratio` sample of the remaining ~80% per curriculum. Pass an
explicit `subject_ids` list to bypass the pipeline entirely.

### Train/eval split
Within the selected subjects, every `eval_every_n`-th session goes to the eval
split (applied **per subject** in multisubject mode via
`utils.multisubject.compute_train_eval_session_ids`, then merged with
`subject_index` prepended). `ignore_policy` controls no-response trials
(`exclude` → 2 classes, `include` → 3).

### Optional timing inputs — reaction time & lick counts (`data.timing_features`)

Off by default. When enabled, the loader appends **previous-trial** reaction time
and post-go-cue lick counts to the observation vector, alongside the standard
prev-choice + prev-reward:

```yaml
data:
  timing_features:
    enabled: true        # master switch; false -> bit-identical to the base model
    reaction_time: true  # +1 feature: prev log RT
    lick_counts: true    # +2 features: prev n_lick_left, prev n_lick_right
    lick_window_s: 2.0   # licks counted in [go_cue, go_cue + this)
    standardize: true    # center/scale continuous channels (see below)
```

`obs_size` **widens automatically** — the trainers derive it from the input tensor
width (`dataset._xs.shape[2]` minus packed context features), so no MODEL block
change is needed. With all features on, `obs_size` goes 2 → 5 and `x_names`
becomes `[prev choice, prev reward, prev log RT, prev n_lick_left,
prev n_lick_right]`.

**Where the numbers come from (two readers, two conventions).** The parquet cache
is built by two NWB readers that store timing differently, so
`utils/trial_timing_features.py` normalizes both:

| quantity | `bonsai_s3` | `co_asset` |
|---|---|---|
| reaction time | `reaction_time` column | `choice_time_in_session − goCue_start_time_in_session` |
| go-cue | `goCue_start_time` | `goCue_start_time_in_session` |
| lick times | trial-table VARCHAR arrays | **absent** — trial-table arrays unpopulated |

A `COALESCE` recovers reaction time on ~100% of responded trials. Lick counts are
always taken from the **event table** (`left_lick_time` / `right_lick_time` on the
session clock), which is populated for both readers; event-derived counts were
cross-validated to exact equality against the bonsai trial-table arrays.

**`features` REPLACES, it does not extend.** A non-empty `features` dict fully
overrides the library default, so `_augment_features_with_timing()` re-lists
prev-choice/prev-reward explicitly. Omitting them would silently drop the base
inputs — the failure mode to watch for if you hand-write a `features` mapping.

**`standardize` (default true).** Centers/scales the continuous channels by fixed
documented population constants. This matters because the disRNN bottleneck's KL
penalty is *quadratic* in input magnitude
(`elementwise_kl = mus**2 + sigma**2 - 1 - log(sigma**2)`): measured at
multiplier 1, mean `mus**2` was 0.37 for a binary choice channel but 32.1 for raw
lick counts — an ~87× spread in KL cost for the same information. Standardizing
brings all five channels into 0.37–1.10. The learned per-dimension multiplier can
absorb scale in principle, but (a) early training still pays the inflated penalty
and (b) the **per-channel σ readouts** (`bottlenecks/update_net_obs_*`) are only
comparable across channels when inputs are on comparable scales. For a **GRU**
(no bottleneck) the choice is largely cosmetic, but keep it consistent across
model families so an architecture comparison is not confounded by preprocessing.

The constants are *fixed*, not per-run fitted, because the train and held-out
loaders are instantiated independently — a fitted transform would have to be
persisted and threaded between them, with a silent train/held-out mismatch as the
failure mode. Standardization is **global, never per-subject**: per-subject
z-scoring would erase the between-mouse differences in reaction time and licking
vigor that the subject embedding exists to capture.

> **⚠️ Inherited int64 truncation bug (worked around here).**
> `data_loaders.disrnn_dataset.create_disrnn_dataset` allocates the input tensor as
> `np.full((...), -1)` — an **int64** array — so float feature columns are
> truncated toward zero on assignment; the later `.astype(float)` is too late.
> This is invisible for the stock integer features (`animal_response`,
> `rewarded`) but destroys continuous ones: on real data, 24,149 distinct log-RT
> values collapsed to **7** integers. Present both in the SHA pinned by
> `pyproject.toml` and in the latest release (0.0.16). `_create_disrnn_dataset()`
> therefore routes any feature set containing a continuous column to
> `utils.trial_timing_features.create_disrnn_dataset_float`, which is semantically
> identical but allocates float. Integer-only feature sets still call upstream, so
> **existing runs remain bit-for-bit reproducible**. Remove the shim once the
> upstream dtype is fixed and the pin moves.

**Calibrating before you spend GPU.** `analysis/calibrate_timing_features.py`
fits nested logistic models with a session-held-out split to measure the
*incremental* predictive value of these inputs over a choice+reward history
baseline. It is a linear lower bound — a disRNN can exploit structure a logistic
model cannot — so treat a positive Δ as a conservative go-signal and a near-zero Δ
as an argument against.

---

## 7. Outputs & Weights & Biases

A run writes to `/results/`:
- `/results/inputs.yaml` — the fully resolved config (backup).
- `/results/outputs/params.json` — trained parameters.
- `/results/outputs/{gru,disrnn}_config.json` — resolved model config.
- `/results/outputs/output_summary.json` — final metrics + metadata.
- `/results/outputs/subject_index_map.json`, `session_context_map.json`,
  `multisubject_metadata.json` — multisubject artifacts (required by downstream
  analysis / fine-tuning).
- `/results/outputs/checkpoints/` — per-checkpoint params + `index.json`.
- figures + (optional) `output_df.csv`.

**W&B keys** (logged to the run named `${data.run_name_component}_${model.run_name_component}`):
- Training subjects: `checkpoint/train_likelihood`, `checkpoint/eval_likelihood`,
  `checkpoint/step`, plus `final/*` summary and example/diagnostic images.
- Held-out subjects (multisubject auto fine-tuning, see §8): `heldout/train_likelihood`,
  `heldout/eval_likelihood`, `heldout/train_loss`, `heldout/eval_loss`,
  `heldout/step`, and `heldout/final/*` — logged into the **same** run, on a step
  axis offset past the training steps.

---

## 8. Held-out test subjects

The ~20% reserved held-out subjects (see §6) are handled differently by mode:

- **Single-subject runs:** held-out evaluation auto-enables for database
  (`mice_snapshot`) runs (`HeldoutEvalConfig.enabled`; disable with
  `data.heldout_eval: false`). `run_capsule` evaluates the trained model directly
  on the held-out subjects at checkpoints and at the end.

- **Multisubject GRU/disRNN runs:** the model has no embedding for an unseen
  subject, so it can't be evaluated zero-shot. Held-out generalization is measured
  by **fine-tuning a fresh embedding per held-out subject** (the rest of the model
  frozen) and reporting likelihood.

- **baseline_rl runs (single- or multi-subject):** RL agents fit per subject, so
  held-out subjects are simply **re-fit** (a fresh agent per held-out subject,
  train/eval split) — gated by `model.heldout_refit.enabled` (see §5). Reports
  `heldout/train_likelihood` + `heldout/eval_likelihood` like the NN models. No
  embedding transfer/fine-tuning is involved.

### Automatic multisubject held-out fine-tuning + evaluation
Controlled by `model.training.auto_heldout_finetune` (enabled by default in the
active GRU/disRNN configs). At the end of a multisubject GRU/disRNN run,
`run_capsule` automatically:
1. resolves the just-finished run at `/results` (fine-tuning from the `final`
   checkpoint by default),
2. loads the reserved held-out subjects and splits **each** subject's sessions
   into train/eval with the same `eval_every_n` as training subjects,
3. expands the subject-embedding table with a new row per held-out subject and
   fine-tunes **only those rows**,
4. logs aggregate held-out **train/eval split likelihoods** into the same W&B run
   under `heldout/*` (mirroring the training-subject metrics).

**Logging/plotting cadence mirrors training.** `checkpoint_every_n_steps` (default
10) controls how often the held-out fine-tune evaluates and logs
`heldout/train_likelihood`/`heldout/eval_likelihood` (so you get a curve, not just
endpoints); `checkpoint_plot_split_examples_every_n` (default 10) controls
split-example plotting. State-space figures and the loss/likelihood-over-checkpoints
curves are logged as images under `heldout/fig/*` (parity with the training
`fig/validation_loss_curve`). Set the cadence knobs to 0 for endpoints-only / no plots.

It is guarded so a fine-tuning failure never fails an otherwise-successful run
(records `output["heldout_finetune"]` with the error instead). Tune via
`auto_heldout_finetune.{n_steps,lr,checkpoint_policy,checkpoint_every_n_steps,checkpoint_plot_split_examples_every_n,...}`;
set `enabled: false` to skip.

### Standalone fine-tuning CLI
The same pipeline can be run manually against any trained run:
```
python run_heldout_subject_finetuning.py --config configs/config_heldout_subject_finetuning.yaml
```
This writes its own run dir under `output.output_root` and (optionally) its own
W&B run. Implementation: `post_training_analysis/heldout_finetuning.py`.

The standalone config can instead define `target_data` with an
`ExternalBanditDatasetLoader`. In that mode the frozen source recurrent core is
evaluated zero-shot or after adapting only the new subject-embedding rows. The
policy is `fixed_final`: target-test likelihood never selects a checkpoint or
fine-tuning step. After fitting, evaluation replays each complete target
sequence to reconstruct the recurrent/Q state at the test boundary, keeps all
learned parameters fixed, and scores only immutable test rows. Both neural and
Q-learning paths write `test_trial_predictions.csv` and `test_metrics.json`
with matched `(subject_id, ses_idx, trial)` keys, log likelihood in nats/bits,
normalized likelihood, Brier score, accuracy, and calibration-ready
`choice`/`probability_choice_1` columns.

> Note: held-out **generative/rollout post-training analysis** (the
> `generative_analysis` path) remains unsupported for multisubject runs — see
> [README.md](../README.md). The auto fine-tuning above is the supported way to get
> held-out *likelihood* numbers for multisubject models.

---

## 9. Testing

`unittest` suites live in `tests/`. Run them with `code/` on the path:
```bash
cd code
PYTHONPATH=/root/capsule/code python -m unittest tests.test_gru_trainer -v
# or several:
PYTHONPATH=/root/capsule/code python -m unittest \
  tests.test_disrnn_trainer tests.test_heldout_finetuning tests.test_run_helpers
```
Key suites: `test_gru_trainer`, `test_disrnn_trainer`, `test_disrnn_distillation`,
`test_heldout_finetuning`, `test_multisubject_utils`, `test_run_helpers`, plus the
post-training-analysis suites. Suites self-skip if `jax`/`haiku`/
`disentangled_rnns` aren't importable — confirm those import
before trusting a green result.

**Known environment-dependent failures** (present on a clean checkout, not caused
by app code): some disRNN multisubject training tests fail with
`NaN value for non-ignored trial`, and some analysis tests need a
`ex_model_dir-*` fixture that isn't present in every environment. Treat the
*delta* against a baseline run, not absolute green, when working in such an
environment.

---

## 10. Extending the codebase

- **New model trainer:** subclass `BaseMultisubjectTrainer` (if multisubject/
  session-conditioning applies) or `base.interfaces.ModelTrainer`; implement
  `fit(bundle, loggers) -> dict`; set the hook methods; add a `config_<model>.yaml`
  and a `tests/test_<model>_trainer.py`. Reuse the base snapshot/plot/eval pipeline
  rather than re-implementing it.
- **New data loader:** subclass `base.interfaces.DatasetLoader`; return a
  `DatasetBundle` (raw df, train_set, eval_set, metadata); add a `data:` preset.
- **New config key:** read it with `getattr(cfg, "key", default)` so older configs
  stay valid, and document it inline in the config templates.
- **Always:** add/extend a test, run the affected suites, and **update this file +
  the Changelog**.

---

## Changelog

> Add a dated entry (newest first) whenever you add or change a feature.

### 2026-09-07
- **Frozen-random GRU reservoir ablation.** Added the opt-in
  `model.training.freeze_gru_core` switch. It leaves initialization, batching,
  checkpointing, and held-out embedding adaptation unchanged while restricting
  source training to subject embeddings and the final choice readout.

### 2026-09-04
- **External target transfer and matched neural/Q evaluation.** Held-out
  fine-tuning now accepts external `DatasetBundle` targets, supports session- and
  prefix-trial few-shot budgets without changing test membership, prohibits
  target-test checkpoint selection, and exports a shared trial-level prediction
  and metric contract. Prefix/suffix Q evaluation replays the full sequence with
  fixed fitted parameters and scores only the suffix.
- **Canonical external-bandit loader and explicit session manifests.**
  `ExternalBanditDatasetLoader` validates binary two-arm trial tables, checks
  dataset/species provenance, and builds multisubject bundles from a versioned
  JSON manifest whose adaptation/test sessions are exhaustive and disjoint.
  Existing AIND loaders retain their interleaved `eval_every_n` behavior.

### 2026-09-03
- **`beta_max` is recorded in the fit artifact's meta.** Per-session parameters
  reconstructed offline from `theta_raw` invert `phi(theta[..., 3]) * beta_max`, and
  `beta_max` is a config key a rung can override away from the published 10.0. Unrecorded,
  an offline replay would silently rescale `softmax_inverse_temperature`. Paired with
  keeping `theta_raw` in `save_fit`'s session keep list
  (`aind-dynamic-foraging-models`), without which the one_stage estimator persisted no
  session-level parameters at all.
- **HB fits now persist the session level, by default.** `save_fit`'s
  `include_session_sites` defaults to off and `HBTrainer` never overrode it, so no HB fit
  ever written had a session level — and a site exists only while the sampler does, so
  those fits can be recovered only by re-fitting, at hours each. New `HBTrainer`
  argument `save_session_sites` (default **True**), threaded to
  `save_fit(include_session_sites=...)` and set explicitly in the dispatcher's
  `code/config/model/hb_hattori.yaml`. `HBTrainer.__init__` absorbs unknown keyword
  arguments, so a misnamed config key would have been accepted in silence; the guard is
  `tests/test_hb_trainer.py::TestSessionSitePersistence`, which stubs `save_fit` and
  asserts the value the callee actually received (both True by default and False when
  turned off) rather than that the key parses.
- **What that does *not* yet buy: per-session latent trajectories.** Verified against a
  sampler run, not the site names: `hattori2019_three_level` — the `one_stage` estimator,
  i.e. every production rung — registers exactly one session-level site,
  `session_log_lik`. The five per-session parameters that `artifacts.SESSION_SITES` names
  (`learn_rate_rew`, `learn_rate_unrew`, `forget_rate_unchosen`,
  `softmax_inverse_temperature`, `bias_l`) are sites only in the *two-level* model, and
  `theta_raw` — shape draws x subjects x sessions x 5, the offsets those parameters are
  computed from together with the kept `mu_p` and `log_sigma` — is in no keep list. So
  turning this knob on makes a fit re-scorable and WAIC/PSIS-LOO-able, but an offline
  decision-variable replay additionally needs `save_fit`'s keep list widened in
  `aind-dynamic-foraging-models`. Tracked on dispatcher #115; the 2026-09-03 note below
  that says trajectories need "the session-level sites that `save_fit` excludes by
  default" understates it by that one array.
- **HB runs now log figures.** Before this, an HB fit produced numbers and no pictures:
  `hb_trainer.py` had no plotting code at all, and the three plotting helpers in
  `aind_dynamic_foraging_models.hierarchical_bayes.plotting` were imported by nothing in
  either repo. A run that reports a held-out likelihood but no divergence or convergence
  diagnostic figure
  cannot be judged — those decide whether its number may be quoted. New
  `utils/hb_figures.py`, called at the end of `HBTrainer.fit`, logs three:
  `hb/diagnostics` (population traces, plus rank and energy when the sampler supports
  them), `hb/population_posterior` (per-parameter densities mapped through `to_bounded`
  onto the model's own units, since plotting unconstrained coordinates under a bounded
  parameter's label misstates every value), and `hb/conditioning_curve` (held-out
  likelihood against context sessions k). Figure failures are caught and logged — a
  figure is never worth losing a completed multi-hour fit over.
- **`utils/hb_figures.py` uses no arviz API, deliberately.** The first version of this
  module called `az.from_netcdf`, `az.InferenceData(**groups)`, `idata.groups()`,
  `az.plot_rank` and `az.plot_energy`. Every one of those is wrong on one of arviz's two
  major lines: `InferenceData` is a class on 0.x and an alias for `xarray.DataTree` on 1.x,
  and `.groups` is a method returning bare names on 0.x but a property returning node paths
  on 1.x. The Beaker image resolves 1.3.0, so the module would have failed there — and its
  sibling call in `save_fit` did exactly that, crashing every HB run
  (`aind-dynamic-foraging-models` #64/#68). The module now reads plain
  `xarray.Dataset` objects and draws traces, between-chain rank uniformity and the energy
  distribution directly with matplotlib, so there is no arviz version to be wrong about.
  arviz itself is now pinned to `>=1.0,<2` in the models `[bayes]` extra.
- **`utils/hb_figures.load_fit` reads either artifact layout.** Fits written before
  `aind-dynamic-foraging-models` #64 stored each group at the netCDF root, where
  `az.from_netcdf` returns an *empty* InferenceData without raising; the loader detects
  the empty result and reads the groups directly, so figures can be regenerated from
  archived fits. Verified against the D10 fit from Beaker experiment
  `01M1JWY7VNB9F73BS14679K78Q`.
- **Not included, deliberately.** `plot_shrinkage` needs an *unpooled* per-subject arm to
  make its point (its own docstring says so) and the HB run produces none, so a faithful
  version needs a per-subject MLE pass alongside — omitted rather than shipped degenerate.
  Per-session latent trajectories need the session-level sites that `save_fit` excludes by
  default plus a replay of the Q recursion; `hattori2019_choice_prob` discards Q as the
  scan carry, and the hot likelihood is not reshaped to produce a figure. Both are tracked
  in wrapper #82. Comparator reference lines are not passed from the trainer: those are
  study constants and belong to the study's analysis, not to runtime code that would carry
  them stale.

### 2026-08-22
- **New optional inputs: previous-trial reaction time + lick counts**
  (`data.timing_features`, off by default → base model bit-identical). See §6
  "Optional timing inputs". `utils/trial_timing_features.py` derives reaction time
  by COALESCE-ing the two NWB readers' different columns and counts licks from the
  **event** table (co_asset's trial-table lick arrays are unpopulated;
  event-derived counts cross-validate exactly against bonsai's arrays). `obs_size`
  widens automatically 2 → 5; no MODEL block change. Verified on real data: base
  feature columns and targets stay bit-identical with the switch off, and
  previous-trial features come from the previous *retained* trial (no leakage
  across excluded ignore trials).
- **Fixed an inherited int64 truncation bug for continuous features.**
  `data_loaders.disrnn_dataset.create_disrnn_dataset` allocates `xs` via
  `np.full((...), -1)` (int64), truncating float feature columns before the later
  `.astype(float)`. Harmless for the integer stock features, but it collapsed
  24,149 distinct log-RT values to 7 integers. Present in the pinned SHA *and* the
  latest release (0.0.16). `_create_disrnn_dataset()` now routes continuous
  feature sets to a float-safe re-implementation; integer-only sets still call
  upstream, so prior runs stay bit-for-bit reproducible. **Remove the shim when
  upstream is fixed.**
- **`standardize` switch (default true)** for the continuous channels. The disRNN
  bottleneck KL is quadratic in input magnitude, and raw lick counts entered at
  ~87× the KL cost of a binary channel (mean `mus**2` 32.1 vs 0.37); standardizing
  brings all channels to 0.37–1.10 and makes the per-channel σ/openness readouts
  comparable. Constants are fixed and global (not per-run fitted, not per-subject)
  — see §6 for why.
- **New:** `analysis/calibrate_timing_features.py` — session-held-out nested
  logistic probe quantifying the incremental value of the new inputs before
  spending GPU. `tests/test_trial_timing_features.py` covers the transforms, the
  float-safe builder, and the routing predicate.

### 2026-07-04
- **Threshold-free bottleneck-sparsity metrics.** `compute_bottleneck_sparsity_metrics`
  now also logs, per family: `n_eff_open` + `n_eff_open_frac` (normalized participation
  ratio of channel openness `1−σ`, comparable across families of different size),
  `total_openness` (Σ(1−σ)), `sigma_p10/median/p90`, and a multi-threshold `frac_open`
  curve (σ<0.03/0.1/0.5/0.9/0.97). Motivated by the single-threshold `frac_open`
  saturating to 0 (σ<0.1) while the dashboard figure reads "open" (σ<0.97) — see §1.5.
- **Docs: new §1.5 "Run lifecycle & key switches (read first)".** Consolidates the
  four run phases, the two *different* held-out switches
  (`checkpoint_run_heldout_eval` vs `auto_heldout_finetune.enabled`), and
  checkpoints/resumability/extendability upfront, after two sessions misread the
  per-checkpoint-eval skip log line as "no held-out at all."
- **Clarified the held-out skip log messages** in `gru_trainer`, `disrnn_trainer`,
  and `training_runner` to name *which* held-out path is skipped and state that the
  end-of-training `auto_heldout_finetune` still runs (with `heldout/*` metrics).

### 2026-06-18
- **Held-out auto fine-tune mirrors training's logging/plotting cadence.** The
  `auto_heldout_finetune` cadence knobs now default non-zero
  (`checkpoint_every_n_steps: 10`, `checkpoint_plot_split_examples_every_n: 10`) so
  the held-out fine-tune logs a `heldout/train_likelihood`/`heldout/eval_likelihood`
  curve + split-example plots at a controllable interval, and the
  loss/likelihood-over-checkpoints curves are now logged to W&B under
  `heldout/fig/loss_curve` + `heldout/fig/likelihood_curve` (parity with the training
  `fig/validation_loss_curve`). The inner per-step loss logging was coarsened from
  every step to every 10 steps (`log_losses_every=10`) to match training's cadence.
- **baseline_rl fits + reports likelihood on held-out subjects.** Added
  `BaselineRLTrainer.fit_heldout` — re-fits a fresh RL agent per reserved held-out
  subject (train/eval split) and logs aggregate `heldout/train_likelihood` +
  `heldout/eval_likelihood` into the same W&B run (parity with GRU/disRNN), gated by
  `model.heldout_refit.enabled` (default on in `config_baseline_rl.yaml`). Training
  subjects also surface `checkpoint/train_likelihood`/`checkpoint/eval_likelihood`.
  `run_capsule` now drives this for both single- and multi-subject baseline_rl,
  replacing the retired transfer-based held-out eval. Database (`mice_snapshot`)
  loading works for both subject groups.
- **Post-training evaluation reorganized into `evaluation/` + unified CLI.** Eval
  primitives moved from `utils/` into the new `evaluation/` package (transparent
  re-export shims left behind, so training imports are unchanged); shared
  `HeldoutEvalConfig`/helpers split into `evaluation/heldout_eval_config.py` and
  `evaluation/common.py`. Added `run_analysis.py` (one sub-command per analysis) and
  deprecated the `run_capsule-test_*.py` scratch scripts to stubs. New living guide:
  **[POST_TRAINING_ANALYSIS.md](POST_TRAINING_ANALYSIS.md)**. No training-code changes. (commit `73fd4c2`)

### 2026-06-17
- **Automatic multisubject held-out fine-tuning + evaluation.** Added
  `model.training.auto_heldout_finetune` (on by default for GRU/disRNN); at the end
  of a multisubject run, held-out subjects are fine-tuned and their aggregate
  train/eval split likelihoods are logged into the same W&B run under `heldout/*`.
  `run_heldout_subject_finetuning_from_config` gained optional
  `wandb_run`/`wandb_key_prefix`/`wandb_step_offset` (standalone CLI behavior
  unchanged). (commit `f675ca2`)
- **Trainer modularization + bug fixes.** Extracted ~1,900 lines of duplicated
  logic from `gru_trainer.py`/`disrnn_trainer.py` into the new
  `BaseMultisubjectTrainer`; fixed disRNN step validation, empty-loss guards,
  strict logit-count resolution, and the held-out summary key. (commit `6d9b379`)
- **Config documentation.** Rewrote `config_gru.yaml`/`config_disrnn.yaml` with a
  documented comment on every key and corrected stale single-subject presets.
  (commit `29911dc`)
