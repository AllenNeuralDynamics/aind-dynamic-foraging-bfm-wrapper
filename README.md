# aind-dynamic-foraging-bfm-wrapper

**The training runtime of the `dynamic-foraging-bfm` stack** — a behavioural foundation model
for rodent dynamic foraging.

This repo holds the models, data loaders, training loop and post-training analysis. It does
not decide *what* to run or *how* to launch it. Those live in three sibling repositories:

| Repository | Role |
|---|---|
| [`aind-dynamic-foraging-bfm-studies`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies) | What to run and why: one folder per scientific question, with its sweep manifests, launch records, frozen inputs, analysis and reports |
| [`aind-dynamic-foraging-bfm-dispatcher`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher) | How to launch it: shared Hydra configs, the Beaker / SLURM / Code Ocean launchers, launch provenance, the agent skills pack |
| **`aind-dynamic-foraging-bfm-wrapper`** (this repo) | What runs: data loaders, the four model families, the training loop, post-training analysis |
| [`aind-dynamic-foraging-bfm-atlas`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-atlas) | How to read it: a [static site](https://allenneuraldynamics.github.io/aind-dynamic-foraging-bfm-atlas/) tracing every reported number to its launch and commits |

The cognitive models and agents (baseline RL fits, the hierarchical Bayes likelihoods) come
from [`aind-dynamic-foraging-models`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models),
a dependency of this package.

Four model families train here — GRU, disRNN, baseline RL and hierarchical Bayes — behind a
single `ModelTrainer.fit` interface, all reporting the same held-out likelihood.

[![Behavioral foundation model stack](https://raw.githubusercontent.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/main/docs/diagrams/bfm-stack.png)](https://raw.githack.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/main/docs/diagrams/bfm-stack.html)

**[Open the interactive version](https://raw.githack.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/main/docs/diagrams/bfm-stack.html)**
to pan, zoom and trace one relationship; the atlas's
[About page](https://allenneuraldynamics.github.io/aind-dynamic-foraging-bfm-atlas/#/about)
shows it too. This repo is the training-runtime box: `run_hpc` composes the dispatcher's
config, a `DatasetLoader` reads a pinned snapshot, and the trainers report
`heldout/eval_likelihood` to W&B, where the studies repo picks the runs up. Diagram sources
live in the dispatcher's `docs/diagrams/`.

## Where the configs live — read this first

This trips up everyone once. **The config tree is not in this repo** for the routes you will
most likely use:

| Route | Config comes from |
|---|---|
| HPC / Beaker (`run_hpc.py`) | the **dispatcher's** `code/config/`, resolved as a sibling directory at import time |
| Code Ocean (`run_capsule.py`) | injected by the pipeline at `/data/jobs/**/config.yaml`; `code/configs/*.yaml` here are the maintained templates to copy from |

So the dispatcher must be checked out beside this repo. Study launches run from a studies
checkout beside both, and some study analysis scripts import this package:

```text
/path/to/parent/
  aind-dynamic-foraging-bfm-dispatcher/
  aind-dynamic-foraging-bfm-studies/
  aind-dynamic-foraging-bfm-wrapper/        # you are here
```

## Install

Python 3.12.

```bash
cd /path/to/parent/aind-dynamic-foraging-bfm-wrapper

conda create -n dynamic-foraging-bfm-cpu python=3.12 -y
conda activate dynamic-foraging-bfm-cpu
pip install -e .

conda create -n dynamic-foraging-bfm-gpu python=3.12 -y
conda activate dynamic-foraging-bfm-gpu
pip install -e ".[gpu]"          # jax[cuda12]
```

The SLURM scripts activate `dynamic-foraging-bfm-{cpu,gpu}`, falling back to the legacy
`disrnn-{cpu,gpu}` names if the new ones are absent. `.[dev]` adds the formatters and pytest;
`.[stan]` adds PyStan and ArviZ for the Stan arm of the HB baseline (`model=hb_hattori_stan`).

> **Known gap:** `pip install -e .` pulls `aind-dynamic-foraging-models` *without* its
> `[bayes]` extra, so `numpyro`/`arviz` are missing and `model=hb_hattori` will fail at
> import. Only the Beaker image installs `...models[bayes]`. Install it yourself if you need
> to run the hierarchical Bayes baseline locally.

## Entry points

All are run as `python <file>` or `python -m <module>` from `code/` — the package declares no console scripts.

| File | What it is |
|---|---|
| `code/run_hpc.py` | Training entry point for **HPC and Beaker**. Hydra-driven; reads the dispatcher's config tree. |
| `code/run_capsule.py` | Training entry point for **Code Ocean**. Finds the injected Hydra config, then delegates to the same code. |
| `code/training_runner.py` | The shared body both call: `run_training()` — dispatches on `model.type` ∈ `{disrnn, gru, baseline_rl, hb}` and orchestrates held-out evaluation. |
| `code/run_analysis.py` | Post-training analysis CLI (below). |
| `code/resume_heldout.py`, `code/resume_heldout_beaker.py` | Re-score or resume held-out evaluation of a finished run without retraining. |
| `code/run_heldout_subject_finetuning.py` | Fine-tune a trained model's subject embedding on held-out subjects (config: the dispatcher's `config_heldout_subject_finetuning.yaml`). |
| `code/run_embedding_space_analysis.py` | Embedding-space analysis of a saved run. |
| `python -m external_transfer.gru_transfer` | Transfer a trained GRU to an external two-arm-bandit dataset: freeze its core, adapt a new-subject embedding, score the held-out split (study 09). |
| `code/load_mice_data.py` | Offline loader — pulls multi-subject mouse behaviour and saves snapshots to disk. |

Model code sits in `code/models/` and `code/model_trainers/` (`GruTrainer`, `DisrnnTrainer`,
`BaselineRLTrainer`, `HBTrainer`), data loading in `code/data_loaders/`, shared evaluation in
`code/evaluation/`, and the external-dataset transfer runners in `code/external_transfer/`.

## Three ways to run it

Launch orchestration lives in the dispatcher for all three; this repo supplies the payload.

| Route | Launch with | Detail |
|---|---|---|
| **Beaker / AI Hub** | dispatcher's `code/launch_beaker_resumable.py` | [`beaker/README.md`](beaker/README.md) — image plane, GPU benchmarks |
| **Allen SLURM** | dispatcher's `code/launch_hpc.py` | dispatcher's `code/hpc/README.md` |
| **Code Ocean** | the [CO pipeline](https://github.com/AllenNeuralDynamics/aind-disrnn-pipeline) | `.codeocean/` |

On Beaker, **code is not frozen into the image**: `beaker/entrypoint.sh` re-pulls the
wrapper, the dispatcher and `aind-dynamic-foraging-models` at job start, so ordinary code
changes need no rebuild — pin a specific state with `WRAPPER_REF`, `DISPATCHER_REF` and
`FORAGING_MODELS_REF`. Rebuild only when dependencies change.

For a one-off debug run, submit through SLURM and invoke the module directly:

```bash
cd /path/to/aind-dynamic-foraging-bfm-wrapper/code
export PYTHONPATH=/path/to/aind-dynamic-foraging-bfm-wrapper/code:${PYTHONPATH:-}
export BFM_OUTPUT_DIR=$HOME/outputs/dynamic-foraging-bfm
python -m run_hpc job_id=42 data=mice model=disrnn
```

Never run training on the HPC login node — always `srun`/`sbatch`/`salloc` onto a compute
node. Add `--cfg job` to print the composed config and exit without training.

The `BFM_*` runtime variables go through `code/utils/env_config.py`, which rejects other
prefixes: `BFM_OUTPUT_DIR`,
`BFM_RESUMABLE_OUTPUT_DIR`, `BFM_RESTORE_FROM_RUN_ID`, and the `BFM_META_*` provenance set.

## Post-training analysis

`code/run_analysis.py` is a single CLI over a saved run directory:

```bash
python run_analysis.py likelihood-advantage --help
```

Sub-commands: `generative`, `from-histories`, `likelihood-comparison`,
`likelihood-advantage`, `state-space-{condition,subject,overview}`,
`q-space-{condition,subject}`, `embedding-params`, `embedding`, `baseline-rl`,
`finetune`.

It is deliberately two-stage: `likelihood-advantage` does the expensive work once — loads
the models, evaluates, rolls out the baseline agent, extracts hidden states — and writes
`trial_advantage.pkl`; the `state-space-*` and `q-space-*` plots then re-read that pickle
with no model load. Full contract in
[`code/POST_TRAINING_ANALYSIS.md`](code/POST_TRAINING_ANALYSIS.md).

## Reproducibility

Every run traces back to the code and command that launched it:

- **Local artifacts.** Each `run_hpc.py` run copies the dispatcher's `code/config/` tree into
  `inputs/` and writes `inputs.yaml` / `inputs.json`
  into both the run output directory and the W&B run folder — the effective config after
  defaults, includes and overrides.
- **W&B config.** `wandb.init(config=...)` records the `data`, `model` and `meta` blocks, so
  they are filterable and groupable. Beaker runs also stamp `wrapper_commit`,
  `dispatcher_commit` and `foraging_models_commit` after refreshing all three repos.
- **Study identity.** The dispatcher's launchers pass the study, variant, launch id, label,
  note, config hash and the **studies commit that held the sweep** as `BFM_META_*` env on
  Beaker or `+meta.*` overrides on HPC; `utils/run_helpers._launch_meta` merges them into
  `meta`, so every run carries `meta.studies_commit`. Runs without it predate the studies
  split, and their sweep lives at `dispatcher_commit` in the dispatcher.
- **Dispatcher lineage.** At sweep-creation time the dispatcher captures the git state of the
  repos involved and the launch context and appends them as `+meta.*` Hydra overrides, so
  each run carries `meta.dispatcher_git_commit`, `meta.wrapper_git_commit`,
  `meta.sweep_yaml`, `meta.owner`, `meta.launcher_cmd` and `meta.mode`. The launch record
  the dispatcher writes, kept in the study's `launch_record/` (data snapshot, image, all
  four SHAs), is the authoritative copy.

In W&B sweep mode, per-run `wandb.entity`/`wandb.project` overrides are ignored — routing
comes from the top-level `entity` and `project` in the sweep YAML. Commit before launching
so the git lineage identifies the code state uniquely.

## Read next

| For | Read |
|---|---|
| **Interpreting any run's logs or metrics** — start at §1.5 | [`code/TRAINING.md`](code/TRAINING.md) |
| The four run phases, the two *different* held-out switches, checkpoints vs resume vs extend | `code/TRAINING.md` §1.5 |
| Post-training analysis code and its run-directory contract | [`code/POST_TRAINING_ANALYSIS.md`](code/POST_TRAINING_ANALYSIS.md) |
| Building and pinning the Beaker image; GPU efficiency notes | [`beaker/README.md`](beaker/README.md) |
| Working rules for humans and agents | [`AGENTS.md`](AGENTS.md) |
| Vocabulary, launching, studies, provenance | the dispatcher's `CONTEXT.md` and skills pack |
| Study layout, launch-record and report schemas | the studies repo's `CONTEXT.md` and `schemas/` |

## Tests

```bash
pip install -e ".[dev]"
cd code && python -m unittest discover -s tests -t . -v
```

The trainer tests do real (tiny) training runs, so the first pass is dominated by XLA
compilation; `code/tests/__init__.py` sets up a persistent JAX compile cache to make repeat
runs fast. This is what CI runs.

## License

MIT — see [LICENSE](LICENSE). Contributor expectations: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
