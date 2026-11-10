#!/usr/bin/env bash
# Smoke-test a Cedar image in a running session: the same checks as beaker/smoke.yaml
# (GPU + bayes imports + Hydra + first training steps, no W&B), plus a database read.
# It fails on a session without a GPU: jax.devices("gpu") raises when JAX finds none.
#
# Run it in a session started from the image, e.g. from your computer:
#   ssh cedar-<workspace>-<session> 'bash /opt/bfm/aind-dynamic-foraging-bfm-wrapper/cedar/smoke.sh'
#
# It goes through beaker/entrypoint.sh, pinned to the commits already in the image, so it
# also checks that the session user can `git fetch` the repos without changing the code.
set -euo pipefail

BFM_ROOT="${BFM_ROOT:-/opt/bfm}"
head_of() { git -C "$BFM_ROOT/$1" rev-parse HEAD; }
export WRAPPER_REF="${WRAPPER_REF:-$(head_of aind-dynamic-foraging-bfm-wrapper)}"
export DISPATCHER_REF="${DISPATCHER_REF:-$(head_of aind-dynamic-foraging-bfm-dispatcher)}"
export FORAGING_MODELS_REF="${FORAGING_MODELS_REF:-$(head_of aind-dynamic-foraging-models)}"
export BFM_OUTPUT_DIR="${BFM_OUTPUT_DIR:-/scratch/bfm-smoke}"
export WANDB_MODE=disabled

exec bash "$BFM_ROOT/aind-dynamic-foraging-bfm-wrapper/beaker/entrypoint.sh" bash -c '
set -e
python -c "import jax; print(\"JAX GPU devices:\", jax.devices(\"gpu\"))"
python -c "import aind_dynamic_foraging_models as m; print(\"foraging models:\", m.__version__)"
python -c "import numpyro, arviz; print(\"bayes extra: numpyro\", numpyro.__version__, \"arviz\", arviz.__version__)"
python -c "from aind_dynamic_foraging_models import hierarchical_bayes; print(\"hierarchical_bayes import: OK\")"
python -m run_hpc --cfg job data=synthetic model=disrnn > /dev/null
echo "hydra compose: OK"
python -m run_hpc data=synthetic_hierarchical model=disrnn \
  data.num_subjects=4 \
  data.num_sessions_per_subject=4 \
  data.num_trials=50 \
  data.batch_size=4 \
  model.training.n_steps=2 \
  model.training.n_warmup_steps=1 \
  model.training.checkpoint_every_n_steps=1 \
  model.training.checkpoint_run_heldout_eval=false \
  model.training.plot_choice_rule=false \
  model.training.plot_update_rules=false \
  model.training.plot_session_context_state_space=false \
  model.training.auto_heldout_finetune.enabled=false
python -c "import aind_dynamic_foraging_database as db; s = db.select_sessions(snapshot=\"20260603\"); print(\"database snapshot 20260603:\", len(s), \"sessions\")"
echo "SMOKE OK"
'
