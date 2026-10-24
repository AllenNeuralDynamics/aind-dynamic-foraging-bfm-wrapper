"""Runners that transfer trained models and fit baselines on external two-arm-bandit datasets.

Study 09 (cross-species transfer) needs workflows the generic ``run_hpc`` entrypoint does not
have: start from an already-trained GRU and adapt only a new-subject embedding on an external
dataset, and fit author-selected baselines that are not ``ModelTrainer`` subclasses. These
modules were study scripts until the studies split; they live here so jobs run them from the
wrapper checkout that ``beaker/entrypoint.sh`` already refreshes, with no studies checkout
(dispatcher #179, option 2). Every input arrives on the command line or from the external
dataset cache; none is read from a study directory.

    python -m external_transfer.gru_transfer --dataset chen --source-key d10-s0 \
        --source-manifest-json "$(cat source_runs.json)" --source-manifest-name source_runs.json
    python -m external_transfer.q_baseline --dataset chen --data-root ... --output-root ...
"""
