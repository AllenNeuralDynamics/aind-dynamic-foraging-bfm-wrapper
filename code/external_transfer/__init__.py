"""Transfer a trained multisubject model to an external two-arm-bandit dataset.

``gru_transfer`` starts from an already-trained GRU (a pinned W&B training-output artifact),
freezes its core, adapts only a new-subject embedding on each external subject's adaptation
split, and scores the held-out split. The generic ``run_hpc`` entrypoint has no equivalent:
it trains from scratch on AIND mice. Study 09 (cross-species transfer) is its first user.

It lives in the wrapper, not the study, because it runs inside Beaker jobs, whose runtime
holds the wrapper / dispatcher / models checkouts that ``beaker/entrypoint.sh`` refreshes but
no studies checkout (dispatcher #179). Every input arrives on the command line or from the
external dataset cache; none is read from a study directory. Study-specific author baselines
stay with their study and run on HPC from a studies checkout.

    python -m external_transfer.gru_transfer --dataset chen --source-key d10-s0 \
        --source-manifest-json "$(cat source_runs.json)" --source-manifest-name source_runs.json
"""
