#!/usr/bin/env bash
# Runtime entrypoint for the disRNN Beaker image.
#
# Runs at CONTAINER STARTUP on every Beaker job (NOT at image build time): it
# refreshes all three repos to the requested code on GitHub *before* running the job, so
# code/config edits take effect by just launching a new run — no image rebuild.
#
# Invoked from the Beaker spec's `command`, e.g.:
#   command: [bash, /workspace/aind-dynamic-foraging-bfm-wrapper/beaker/entrypoint.sh,
#             wandb, agent, --count, "1", "<SWEEP_ID>"]
# Everything after the script path is the real work, exec'd once code is fresh.
#
# WHEN YOU STILL NEED A REBUILD: only when DEPENDENCIES change (pyproject.toml /
# the pinned git deps). Plain code or Hydra-config edits never need one. If a run
# fails with ImportError/ModuleNotFound after a pull, that's the signal to rebuild.
#
# Pin a run for reproducibility by passing commit SHAs instead of branches:
#   envVars: [{name: WRAPPER_REF, value: <sha>},
#             {name: DISPATCHER_REF, value: <sha>},
#             {name: FORAGING_MODELS_REF, value: <sha>}]
set -euo pipefail

WRAPPER_REF="${WRAPPER_REF:-main}"
DISPATCHER_REF="${DISPATCHER_REF:-main}"
FORAGING_MODELS_REF="${FORAGING_MODELS_REF:-main}"
# Where the image put the three repos: /workspace on Beaker, /opt/bfm on Cedar
# (cedar/Dockerfile sets it), because Cedar mounts team storage over /workspace.
BFM_ROOT="${BFM_ROOT:-/workspace}"

refresh() {  # <repo dir> <ref> <commit env var>
    local dir="$1" ref="$2" commit_env="$3" commit
    # --depth 1 keeps the shallow clone shallow; works for a branch, tag, or
    # (GitHub allows) a specific commit SHA. Detach at the fetched commit.
    git -C "$dir" fetch --depth 1 origin "$ref"
    git -C "$dir" checkout -f --detach FETCH_HEAD
    commit="$(git -C "$dir" rev-parse HEAD)"
    printf -v "$commit_env" '%s' "$commit"
    export "$commit_env"
    echo "[entrypoint]   $(basename "$dir") @ ${ref} -> ${commit:0:7}"
}

echo "[entrypoint] refreshing source from GitHub before the run..."
refresh "$BFM_ROOT/aind-dynamic-foraging-bfm-dispatcher"          "$DISPATCHER_REF"       DISPATCHER_COMMIT
refresh "$BFM_ROOT/aind-dynamic-foraging-models"    "$FORAGING_MODELS_REF"  FORAGING_MODELS_COMMIT
refresh "$BFM_ROOT/aind-dynamic-foraging-bfm-wrapper"             "$WRAPPER_REF"          WRAPPER_COMMIT

if [ "$#" -eq 0 ]; then
    echo "[entrypoint] no command given; nothing to run." >&2
    exit 1
fi

cd "$BFM_ROOT/aind-dynamic-foraging-bfm-wrapper/code"
echo "[entrypoint] launching: $*"
exec "$@"
