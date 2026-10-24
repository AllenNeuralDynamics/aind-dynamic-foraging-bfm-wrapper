#!/usr/bin/env bash
# Fetch the studies repository at a pinned commit, then run the given command.
#
# For jobs that execute study code at runtime (study 09's run_*.py; dispatcher #179). Since
# the studies split (dispatcher #124) those scripts live in aind-dynamic-foraging-bfm-studies,
# which the image does not contain. Use it as a prefix after entrypoint.sh, which refreshes
# this wrapper first, so the script is always present without an image rebuild:
#
#   command: [bash, /workspace/aind-dynamic-foraging-bfm-wrapper/beaker/entrypoint.sh,
#             bash, /workspace/aind-dynamic-foraging-bfm-wrapper/beaker/with_studies.sh,
#             python, /workspace/aind-dynamic-foraging-bfm-studies/studies/<study>/run_x.py, ...]
#
# STUDIES_REF        required; the dispatcher launchers set it to the launch's studies commit
#                    (a full SHA they have checked exists on GitHub).
# STUDIES_ROOT       checkout location (default /workspace/aind-dynamic-foraging-bfm-studies).
# STUDIES_REPO_URL   clone URL (default the public GitHub repository; tests point it elsewhere).
#
# Exports STUDIES_COMMIT, like entrypoint.sh exports DISPATCHER_COMMIT and friends.
set -euo pipefail

STUDIES_REF="${STUDIES_REF:?STUDIES_REF is required (the launcher sets it to the studies commit)}"
STUDIES_ROOT="${STUDIES_ROOT:-/workspace/aind-dynamic-foraging-bfm-studies}"
STUDIES_REPO_URL="${STUDIES_REPO_URL:-https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies.git}"

if [ ! -d "$STUDIES_ROOT/.git" ]; then
    git init -q "$STUDIES_ROOT"
    git -C "$STUDIES_ROOT" remote add origin "$STUDIES_REPO_URL"
fi
# --depth 1 at a commit SHA works on GitHub; detach at exactly what was fetched.
git -C "$STUDIES_ROOT" fetch -q --depth 1 origin "$STUDIES_REF"
git -C "$STUDIES_ROOT" checkout -q -f --detach FETCH_HEAD
STUDIES_COMMIT="$(git -C "$STUDIES_ROOT" rev-parse HEAD)"
export STUDIES_COMMIT STUDIES_ROOT
echo "[with_studies]   $(basename "$STUDIES_ROOT") @ ${STUDIES_REF} -> ${STUDIES_COMMIT:0:7}"

if [ "$#" -eq 0 ]; then
    echo "[with_studies] no command given; nothing to run." >&2
    exit 1
fi
exec "$@"
