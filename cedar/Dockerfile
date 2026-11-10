# Cedar session image for the dynamic-foraging-bfm wrapper — GPU.
#
# Same software as beaker/Dockerfile (dispatcher + wrapper[gpu] + models[bayes], cloned from
# GitHub), rebuilt on Cedar's own session image so Cedar can start it. Build and push it with
# cedar/build_and_push.sh, which pulls the base as `base:local` for linux/amd64.
#
# What a Cedar image must keep (Cedar docs, "Extend an existing image"):
#   - The base image's startup command runs the SSH server and logs you in as `cedar`. So:
#     no ENTRYPOINT, no CMD, and no USER line at the end.
#   - Cedar mounts storage over /workspace, /home, /homedir, /scratch and /output, which
#     hides whatever the image put there. So the repos live in /opt/bfm, not /workspace.
#   - Ubuntu 24.04 refuses `pip install` into the system Python, so packages go into a
#     virtual environment at /opt/env, first on PATH. Cedar passes the image's environment
#     to the login shell, so `python` in a session is this one.
#
# The repos and the venv belong to `cedar` (uid 1001 in the base image), so that
# beaker/entrypoint.sh can `git fetch` the requested refs at run time, and a quick
# `pip install` in a session works.
#
# jax comes from the wrapper's [gpu] extra (pinned at 0.6.1 there) and ships the CUDA runtime
# as pip wheels; Cedar's GPU nodes provide the driver. FORAGING_MODELS_REF must point at a
# ref that has the [bayes] extra (see the note in beaker/Dockerfile).

FROM base:local

# The base image's tag and digest, recorded on the image because `latest` moves.
ARG CEDAR_BASE=unknown
LABEL org.alleninstitute.cedar-base="${CEDAR_BASE}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1 \
    BFM_ROOT=/opt/bfm \
    PATH=/opt/env/bin:${PATH} \
    PYTHONPATH=/opt/bfm/aind-dynamic-foraging-bfm-wrapper/code

# Which refs to bake in. build_and_push.sh passes full commit SHAs.
ARG GH_ORG=https://github.com/AllenNeuralDynamics
ARG DISPATCHER_REF=main
ARG WRAPPER_REF=main
ARG FORAGING_MODELS_REF=main

# CACHEBUST forces a fresh clone + reinstall; see beaker/Dockerfile.
ARG CACHEBUST=0
RUN clone_ref() { \
      repo="$1"; dir="$2"; ref="$3"; \
      git clone --depth 1 "$repo" "$dir"; \
      git -C "$dir" fetch --depth 1 origin "$ref"; \
      git -C "$dir" checkout -f --detach FETCH_HEAD; \
    }; \
    echo "cachebust=$CACHEBUST" && \
    clone_ref "$GH_ORG/aind-dynamic-foraging-bfm-dispatcher.git" \
      /opt/bfm/aind-dynamic-foraging-bfm-dispatcher "$DISPATCHER_REF" && \
    clone_ref "$GH_ORG/aind-dynamic-foraging-models.git" \
      /opt/bfm/aind-dynamic-foraging-models "$FORAGING_MODELS_REF" && \
    clone_ref "$GH_ORG/aind-dynamic-foraging-bfm-wrapper.git" \
      /opt/bfm/aind-dynamic-foraging-bfm-wrapper "$WRAPPER_REF" && \
    python3 -m venv /opt/env && \
    pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -e "/opt/bfm/aind-dynamic-foraging-bfm-wrapper[gpu]" && \
    pip install --no-cache-dir -e "/opt/bfm/aind-dynamic-foraging-models[bayes]" && \
    chown -R cedar:cedar /opt/bfm /opt/env
