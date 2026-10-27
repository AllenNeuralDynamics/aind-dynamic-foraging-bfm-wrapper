# Beaker (AI Hub) image for the disRNN wrapper — GPU.
#
# Clones the dispatcher, wrapper, and dynamic-foraging-models repos from GitHub.
# The dispatcher and wrapper use the sibling layout that run_hpc.py expects:
#   /workspace/aind-dynamic-foraging-bfm-wrapper/code/run_hpc.py reads its Hydra configs from
#   ../../aind-dynamic-foraging-bfm-dispatcher/code/config  ->  /workspace/aind-dynamic-foraging-bfm-dispatcher/...
# so W&B sweep agents (which invoke `python -m run_hpc <overrides>`) can
# re-compose the config on Beaker with no code change.
#
# Because all repos are cloned from git, the build context is irrelevant — you
# can build from anywhere. All repos/deps are public, so no token is needed:
#
#   docker build \
#     --platform linux/amd64 \
#     -f beaker/Dockerfile \
#     -t disrnn-wrapper beaker/        # context only has to exist; nothing is COPYed
#
# jax[cuda12] ships the CUDA runtime as pip wheels, so a slim Python base is
# enough — Beaker nodes provide the GPU driver.
#
# The models package is installed with its [bayes] extra, which adds numpyro, arviz,
# xarray and h5netcdf for the hierarchical Bayesian baseline, and its [glmhmm] extra, which
# adds dynamax and one pinned tfp-nightly build for the GLM-HMM baseline.
#
# FORAGING_MODELS_REF must point at a ref that HAS both extras. pip does not fail on an
# unknown extra -- it exits 0 and installs without it -- so building against a ref where
# [bayes] or [glmhmm] is absent yields an image that looks fine and dies on `import numpyro`
# or `import dynamax` at job time. Build only from a ref where the extras exist.
#
# jax comes from the wrapper's [gpu] extra and is pinned at 0.6.1 there; the models
# package caps numpyro below 0.20 to stay compatible, since 0.20+ requires jax >= 0.7.
# Raising either needs the HB suite re-run, not assumed.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Which refs to bake in. Branches, tags, and full commit SHAs are supported.
ARG GH_ORG=https://github.com/AllenNeuralDynamics
ARG DISPATCHER_REF=main
ARG WRAPPER_REF=main
ARG FORAGING_MODELS_REF=main

# Clone all three repos and install the wrapper + GPU JAX. All repos
# and the wrapper's git deps (disentangled-rnns) are public,
# so no credentials are needed.
#
# The baked code is only a STARTING POINT: at job startup beaker/entrypoint.sh
# re-pulls all three repos to the requested refs (default main), so the image
# rarely needs rebuilding — only when DEPENDENCIES change. The build-time clone exists
# mainly to (a) install deps from pyproject and (b) seed the repos so entrypoint.sh
# can `git fetch` them. See beaker/entrypoint.sh and the README.
#
# For release images, point all *_REF args at full commit SHAs.
#
# CACHEBUST: the clone instruction text is identical across builds, so Docker
# would otherwise reuse a stale cached clone and silently bake old code. Pass a
# changing value (build_and_push.sh sends a timestamp) to force a fresh clone +
# reinstall whenever you intentionally rebuild.
ARG CACHEBUST=0
RUN clone_ref() { \
      repo="$1"; dir="$2"; ref="$3"; \
      git clone --depth 1 "$repo" "$dir"; \
      git -C "$dir" fetch --depth 1 origin "$ref"; \
      git -C "$dir" checkout -f --detach FETCH_HEAD; \
    }; \
    echo "cachebust=$CACHEBUST" && \
    clone_ref "$GH_ORG/aind-dynamic-foraging-bfm-dispatcher.git" \
      /workspace/aind-dynamic-foraging-bfm-dispatcher "$DISPATCHER_REF" && \
    clone_ref "$GH_ORG/aind-dynamic-foraging-models.git" \
      /workspace/aind-dynamic-foraging-models "$FORAGING_MODELS_REF" && \
    clone_ref "$GH_ORG/aind-dynamic-foraging-bfm-wrapper.git" \
      /workspace/aind-dynamic-foraging-bfm-wrapper "$WRAPPER_REF" && \
    pip install --no-cache-dir -e "/workspace/aind-dynamic-foraging-bfm-wrapper[gpu]" && \
    pip install --no-cache-dir -e "/workspace/aind-dynamic-foraging-models[bayes,glmhmm]"

WORKDIR /workspace/aind-dynamic-foraging-bfm-wrapper/code
ENV PYTHONPATH=/workspace/aind-dynamic-foraging-bfm-wrapper/code
