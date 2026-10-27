# Beaker (AI Hub) image for GLM-HMM fitting (study 14) — GPU.
#
# Same layout and runtime code-pull as beaker/Dockerfile (the GRU / HB image), with two
# differences:
#   - the models package is installed with its [glmhmm] extra (dynamax and one pinned
#     tfp-nightly build) instead of [bayes];
#   - every pip install is constrained by cuda-constraints.txt, which holds cuSOLVER and its
#     dependencies to CUDA 12.8. With the CUDA libraries pip resolves by default, GPU linear
#     solves fail, and the GLM-HMM's Newton M-step needs them (wrapper issue #111).
# A separate image keeps that pin away from the GRU image, which does no linear solves.
#
# Build (from the repo root; the Dockerfile clones the repos itself, so nothing else is
# needed):
#
#   bash beaker/build_and_push.sh --dockerfile beaker/glmhmm/Dockerfile \
#     --name dynamic-foraging-bfm-glmhmm-<YYYYMMDD> --ref <full SHA or main>
#
# FORAGING_MODELS_REF must have the [glmhmm] extra, and WRAPPER_REF must have GLMHMMTrainer.
# pip does not fail on an unknown extra: it exits 0 without it, and the image then dies on
# `import dynamax` at job time. Build only from refs where both exist.
#
# The compute device is chosen per job (AIND_GLMHMM_DEVICE in the experiment spec), not here.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ARG GH_ORG=https://github.com/AllenNeuralDynamics
ARG DISPATCHER_REF=main
ARG WRAPPER_REF=main
ARG FORAGING_MODELS_REF=main

COPY cuda-constraints.txt /workspace/cuda-constraints.txt

# As in beaker/Dockerfile: the baked code only seeds the repos and installs their
# dependencies; beaker/entrypoint.sh re-pulls all three to the requested refs at job start.
# CACHEBUST forces a fresh clone when build_and_push.sh --force-rebuild is passed.
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
    pip install --no-cache-dir -c /workspace/cuda-constraints.txt \
      -e "/workspace/aind-dynamic-foraging-bfm-wrapper[gpu]" && \
    pip install --no-cache-dir -c /workspace/cuda-constraints.txt \
      -e "/workspace/aind-dynamic-foraging-models[glmhmm]"

WORKDIR /workspace/aind-dynamic-foraging-bfm-wrapper/code
ENV PYTHONPATH=/workspace/aind-dynamic-foraging-bfm-wrapper/code
