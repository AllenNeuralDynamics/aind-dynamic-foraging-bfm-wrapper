#!/usr/bin/env bash
# Build the dynamic-foraging-bfm wrapper GPU image for Cedar and push it to a Cedar workspace.
#
# Run it on a computer with Docker (a Mac is fine; the build targets linux/amd64), signed in
# to Cedar (`cedar auth whoami`). The Dockerfile git-clones all three repos itself, so you can
# run this from any directory.
#
# Usage:
#   bash cedar/build_and_push.sh [options]
#
# Options:
#   --name NAME             Cedar image name (default: dfbfm-wrapper)
#   --tag TAG               Image tag (default: <YYYYMMDD>-w<wrapper sha7>). Tags are
#                           immutable here: the script stops if the tag already exists.
#   --workspace WS          Cedar workspace (default: neural-dynamics-playground)
#   --base-tag TAG          cedar-session tag to build on (default: latest)
#   --ref REF               Branch/tag/SHA to bake into all repos (default: main)
#   --wrapper-ref REF       Override the wrapper repo ref only
#   --dispatcher-ref REF    Override the dispatcher repo ref only
#   --foraging-models-ref REF
#                           Override the aind-dynamic-foraging-models ref only
#   --latest                Also tag the pushed image as `latest`
#   --force-rebuild         Bust Docker's cache for the clone + install layer. New SHAs
#                           already do that; use this only to re-install the same SHAs.
#   -h, --help              Show this help.
#
# Every ref is resolved to a full commit SHA before the build, so the image always records
# exact code. Add a line to beaker/BUILD_LOG.md for each image you push.
set -euo pipefail

IMAGE_NAME="dfbfm-wrapper"
TAG=""
WORKSPACE="neural-dynamics-playground"
BASE_TAG="latest"
WRAPPER_REF="main"
DISPATCHER_REF="main"
FORAGING_MODELS_REF="main"
PUSH_LATEST=0
FORCE_REBUILD=0
GH_ORG="https://github.com/AllenNeuralDynamics"
LOCAL_IMAGE="dfbfm-wrapper-cedar:local"

usage() { sed -n '8,28p' "$0"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --name)                IMAGE_NAME="$2"; shift 2 ;;
        --tag)                 TAG="$2"; shift 2 ;;
        --workspace)           WORKSPACE="$2"; shift 2 ;;
        --base-tag)            BASE_TAG="$2"; shift 2 ;;
        --ref)                 WRAPPER_REF="$2"; DISPATCHER_REF="$2"; FORAGING_MODELS_REF="$2"; shift 2 ;;
        --wrapper-ref)         WRAPPER_REF="$2"; shift 2 ;;
        --dispatcher-ref)      DISPATCHER_REF="$2"; shift 2 ;;
        --foraging-models-ref) FORAGING_MODELS_REF="$2"; shift 2 ;;
        --latest)              PUSH_LATEST=1; shift ;;
        --force-rebuild)       FORCE_REBUILD=1; shift ;;
        -h|--help)             usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CEDAR="$(command -v cedar || echo "$HOME/.cedar/cedar")"
export CEDAR_NO_UPDATE_CHECK=1

# Resolve a branch or tag to a full SHA; a full SHA passes through unchanged.
resolve() {  # <repo> <ref>
    local repo="$1" ref="$2" sha
    if [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then echo "$ref"; return; fi
    sha="$(git ls-remote "$GH_ORG/$repo.git" "refs/heads/$ref" "refs/tags/$ref^{}" "refs/tags/$ref" \
           | head -1 | cut -f1)"
    if [ -z "$sha" ]; then
        echo "ERROR: $repo has no branch or tag '$ref' (pass a full 40-character SHA instead)" >&2
        exit 1
    fi
    echo "$sha"
}
WRAPPER_SHA="$(resolve aind-dynamic-foraging-bfm-wrapper "$WRAPPER_REF")"
DISPATCHER_SHA="$(resolve aind-dynamic-foraging-bfm-dispatcher "$DISPATCHER_REF")"
FORAGING_MODELS_SHA="$(resolve aind-dynamic-foraging-models "$FORAGING_MODELS_REF")"
TAG="${TAG:-$(date +%Y%m%d)-w${WRAPPER_SHA:0:7}}"

echo "image: $IMAGE_NAME:$TAG   workspace: $WORKSPACE   base: cedar-session:$BASE_TAG"
echo "wrapper=$WRAPPER_SHA ($WRAPPER_REF)"
echo "dispatcher=$DISPATCHER_SHA ($DISPATCHER_REF)"
echo "foraging-models=$FORAGING_MODELS_SHA ($FORAGING_MODELS_REF)"

# Pre-flight: signed in, and the tag is new (a pushed tag is what a run records).
"$CEDAR" auth whoami >/dev/null
tag_exists() {
    "$CEDAR" image show "$IMAGE_NAME" --workspace-name "$WORKSPACE" --query "$TAG" --format json 2>/dev/null \
        | grep -q "\"$TAG\""
}
if tag_exists; then
    echo "ERROR: $IMAGE_NAME:$TAG already exists in $WORKSPACE; choose another --tag" >&2
    exit 1
fi

# Pull the base for linux/amd64. `cedar image pull` runs `docker pull`, which on an Apple
# Silicon Mac would otherwise fetch the arm64 variant; Cedar's nodes are amd64.
DOCKER_DEFAULT_PLATFORM=linux/amd64 \
    "$CEDAR" image pull "cedar-session:$BASE_TAG" base:local --tenant
ARCH="$(docker image inspect base:local --format '{{.Architecture}}')"
if [ "$ARCH" != "amd64" ]; then
    echo "ERROR: base:local is $ARCH, not amd64; Cedar's nodes would not run it" >&2
    exit 1
fi
BASE_DIGEST="$(docker image inspect base:local --format '{{join .RepoDigests " "}}' | cut -d' ' -f1)"

cachebust="$(date +%s)"
[ "$FORCE_REBUILD" -eq 1 ] || cachebust=0
docker build \
    --platform linux/amd64 \
    --build-arg CACHEBUST="$cachebust" \
    --build-arg CEDAR_BASE="cedar-session:$BASE_TAG@${BASE_DIGEST##*@}" \
    --build-arg WRAPPER_REF="$WRAPPER_SHA" \
    --build-arg DISPATCHER_REF="$DISPATCHER_SHA" \
    --build-arg FORAGING_MODELS_REF="$FORAGING_MODELS_SHA" \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$LOCAL_IMAGE" \
    "$SCRIPT_DIR"

latest_arg=()
[ "$PUSH_LATEST" -eq 0 ] || latest_arg=(--latest)
# The ${a[@]+...} form keeps an empty array legal under `set -u` in macOS's bash 3.2.
"$CEDAR" image push "$LOCAL_IMAGE" "$IMAGE_NAME:$TAG" --workspace-name "$WORKSPACE" ${latest_arg[@]+"${latest_arg[@]}"}
if ! tag_exists; then
    echo "ERROR: the push finished but Cedar does not list $IMAGE_NAME:$TAG" >&2
    exit 1
fi
"$CEDAR" image show "$IMAGE_NAME" --workspace-name "$WORKSPACE" --query "$TAG"

cat <<MSG

Done: $IMAGE_NAME:$TAG in workspace $WORKSPACE.
Try it on one A100 before sharing it (pick a free GPU cluster: cedar cluster list).
Add --all-ssh-keys to session create so every computer whose key you saved in Cedar
(cedar ssh-key list) can reach the session, not only this one:

  cedar session create ${IMAGE_NAME}-smoke --workspace-name $WORKSPACE \\
    --cluster-name octo.hub-usc1-gcp-a100 --gpu 1 --cpu 8 --ram 64Gi \\
    --image $IMAGE_NAME:$TAG --max-live-time 60m
  ssh cedar-$WORKSPACE-${IMAGE_NAME}-smoke 'bash /opt/bfm/aind-dynamic-foraging-bfm-wrapper/cedar/smoke.sh'
  cedar session stop ${IMAGE_NAME}-smoke --workspace-name $WORKSPACE
MSG
