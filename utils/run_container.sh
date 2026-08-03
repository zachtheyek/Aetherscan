#!/usr/bin/env bash
# Aetherscan container wrapper.
#
# Auto-detects whether `apptainer` (e.g. Apptainer 1.4.5 on the Ampere cluster)
# or `singularity` (e.g. SingularityCE 4.1.1 on the Blackwell cluster) is the
# available runtime (Apptainer preferred when both present), then runs the
# requested command inside the prebuilt .sif image with GPU passthrough and the
# standard bind mounts.
#
# The two runtimes are CLI-compatible for the flags used here (exec, --nv,
# --bind, --pwd, --env), and both consume the same aetherscan.def recipe, so
# either cluster can build and run the image without code changes.
#
# Usage:
#     ./utils/run_container.sh python -m aetherscan.main train --save-tag test
#     ./utils/run_container.sh python -m aetherscan.main inference --inference-files <csv>
#
# Override via env var:
#     SIF                       Path to the .sif image
#                               (default: <repo>/aetherscan-ngc25.02.sif)
#     AETHERSCAN_DATA_PATH      Host data dir, bound 1:1
#                               (default: /datax/scratch/zachy/data/aetherscan)
#     AETHERSCAN_MODEL_PATH     Host models dir, bound 1:1
#                               (default: /datax/scratch/zachy/models/aetherscan)
#     AETHERSCAN_OUTPUT_PATH    Host outputs dir, bound 1:1
#                               (default: /datax/scratch/zachy/outputs/aetherscan)
#     AETHERSCAN_EXTRA_BINDS    Comma-separated extra host paths, each bound 1:1
#                               and appended to the standard bind list, e.g.
#                               AETHERSCAN_EXTRA_BINDS=/datag for inference
#                               (default: none)
#     SLACK_BOT_TOKEN           Slack bot token, forwarded into the container
#     SLACK_CHANNEL             Slack channel, forwarded into the container
#     HF_TOKEN                  HuggingFace token (write access) for --hf-upload,
#                               forwarded into the container; never logged
#     HF_HOME                   HuggingFace cache home. If set, must be an existing ABSOLUTE
#                               directory; it is bound 1:1 and forwarded so downloaded weights
#                               persist there (e.g. on scratch) instead of filling $HOME.
#                               Unset -> container default (~/.cache/huggingface).
#
# Note, the runtime's native SINGULARITY_BIND / APPTAINER_BIND env vars still pass
# through untouched and are additive with the binds set up from AETHERSCAN_EXTRA_BINDS.
#
# <repo>/.env is auto-loaded if present, so secrets set by "source .env" in the
# user's shell survive the trip into this child process. Anything already in our
# exec env (inline VAR=val invocation, real exports) takes precedence over the
# corresponding .env value.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    echo "  e.g. $0 python -m aetherscan.main train --save-tag test" >&2
    exit 1
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Load <repo>/.env without clobbering anything already exported in our env.
# Plain "source .env" with set -a would also work, but it overrides existing
# values; we want inline VAR=val to win.
if [[ -f "$REPO/.env" ]]; then
    while IFS= read -r line || [[ -n $line ]]; do
        # Strip trailing CR (Windows line endings), skip blanks + comments.
        line=${line%$'\r'}
        [[ -z $line || $line =~ ^[[:space:]]*# ]] && continue
        # Split on the first '='. Keys without '=' are malformed; skip them.
        [[ $line != *=* ]] && continue
        key=${line%%=*}
        value=${line#*=}
        [[ -z $key ]] && continue
        # ${!key+x} is non-empty iff $key is already set in our env (even if
        # the existing value is empty) — preserves inline + export precedence.
        [[ -z ${!key+x} ]] && export "$key=$value"
    done <"$REPO/.env"
fi

SIF=${SIF:-$REPO/aetherscan-ngc25.02.sif}

# Runtime detection. Prefer apptainer when both are installed (more active
# upstream), but transparently fall back to singularity (SingularityCE) so the
# same wrapper works on either cluster.
if command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
else
    echo "Error: neither 'apptainer' nor 'singularity' found in PATH." >&2
    echo "Install Apptainer (https://apptainer.org) or SingularityCE (https://sylabs.io/singularity)." >&2
    exit 1
fi

if [[ ! -f "$SIF" ]]; then
    echo "Error: container image not found at $SIF" >&2
    echo "Build it from the repo root with:" >&2
    echo "    $RUNTIME build aetherscan-ngc25.02.sif aetherscan.def" >&2
    exit 1
fi

DATA_PATH=${AETHERSCAN_DATA_PATH:-/datax/scratch/zachy/data/aetherscan}
MODEL_PATH=${AETHERSCAN_MODEL_PATH:-/datax/scratch/zachy/models/aetherscan}
OUTPUT_PATH=${AETHERSCAN_OUTPUT_PATH:-/datax/scratch/zachy/outputs/aetherscan}

# Each of the three data dirs gets a 1:1 bind (host path == container path) so
# absolute paths persisted in the DB / config snapshots stay valid across both
# host and container processes.
BIND_ARGS=(
    --bind "$REPO:/workspace/aetherscan"
    --bind "$DATA_PATH:$DATA_PATH"
    --bind "$MODEL_PATH:$MODEL_PATH"
    --bind "$OUTPUT_PATH:$OUTPUT_PATH"
)

# Extra host paths (comma-separated), each bound 1:1, for data that lives
# outside the standard dirs (e.g. raw .h5 files in /datag during inference).
if [[ -n ${AETHERSCAN_EXTRA_BINDS:-} ]]; then
    IFS=',' read -ra EXTRA_BINDS <<<"$AETHERSCAN_EXTRA_BINDS"
    for extra_path in "${EXTRA_BINDS[@]}"; do
        [[ -z $extra_path ]] && continue
        BIND_ARGS+=(--bind "$extra_path:$extra_path")
    done
fi

# HuggingFace cache home (optional): when HF_HOME is set — e.g. to a scratch dir so
# downloaded weights persist across runs and don't fill $HOME — bind it 1:1 and forward
# it so the container's HF cache lands there. Unset (the off-cluster default) adds nothing,
# and HuggingFace falls back to ~/.cache/huggingface inside the container. Bound separately
# from AETHERSCAN_EXTRA_BINDS so an inline EXTRA_BINDS (e.g. =/datag) can't clobber it.
HF_ENV_ARGS=()
# Test SET (not non-empty): a set-but-empty HF_HOME still forwards to the container by default
# (no --cleanenv), where HuggingFace resolves "" to a RELATIVE hub/ under --pwd — a silent
# download into the repo worktree. Routing empty into the error below makes it actionable.
if [[ -n ${HF_HOME+x} ]]; then
    # Fail fast with guidance: apptainer/singularity won't create a bind source, so a missing,
    # empty, or relative HF_HOME would abort EVERY wrapper invocation (train included) with a
    # cryptic mount FATAL — and HF_HOME is typically a global ~/.bashrc export.
    if [[ $HF_HOME != /* || ! -d $HF_HOME ]]; then
        echo "Error: HF_HOME=$HF_HOME must be an existing absolute directory (it is bound 1:1 into the container)." >&2
        echo "  Create it: mkdir -p \"$HF_HOME\"   (or unset HF_HOME to use the container's ~/.cache/huggingface)." >&2
        exit 1
    fi
    BIND_ARGS+=(--bind "$HF_HOME:$HF_HOME")
    HF_ENV_ARGS+=(--env "HF_HOME=$HF_HOME")
fi

exec "$RUNTIME" exec --nv \
    "${BIND_ARGS[@]}" \
    --pwd /workspace/aetherscan \
    --env PYTHONPATH=/workspace/aetherscan/src \
    --env AETHERSCAN_DATA_PATH="$DATA_PATH" \
    --env AETHERSCAN_MODEL_PATH="$MODEL_PATH" \
    --env AETHERSCAN_OUTPUT_PATH="$OUTPUT_PATH" \
    --env SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}" \
    --env SLACK_CHANNEL="${SLACK_CHANNEL:-}" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    ${HF_ENV_ARGS[@]+"${HF_ENV_ARGS[@]}"} \
    "$SIF" "$@"
