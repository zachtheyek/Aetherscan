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
#     ./utils/run_container.sh python -m aetherscan.main train --save-tag test_v1
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
#     SLACK_BOT_TOKEN           Slack bot token, forwarded into the container
#     SLACK_CHANNEL             Slack channel, forwarded into the container
#
# <repo>/.env is auto-loaded if present, so secrets set by "source .env" in the
# user's shell survive the trip into this child process. Anything already in our
# exec env (inline VAR=val invocation, real exports) takes precedence over the
# corresponding .env value.

set -euo pipefail

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
exec "$RUNTIME" exec --nv \
    --bind "$REPO:/workspace/aetherscan" \
    --bind "$DATA_PATH:$DATA_PATH" \
    --bind "$MODEL_PATH:$MODEL_PATH" \
    --bind "$OUTPUT_PATH:$OUTPUT_PATH" \
    --pwd /workspace/aetherscan \
    --env PYTHONPATH=/workspace/aetherscan/src \
    --env AETHERSCAN_DATA_PATH="$DATA_PATH" \
    --env AETHERSCAN_MODEL_PATH="$MODEL_PATH" \
    --env AETHERSCAN_OUTPUT_PATH="$OUTPUT_PATH" \
    --env SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}" \
    --env SLACK_CHANNEL="${SLACK_CHANNEL:-}" \
    "$SIF" "$@"
