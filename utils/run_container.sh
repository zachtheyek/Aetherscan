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
#                               (default: <repo>/aetherscan-ngc25.02.sif). If it doesn't exist
#                               it's pulled from GHCR (see below); if the pull fails the wrapper
#                               prints aetherscan.def build instructions and exits.
#     AETHERSCAN_IMAGE          GHCR image repo to pull when the .sif is absent
#                               (default: ghcr.io/zachtheyek/aetherscan)
#     AETHERSCAN_IMAGE_TAG      Image tag to pull (default: v<pyproject version>, or `latest`
#                               on a .devN checkout that has no per-version image)
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

# Image acquisition, in priority order:
#   1. Use the local .sif at $SIF if it exists AND still matches the wanted tag (zero-cost).
#   2. Else pull the release-pinned OCI image from GHCR into $SIF (one-time download, cached;
#      the runtime converts docker://… into its own native .sif, so no fork-specific artifact).
#   3. Else fail loudly with build instructions.
#
# The pulled tag defaults to v<pyproject version>, so a checkout of a release tag (vX.Y.Z) pulls
# that release's image; a .devN checkout has no per-version image and falls back to :latest. A
# PULLED image records its tag in "$SIF.pulled-tag"; if a later checkout wants a different tag
# (e.g. a version bump that changed the image), we re-pull instead of silently running the old
# one. A user-BUILT .sif has no sidecar and is always used as-is.
#
# GHCR-pull caveats — the published image is single-arch linux/amd64 on the pinned NGC base.
# If any of these hold, BUILD from aetherscan.def instead of pulling:
#   - non-x86_64 host (e.g. aarch64 Grace/GH200): no matching image exists;
#   - host driver below the base's CUDA 12.8 floor (Blackwell <570 / Ampere <550): a pull would
#     succeed but the container won't see the GPUs — upgrade the driver, or build;
#   - you rebuilt TF from source or edited requirements-container.txt locally: a pull fetches the
#     released image, not your variant — build locally (or set AETHERSCAN_IMAGE_TAG).
IMAGE_REPO=${AETHERSCAN_IMAGE:-ghcr.io/zachtheyek/aetherscan}
if [[ -z ${AETHERSCAN_IMAGE_TAG:-} ]]; then
    # First `version = "..."` line in pyproject.toml; awk (no pipe, portable GNU/BSD) so
    # `set -o pipefail` can't trip the wrapper. `|| true` (+ 2>/dev/null) so a missing file yields
    # an empty VER (-> `latest`) rather than aborting the whole wrapper under `set -e`.
    VER=$(awk -F'"' '/^version = /{print $2; exit}' "$REPO/pyproject.toml" 2>/dev/null || true)
    if [[ $VER =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        AETHERSCAN_IMAGE_TAG="v$VER"
    else
        AETHERSCAN_IMAGE_TAG="latest"
    fi
fi
IMAGE_REF="$IMAGE_REPO:$AETHERSCAN_IMAGE_TAG"

# (Re)pull if there's no image, or the cached one was pulled for a different tag. A $SIF NEWER than
# its sidecar was rebuilt locally over a previously pulled image — treat that as user-built
# (priority 1) and never clobber it (mv preserves the pulled file's mtime; the sidecar is written
# right after, so a pulled .sif is never newer than its sidecar, a local build always is).
need_pull=0
if [[ ! -f "$SIF" ]]; then
    need_pull=1
elif [[ -f "$SIF.pulled-tag" && ! "$SIF" -nt "$SIF.pulled-tag" \
        && "$(cat "$SIF.pulled-tag" 2>/dev/null)" != "$AETHERSCAN_IMAGE_TAG" ]]; then
    echo "Cached $SIF was pulled for '$(cat "$SIF.pulled-tag" 2>/dev/null)' but this checkout wants" \
         "'$AETHERSCAN_IMAGE_TAG' — re-pulling." >&2
    need_pull=1
fi

STALE_SIF_WARNING=""
if [[ $need_pull -eq 1 ]]; then
    echo "Pulling docker://$IMAGE_REF -> $SIF ..." >&2
    tmp="$SIF.pulling.$$"
    # Don't leave a multi-GB partial behind on Ctrl-C. INT/TERM exit explicitly: a trapped signal
    # does NOT terminate bash, so without the exit execution would resume in the failure branch
    # below and (with a stale $SIF present) run the job against the old image.
    trap 'rm -f "$tmp"' EXIT
    trap 'rm -f "$tmp"; exit 130' INT
    trap 'rm -f "$tmp"; exit 143' TERM
    if "$RUNTIME" pull "$tmp" "docker://$IMAGE_REF" >&2; then
        mv -f "$tmp" "$SIF"
        printf '%s\n' "$AETHERSCAN_IMAGE_TAG" >"$SIF.pulled-tag"
        trap - EXIT INT TERM
        echo "Pulled and cached $SIF ($AETHERSCAN_IMAGE_TAG)." >&2
    else
        trap - EXIT INT TERM
        rm -f "$tmp"
        if [[ -f "$SIF" ]]; then
            STALE_SIF_WARNING="WARNING: pull of docker://$IMAGE_REF failed; running against the cached\
 $SIF, which may be a different version (pulled for '$(cat "$SIF.pulled-tag" 2>/dev/null || echo unknown)')."
            echo "$STALE_SIF_WARNING" >&2
        else
            echo "Error: no local image at $SIF, and pulling docker://$IMAGE_REF failed." >&2
            echo "Build it from the repo root with:" >&2
            echo "    $RUNTIME build $SIF aetherscan.def" >&2
            echo "On a hardened HPC node with a quota'd \$HOME, first redirect APPTAINER_CACHEDIR /" >&2
            echo "APPTAINER_TMPDIR (or the SINGULARITY_* equivalents) to scratch — a pull unpacks the" >&2
            echo "~9 GB image through them; see docs/GPU_RUNTIME_GUIDE.md." >&2
            echo "(Or point SIF=/path/to/existing.sif, or set AETHERSCAN_IMAGE_TAG=<tag>.)" >&2
            exit 1
        fi
    fi
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
        echo "Error: HF_HOME='$HF_HOME' must be an existing absolute directory (it is bound 1:1 into the container)." >&2
        echo "  Create it: mkdir -p \"$HF_HOME\"   (or unset HF_HOME to use the container's ~/.cache/huggingface)." >&2
        exit 1
    fi
    BIND_ARGS+=(--bind "$HF_HOME:$HF_HOME")
    HF_ENV_ARGS+=(--env "HF_HOME=$HF_HOME")
fi

# Repeat the stale-image warning right before launch so it isn't buried far above the run in a log.
[[ -n $STALE_SIF_WARNING ]] && echo "$STALE_SIF_WARNING" >&2

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
