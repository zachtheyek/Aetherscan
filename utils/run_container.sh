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
#     AETHERSCAN_IMAGE          GHCR image repo to pull (when the .sif is absent, or a re-pull is
#                               triggered) (default: ghcr.io/zachtheyek/aetherscan)
#     AETHERSCAN_IMAGE_TAG      Image tag to pull (default: v<pyproject version> on a release
#                               checkout; a .devN checkout resolves the newest published release
#                               tag AT OR BELOW its own version via the GHCR API (#424), falling
#                               back to `latest` when the registry is unreachable)
#     AETHERSCAN_FORCE_REPULL   Set to 1 to delete a .sif the wrapper cannot verify (user-built
#                               or manually pulled — no pull-provenance sidecar) and re-pull the
#                               published image in its place (#424). Default: warn and keep it.
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
#   1. Use the local .sif at $SIF if it exists and isn't a pull cached for a different tag (zero-cost).
#   2. Else pull the release-pinned OCI image from GHCR into $SIF (one-time download, cached;
#      the runtime converts docker://… into its own native .sif, so no fork-specific artifact).
#   3. Else fail loudly with build instructions.
#
# The pulled tag defaults to v<pyproject version>, so a checkout of a release tag (vX.Y.Z) pulls
# that release's image; a .devN checkout resolves the newest published release tag AT OR BELOW its
# own version via the GHCR tag API (#424 — strictly below the .dev base: a 1.1.1.dev0 checkout
# wants v1.1.0 even after v1.2.0 publishes, so a checkout never silently runs an image newer than
# its own code), falling back to :latest when the registry/tooling is unavailable. A PULLED image
# records its ref (repo:tag, line 1) and manifest digest (line 2, when obtainable) in
# "$SIF.pulled-tag"; a later checkout wanting a different ref re-pulls, and a matching ref whose
# REMOTE digest has moved (a retag — the #416 drift class) warns, deletes, and re-pulls
# automatically. Both checks fail OPEN: an unreachable registry runs the cached image with a
# warning, never blocks a run that has a cached image (a FIRST pull with no cached image
# still needs the registry, by necessity). A user-BUILT .sif (no sidecar, or newer than its sidecar — see
# the mtime note below) is never verified or clobbered by default: the wrapper warns that it
# cannot vouch for it, and only AETHERSCAN_FORCE_REPULL=1 replaces it with the published image
# (a Blackwell rebuild or edited-requirements build is legitimate and irreplaceable).
#
# GHCR-pull caveats — the published image is single-arch linux/amd64 on the pinned NGC base.
# If any of these hold, BUILD from aetherscan.def instead of pulling:
#   - non-x86_64 host (e.g. aarch64 Grace/GH200): no matching image exists;
#   - host driver below the base's CUDA 12.8 floor (Blackwell <570 / Ampere <550): a pull would
#     succeed but the container won't see the GPUs — upgrade the driver, or build;
#   - you rebuilt TF from source, edited requirements-container.txt locally, or are on a .devN
#     checkout whose requirements-container.txt changed since the last release: a pull fetches the
#     RELEASED image (:latest tracks the newest release, never master HEAD), not your variant —
#     no published tag matches, so build locally (or push your own image and set AETHERSCAN_IMAGE
#     — rm the local $SIF first, or point SIF= elsewhere, since a user-built .sif takes priority
#     over any pull).
IMAGE_REPO=${AETHERSCAN_IMAGE:-ghcr.io/zachtheyek/aetherscan}

# --- GHCR helpers (#424). All fail CLOSED as functions (nonzero return) and every call site
# treats that as "unknown" and fails OPEN — an offline node, a missing curl, or a non-GHCR
# AETHERSCAN_IMAGE never blocks a run that has a cached image. Anonymous pull-scope token
# per SECURITY.md's recipe.
_ghcr_token() {  # $1 = repo path (owner/name); prints a token
    local token
    command -v curl >/dev/null 2>&1 || return 1
    token=$(curl -fsS --max-time 10 "https://ghcr.io/token?scope=repository:$1:pull" 2>/dev/null \
        | sed -n 's/.*"token" *: *"\([^"]*\)".*/\1/p') || return 1
    [[ -n $token ]] || return 1
    printf '%s\n' "$token"
}

_ghcr_ceiling_tag() {  # $1 = repo path, $2 = pyproject version; prints the wanted vX.Y.Z tag
    # Highest published vX.Y.Z at or below the ceiling: inclusive for an exact release (or
    # .post/+local, which FOLLOW their base per PEP 440), strictly below the base for a
    # .dev/rc pre-release (1.1.1.dev0 precedes 1.1.1). Mirrors hf_hub.release_ceiling for
    # the weight side. Returns 1 on infra failure (no curl / registry unreachable), 2 when
    # the registry answered but no tag qualifies under the ceiling — callers distinguish
    # the two so a reachable-but-empty result can warn about its fallback.
    local repo=$1 ver=$2 inclusive major minor patch token tags t best_tag=""
    local bM=-1 bm=-1 bp=-1 M m p keep
    if [[ $ver =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(\.post[0-9]+)?(\+.*)?$ ]]; then
        inclusive=1
    elif [[ $ver =~ ^([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
        inclusive=0
    else
        return 1
    fi
    major=${BASH_REMATCH[1]} minor=${BASH_REMATCH[2]} patch=${BASH_REMATCH[3]}
    token=$(_ghcr_token "$repo") || return 1
    tags=$(curl -fsS --max-time 10 -H "Authorization: Bearer $token" \
        "https://ghcr.io/v2/$repo/tags/list" 2>/dev/null \
        | tr ',[]' '\n' | sed -n 's/.*"\(v[0-9][0-9.]*\)".*/\1/p') || return 1
    while IFS= read -r t; do
        [[ $t =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || continue
        M=${BASH_REMATCH[1]} m=${BASH_REMATCH[2]} p=${BASH_REMATCH[3]}
        keep=0
        if ((M < major)); then keep=1
        elif ((M == major)); then
            if ((m < minor)); then keep=1
            elif ((m == minor)); then
                if ((p < patch)) || { ((p == patch)) && ((inclusive == 1)); }; then keep=1; fi
            fi
        fi
        ((keep == 1)) || continue
        if ((M > bM)) || { ((M == bM)) && ((m > bm)); } \
            || { ((M == bM)) && ((m == bm)) && ((p > bp)); }; then
            bM=$M bm=$m bp=$p best_tag="v$M.$m.$p"
        fi
    done <<<"$tags"
    [[ -n $best_tag ]] || return 2  # registry answered; nothing qualifies under the ceiling
    printf '%s\n' "$best_tag"
}

_ghcr_remote_digest() {  # $1 = repo path, $2 = tag; prints the manifest digest (sha256:…)
    local token digest
    token=$(_ghcr_token "$1") || return 1
    digest=$(curl -fsSI --max-time 10 -H "Authorization: Bearer $token" \
        -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json" \
        "https://ghcr.io/v2/$1/manifests/$2" 2>/dev/null \
        | awk 'tolower($1)=="docker-content-digest:"{print $2}' | tr -d '\r') || return 1
    [[ $digest == sha256:* ]] || return 1
    printf '%s\n' "$digest"
}

# ghcr.io only: a custom AETHERSCAN_IMAGE on another registry skips ceiling + digest checks.
GHCR_REPO_PATH=""
[[ $IMAGE_REPO == ghcr.io/* ]] && GHCR_REPO_PATH=${IMAGE_REPO#ghcr.io/}

if [[ -z ${AETHERSCAN_IMAGE_TAG:-} ]]; then
    # First `version = "..."` line in pyproject.toml; awk (no pipe, portable GNU/BSD) so
    # `set -o pipefail` can't trip the wrapper. `|| true` (+ 2>/dev/null) so a missing file yields
    # an empty VER (-> fallback) rather than aborting the whole wrapper under `set -e`.
    VER=$(awk -F'"' '/^version = /{print $2; exit}' "$REPO/pyproject.toml" 2>/dev/null || true)
    if [[ $VER =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        AETHERSCAN_IMAGE_TAG="v$VER"
    else
        # .devN / unparseable checkout: resolve the ceiling-bounded release tag from the
        # registry (#424); fall back to the historical :latest when that fails (offline,
        # no curl, non-GHCR repo) — fail open, never block a run that has a cached image.
        # A REACHABLE registry with no qualifying tag also falls back, but loudly: that
        # :latest may exceed the ceiling, which the resolver otherwise guarantees against.
        CEILING_TAG=""
        ceiling_rc=1
        if [[ -n $GHCR_REPO_PATH ]]; then
            ceiling_rc=0
            CEILING_TAG=$(_ghcr_ceiling_tag "$GHCR_REPO_PATH" "$VER") || ceiling_rc=$?
        fi
        if [[ -n $CEILING_TAG ]]; then
            AETHERSCAN_IMAGE_TAG="$CEILING_TAG"
            echo "Resolved image tag '$CEILING_TAG' (newest published release at or below" \
                 "this checkout's version $VER, #424)." >&2
        else
            AETHERSCAN_IMAGE_TAG="latest"
            if [[ $ceiling_rc -eq 2 ]]; then
                echo "WARNING: no published release tag at or below this checkout's version" \
                     "$VER exists — falling back to ':latest', which may be NEWER than this" \
                     "checkout's code. Build from aetherscan.def for a version-true image." >&2
            fi
        fi
    fi
fi
IMAGE_REF="$IMAGE_REPO:$AETHERSCAN_IMAGE_TAG"

# (Re)pull if there's no image, the cached one was pulled for a different tag, or the remote
# digest of the SAME ref has moved (#424 — a retag; the sidecar's line 2 records the digest at
# pull time, older ref-only sidecars simply skip the digest check until their next pull). A $SIF
# NEWER than its sidecar was rebuilt locally over a previously pulled image — treat that as
# user-built (priority 1) and never clobber it silently (mv preserves the pulled file's mtime;
# the sidecar is written right after, so a pulled .sif is never newer than its sidecar, a local
# build always is). Sidecar-less / user-built images can't be verified: warn, and replace only
# under AETHERSCAN_FORCE_REPULL=1.
need_pull=0
UNVERIFIED_SIF_WARNING=""
if [[ ! -f "$SIF" ]]; then
    need_pull=1
elif [[ -f "$SIF.pulled-tag" && ! "$SIF" -nt "$SIF.pulled-tag" ]]; then
    sidecar_ref=$(head -n 1 "$SIF.pulled-tag" 2>/dev/null || true)
    if [[ "$sidecar_ref" != "$IMAGE_REF" ]]; then
        echo "Cached $SIF was pulled for '$sidecar_ref' but this checkout wants" \
             "'$IMAGE_REF' — re-pulling." >&2
        need_pull=1
    elif [[ -n $GHCR_REPO_PATH ]]; then
        recorded_digest=$(sed -n '2p' "$SIF.pulled-tag" 2>/dev/null || true)
        if [[ -z $recorded_digest ]]; then
            # Pre-#424 ref-only sidecar: nothing to drift-check against. A dev checkout
            # self-heals when its wanted ref changes; a pinned tag would keep this state
            # forever, so offer the refresh path explicitly.
            if [[ ${AETHERSCAN_FORCE_REPULL:-0} == 1 ]]; then
                echo "AETHERSCAN_FORCE_REPULL=1: re-pulling $IMAGE_REF to refresh the" \
                     "pre-#424 sidecar with a digest record." >&2
                need_pull=1
            else
                echo "NOTE: $SIF.pulled-tag predates the digest record (#424) — remote-drift" \
                     "checking is unavailable until the next pull. Set" \
                     "AETHERSCAN_FORCE_REPULL=1 (or rm the .sif) to refresh it." >&2
            fi
        else
            remote_digest=$(_ghcr_remote_digest "$GHCR_REPO_PATH" "$AETHERSCAN_IMAGE_TAG" || true)
            if [[ -n $remote_digest && "$recorded_digest" != "$remote_digest" ]]; then
                echo "WARNING: $IMAGE_REF has moved on the registry (recorded" \
                     "$recorded_digest, remote $remote_digest) — re-pulling over the" \
                     "cached $SIF (#424)." >&2
                need_pull=1
            fi
        fi
    fi
else
    # No sidecar, or $SIF newer than it: user-built or manually pulled — unverifiable.
    if [[ ${AETHERSCAN_FORCE_REPULL:-0} == 1 ]]; then
        # Deletion is deliberately DEFERRED to the pull's own atomic tmp -> mv publish:
        # if the replacement pull fails (offline node, quota'd cache), the existing —
        # possibly irreplaceable — local build survives and the run proceeds against it
        # with the stale-image warning, instead of being destroyed with nothing to run.
        echo "AETHERSCAN_FORCE_REPULL=1: replacing unverifiable $SIF with a fresh pull of" \
             "$IMAGE_REF (the current image is kept if the pull fails)." >&2
        need_pull=1
    else
        UNVERIFIED_SIF_WARNING="WARNING: $SIF has no pull provenance (user-built or manually\
 pulled) — the wrapper cannot verify it against $IMAGE_REF. If it is not a deliberate local\
 build, set AETHERSCAN_FORCE_REPULL=1 to replace it with the published image (#424)."
        echo "$UNVERIFIED_SIF_WARNING" >&2
    fi
fi

STALE_SIF_WARNING=""
if [[ $need_pull -eq 1 ]]; then
    echo "Pulling docker://$IMAGE_REF -> $SIF ..." >&2
    # Fetch the digest BEFORE pulling (#424): it records what we INTEND to pull, so a
    # retag landing mid-pull shows up as drift on the next run instead of being masked —
    # and the mv -> sidecar-write window below stays microseconds (no network I/O between
    # them), preserving the mtime invariant the user-built detection rides on.
    pulled_digest=""
    [[ -n $GHCR_REPO_PATH ]] \
        && pulled_digest=$(_ghcr_remote_digest "$GHCR_REPO_PATH" "$AETHERSCAN_IMAGE_TAG" || true)
    tmp="$SIF.pulling.$$"
    # Don't leave a multi-GB partial behind on Ctrl-C. INT/TERM exit explicitly: a trapped signal
    # does NOT terminate bash, so without the exit execution would resume in the failure branch
    # below and (with a stale $SIF present) run the job against the old image.
    trap 'rm -f "$tmp"' EXIT
    trap 'rm -f "$tmp"; exit 130' INT
    trap 'rm -f "$tmp"; exit 143' TERM
    # A SIGKILL/OOM/power-loss can't run the traps, so a partial from an earlier run with this PID
    # may still be here; both runtimes refuse to overwrite an existing pull target, so clear it.
    rm -f "$tmp"
    if "$RUNTIME" pull "$tmp" "docker://$IMAGE_REF" >&2; then
        mv -f "$tmp" "$SIF"
        # Sidecar v2 (#424): ref on line 1 (back-compat — every reader takes head -n1),
        # manifest digest on line 2 when the registry yielded one before the pull (enables
        # the drift check above on later runs; a digest-less sidecar just skips that
        # check). Written immediately after the mv — no network I/O between them.
        if [[ -n $pulled_digest ]]; then
            printf '%s\n%s\n' "$IMAGE_REF" "$pulled_digest" >"$SIF.pulled-tag"
        else
            printf '%s\n' "$IMAGE_REF" >"$SIF.pulled-tag"
        fi
        trap - EXIT INT TERM
        echo "Pulled and cached $SIF ($IMAGE_REF${pulled_digest:+ @ $pulled_digest})." >&2
    else
        trap - EXIT INT TERM
        rm -f "$tmp"
        if [[ -f "$SIF" ]]; then
            STALE_SIF_WARNING="WARNING: pull of docker://$IMAGE_REF failed; running against the cached\
 $SIF, which may be a different version (pulled for '$(head -n 1 "$SIF.pulled-tag" 2>/dev/null || echo unknown)')."
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

# Repeat the stale/unverified-image warnings right before launch so they aren't buried far
# above the run in a log.
[[ -n $STALE_SIF_WARNING ]] && echo "$STALE_SIF_WARNING" >&2
[[ -n $UNVERIFIED_SIF_WARNING ]] && echo "$UNVERIFIED_SIF_WARNING" >&2

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
