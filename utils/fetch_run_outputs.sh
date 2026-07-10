#!/usr/bin/env bash
# Pull the outputs of one Aetherscan run from a remote cluster node onto local
# machine, via rsync. You give it the run mode (train|inference), the run's
# --save-tag, and one or more remote machine names; it figures out which files
# that run produced and copies them into the matching local outputs/ subdir,
# prefixing each filename with the machine it came from.
#
# Usage:
#   ./utils/fetch_run_outputs.sh [options] <train|inference> <save_tag> <machine> [machine...]
#
# Options:
#   -a, --all       also pull intermediate train artifacts (per-round checkpoints
#                   & archived dirs). No effect for inference yet (see note below).
#   -d, --db        also pull the SQLite results DB (<output_path>/db/*.db) into
#                   outputs/data/db/<machine>_<filename>. Independent of the mode;
#                   the DB is shared across all runs (query it by its `tag` column).
#   -n, --dry-run   show what rsync would transfer; copy nothing.
#   -h, --help      show this help.
#
# Examples:
#   ./utils/fetch_run_outputs.sh train final_v1 blpc0
#   ./utils/fetch_run_outputs.sh train final_v1 blpc0 blpc1   # same tag, several nodes
#   ./utils/fetch_run_outputs.sh --all train final_v1 blpc0   # + checkpoints/archive
#   ./utils/fetch_run_outputs.sh --db inference run_2026 blpc0 # + the results DB
#
# Naming convention: a remote file BASENAME lands locally as
# "<machine>_BASENAME". e.g. on machine blpc0,
#   /datax/scratch/zachy/models/aetherscan/config_final_v1.json
#     -> outputs/models/blpc0_config_final_v1.json
# Pulling the same tag from several machines is therefore collision-free, which
# is the whole point of the machine prefix.
#
# Remote -> local directory mapping (these local dirs already exist):
#   <MODEL_REMOTE>            -> outputs/models/   (/datax/scratch/zachy/models/aetherscan)
#   <OUTPUT_REMOTE>/plots     -> outputs/plots/    (/datax/scratch/zachy/outputs/aetherscan/plots)
#   <OUTPUT_REMOTE>/logs      -> outputs/logs/     (/datax/scratch/zachy/outputs/aetherscan/logs)
#   <OUTPUT_REMOTE>/db        -> outputs/data/db/  (only with --db)
# Both remote roots honour the same env vars the pipeline reads
# (AETHERSCAN_MODEL_PATH / AETHERSCAN_OUTPUT_PATH), so override them if your
# cluster paths differ.
#
# How "files of interest" are chosen: every FINAL artifact a run writes is
# suffixed "_<save_tag>.<ext>" and sits at the top level of its directory, so a
# single glob "*_<save_tag>.*" selects them all. Intermediate artifacts use
# per-round tags and live in checkpoints/ (and archive/) subdirs, so they are
# naturally excluded unless you pass --all.
#
#   TRAIN, default : model_path/*_<tag>.*  (encoder, decoder, RF, config, rf_eval
#                    & shap & umap joblibs) + plots/*_<tag>.* (png + gif) + the log.
#   TRAIN, --all   : the above, plus model_path/checkpoints, model_path/archive,
#                    and plots/checkpoints (round-tagged — these reflect the most
#                    recent run on the node, since they are NOT save-tag-named).
### TODO:
#   INFERENCE      : PROVISIONAL. The inference pipeline isn't finished, so its
#                    on-disk outputs aren't settled yet — we deliberately do NOT
#                    hard-code its current (half-baked) paths here. For now we pull
#                    only the convention-based, stable artifacts: plots/*_<tag>.*
#                    and the log. Once inference lands, COME BACK AND COMPLETE this
#                    branch (config destination, candidate plots, any results
#                    export). --all adds nothing for inference yet.
#
# Assumptions:
#   * Machine name is used both as the ssh host and the filename prefix, so it must
#     resolve (a Host alias in ~/.ssh/config or DNS). User/port go in ssh config.
#   * Re-running is cheap: rsync -t skips files already present and unchanged.
### TODO:
#   * Log filenames are tagged like every other output, i.e. aetherscan_<tag>.log
#     in logs/. (The logger writes a single untagged aetherscan.log TODAY; per-tag
#     log names are a planned change. This script is written for that future state,
#     so it will report "no match" against today's untagged log until that lands.)

set -euo pipefail

# --- remote roots (mirror config.py's resolution so cluster overrides apply) ---
MODEL_REMOTE="${AETHERSCAN_MODEL_PATH:-/datax/scratch/zachy/models/aetherscan}"
OUTPUT_REMOTE="${AETHERSCAN_OUTPUT_PATH:-/datax/scratch/zachy/outputs/aetherscan}"
PLOTS_REMOTE="${OUTPUT_REMOTE}/plots"
LOGS_REMOTE="${OUTPUT_REMOTE}/logs"
DB_REMOTE="${OUTPUT_REMOTE}/db"

# --- local destinations (already exist under the repo's outputs/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_MODELS="${REPO_ROOT}/outputs/models"
LOCAL_PLOTS="${REPO_ROOT}/outputs/plots"
LOCAL_LOGS="${REPO_ROOT}/outputs/logs"
LOCAL_DB="${REPO_ROOT}/outputs/data/db"

ALL=0
DB=0
DRYRUN=0
POS=()

usage() {
    cat <<'EOF'
Pull one Aetherscan run's outputs from remote cluster node(s) via rsync.

Usage: ./utils/fetch_run_outputs.sh [options] <train|inference> <save_tag> <machine> [machine...]
  -a, --all       also pull intermediate train artifacts (checkpoints, archive)
  -d, --db        also pull the SQLite results DB into outputs/data/db/
  -n, --dry-run   show what rsync would transfer; copy nothing
  -h, --help      show this help

Files land as outputs/<models|plots|logs>/<machine>_<basename>. Selection is by
the "*_<save_tag>.*" suffix every final artifact carries. See the header comment
in this file for the full remote->local mapping and per-mode details.
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    -a | --all) ALL=1 ;;
    -d | --db) DB=1 ;;
    -n | --dry-run) DRYRUN=1 ;;
    -h | --help) usage 0 ;;
    --)
        shift
        while [[ $# -gt 0 ]]; do
            POS+=("$1")
            shift
        done
        break
        ;;
    -*)
        echo "Error: unknown option '$1'." >&2
        usage 2 >&2
        ;;
    *) POS+=("$1") ;;
    esac
    shift
done

# --- validate positionals: <command> <save_tag> <machine> [machine...] ---
if [[ ${#POS[@]} -lt 3 ]]; then
    echo "Error: need <train|inference> <save_tag> <machine> [machine...]." >&2
    usage 2 >&2
fi
COMMAND="${POS[0]}"
TAG="${POS[1]}"
MACHINES=("${POS[@]:2}")

case "$COMMAND" in
train | inference) ;;
*)
    echo "Error: command must be 'train' or 'inference', got '$COMMAND'." >&2
    usage 2 >&2
    ;;
esac
[[ -n "$TAG" ]] || {
    echo "Error: save_tag must be non-empty." >&2
    exit 2
}

# --- ssh/rsync transport. ControlMaster multiplexes the many small transfers
#     over one connection (the per-file prefix-rename forces one rsync per file,
#     and a cross-country cluster handshake per file would be painfully slow). ---
# Keep the control socket under /tmp with a short name: a unix socket path must
# stay under ~104 bytes, and macOS's $TMPDIR is already long enough to blow that.
# %C (a fixed 40-char hash of the connection) differentiates hosts when several
# machines are pulled in one run.
CONTROL_DIR="$(mktemp -d /tmp/asf.XXXXXX)"
trap 'rm -rf "$CONTROL_DIR"' EXIT
SSH_OPTS="-o ControlMaster=auto -o ControlPersist=120 -o ControlPath=${CONTROL_DIR}/cm-%C"

RSYNC_FLAGS=(-tz --partial)
((DRYRUN)) && RSYNC_FLAGS+=(--dry-run)

log() { printf '  %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }

# Copy every top-level file matching a glob from a remote dir into a local dir,
# renaming each to "<host>_<basename>". The remote shell expands the glob.
fetch_glob() {
    local host="$1" rdir="$2" ldir="$3" pattern="$4" listing rfile base dest n=0
    listing="$(ssh $SSH_OPTS "$host" "ls -1 ${rdir}/${pattern} 2>/dev/null" || true)"
    if [[ -z "$listing" ]]; then
        warn "no files matching ${pattern} in ${host}:${rdir}"
        return 0
    fi
    mkdir -p "$ldir"
    while IFS= read -r rfile; do
        [[ -z "$rfile" ]] && continue
        base="$(basename "$rfile")"
        dest="${ldir}/${host}_${base}"
        log "${host}:${rfile} -> ${dest}"
        rsync "${RSYNC_FLAGS[@]}" -e "ssh $SSH_OPTS" "${host}:${rfile}" "$dest"
        n=$((n + 1))
    done <<<"$listing"
    log "(${n} file(s) from ${host}:${rdir})"
}

# Mirror a whole remote subtree (intermediates) into "<lparent>/<host>_<name>/".
# Dir-level prefix keeps round-tagged files grouped without renaming each one.
fetch_subtree() {
    local host="$1" rdir="$2" lparent="$3" name="$4" dest
    if ! ssh $SSH_OPTS "$host" "test -d '$rdir'"; then
        warn "missing dir ${host}:${rdir}"
        return 0
    fi
    dest="${lparent}/${host}_${name}"
    mkdir -p "$dest"
    log "${host}:${rdir}/ -> ${dest}/"
    rsync "${RSYNC_FLAGS[@]}" -r -e "ssh $SSH_OPTS" "${host}:${rdir}/" "${dest}/"
}

echo "Fetching '${COMMAND}' run outputs for tag '${TAG}' from: ${MACHINES[*]}"
((ALL)) && echo "(--all: including intermediate artifacts)"
((DB)) && echo "(--db: including the SQLite results DB)"
((DRYRUN)) && echo "(--dry-run: no files will be written)"

for m in "${MACHINES[@]}"; do
    echo "=== ${m} ==="
    if [[ "$COMMAND" == "train" ]]; then
        fetch_glob "$m" "$MODEL_REMOTE" "$LOCAL_MODELS" "*_${TAG}.*"
        fetch_glob "$m" "$PLOTS_REMOTE" "$LOCAL_PLOTS" "*_${TAG}.*"
        fetch_glob "$m" "$LOGS_REMOTE" "$LOCAL_LOGS" "*_${TAG}.*"
        if ((ALL)); then
            fetch_subtree "$m" "${MODEL_REMOTE}/checkpoints" "$LOCAL_MODELS" "checkpoints"
            fetch_subtree "$m" "${MODEL_REMOTE}/archive" "$LOCAL_MODELS" "archive"
            fetch_subtree "$m" "${PLOTS_REMOTE}/checkpoints" "$LOCAL_PLOTS" "checkpoints"
        fi
    else
        # PROVISIONAL inference branch — pipeline unfinished, paths not settled.
        # Only the convention-based artifacts are pulled. TODO: complete once the
        # inference pipeline lands (config destination, candidate plots, exports).
        fetch_glob "$m" "$PLOTS_REMOTE" "$LOCAL_PLOTS" "*_${TAG}.*"
        fetch_glob "$m" "$LOGS_REMOTE" "$LOCAL_LOGS" "*_${TAG}.*"
    fi
    if ((DB)); then
        fetch_glob "$m" "$DB_REMOTE" "$LOCAL_DB" "*.db"
    fi
done

echo "Done."
