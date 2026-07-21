#!/usr/bin/env bash
# Kill a running Aetherscan pipeline (the main process + every worker it spawned).
#
# Run this from a *separate* shell on the *same* machine where the pipeline is
# running. It finds the pipeline on its own — no PID argument needed.
#
# Usage:
#     ./utils/kill_pipeline.sh              # graceful: SIGTERM, then SIGKILL after a timeout
#     ./utils/kill_pipeline.sh --dry-run    # show what would be killed, send nothing
#     ./utils/kill_pipeline.sh --force      # skip graceful shutdown, SIGKILL immediately
#     ./utils/kill_pipeline.sh --timeout 60 # wait 60s for graceful shutdown (default 30)
#
# Assumptions (kept deliberately narrow — read before relying on this):
#
#   1. SINGLE INSTANCE. At most one Aetherscan main process runs on the machine at
#      a time. If two are found the script refuses to guess and exits non-zero
#      rather than kill the wrong tree.
#
#   2. BOTH RUN MODES land the same way. Whether launched directly
#      (`PYTHONPATH=src python -m aetherscan.main ...`) or via the container
#      (`./utils/run_container.sh python -m aetherscan.main ...`), the actual
#      worker-spawning process is a Python interpreter whose argv contains
#      `-m aetherscan.main`. Apptainer/Singularity `exec` shares the host PID
#      namespace (no `--pid`), so that Python process and its pool workers are
#      directly visible and signalable from the host — exactly as they appear in
#      htop. The container runtime is merely the *parent* of that Python process;
#      once the Python process exits, the runtime wrapper exits on its own, so we
#      target the Python "main" process, not the wrapper.
#
#   3. FORK START METHOD. Pool workers are forks of the main process, so they
#      share its argv (`python -m aetherscan.main ...`) — this is what lets us
#      tell the main process apart from its workers by process tree. We still walk
#      the live child tree for the actual kill, so spawn/forkserver workers (which
#      would have a different argv) are caught too.
#
# Graceful path (default): send SIGTERM to the main process only. Its
# ResourceManager signal handler (manager.py:_signal_handler) runs cleanup_all()
# in the main process — closing the multiprocessing pools (which SIGTERM->SIGKILL
# their own workers), unlinking shared memory, and shutting down the monitor, DB,
# and logger — then sys.exit(0). We poll until the whole tree is gone or the
# timeout elapses, then escalate to SIGKILL.
#
# A forced SIGKILL (either --force or timeout escalation) bypasses that cleanup,
# so orphaned /dev/shm segments may be left behind (the OS reclaims GPU memory and
# RAM on process exit, but POSIX shared memory persists). See the cleanup hint
# printed at the end and KNOWN_ISSUES.md ("Pool Cleanup Hangs").
#
# Post-mortem sweep: if no main process matches (it already died — e.g. kill -9,
# OOM), a RoundDataProducer may have survived as an orphan (reparented to PID 1,
# argv `... spawn_main ...` — invisible to $PATTERN). The producer records its PID
# in {round_data_root}/{tag}/producer.pid (removed on graceful shutdown), so the
# script reads any leftover pidfiles under the round-data root and reaps those
# trees. Pass --round-data-root if the run used a non-default --round-data-dir.

set -euo pipefail

# Full-cmdline match for the pipeline. Matches the direct Python invocation, its
# fork workers (same argv), and the container wrapper (whose argv is
# "... python -m aetherscan.main ..."). The `-m ` anchor keeps it from matching
# unrelated processes that merely have "aetherscan" in a path.
PATTERN='-m[[:space:]]+aetherscan\.main'

# Where RoundDataProducer.start() drops its per-tag producer.pid (config default:
# {data_path}/training/round_data — see config.py get_training_file_path).
ROUND_DATA_ROOT="${AETHERSCAN_DATA_PATH:-/datax/scratch/zachy/data/aetherscan}/training/round_data"

TIMEOUT=30
FORCE=0
DRYRUN=0

usage() {
    cat <<'EOF'
Kill a running Aetherscan pipeline (main process + every worker it spawned).
Run from a separate shell on the same machine. Finds the pipeline itself.

Usage: ./utils/kill_pipeline.sh [options]
  (default)         graceful: SIGTERM, then SIGKILL after the timeout
  -n, --dry-run     show what would be killed, send nothing
  -f, --force       skip graceful shutdown, SIGKILL immediately
  -t, --timeout N   seconds to wait for graceful shutdown (default 30)
  -r, --round-data-root DIR
                    where to look for orphaned-producer pidfiles when the main
                    process is already dead (default: AETHERSCAN_DATA_PATH-based)
  -h, --help        show this help
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --force | -f) FORCE=1 ;;
    --dry-run | -n) DRYRUN=1 ;;
    --timeout | -t)
        TIMEOUT="${2:-}"
        [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || {
            echo "Error: --timeout needs an integer (seconds)." >&2
            exit 2
        }
        shift
        ;;
    --round-data-root | -r)
        ROUND_DATA_ROOT="${2:-}"
        [[ -n "$ROUND_DATA_ROOT" ]] || {
            echo "Error: --round-data-root needs a directory." >&2
            exit 2
        }
        shift
        ;;
    --help | -h) usage 0 ;;
    *)
        echo "Error: unknown argument '$1'." >&2
        usage 2 >&2
        ;;
    esac
    shift
done

# Echo every PID descended from $1, recursively (children, grandchildren, ...).
descendants() {
    local child
    for child in $(pgrep -P "$1" 2>/dev/null || true); do
        echo "$child"
        descendants "$child"
    done
}

# Return 0 if any of the given PIDs is still alive.
any_alive() {
    local pid
    for pid in "$@"; do
        kill -0 "$pid" 2>/dev/null && return 0
    done
    return 1
}

# Find the single Python "main" process: a `python*` process running
# `-m aetherscan.main` whose parent is NOT itself a python process. That filter
# drops the pool workers (their parent IS the main python process) and the
# container runtime wrapper (its comm is apptainer/singularity/starter, not
# python). Per the single-instance assumption there is exactly one such process.
find_main_pid() {
    local pids pid comm ppid pcomm main=""
    mapfile -t pids < <(pgrep -f -- "$PATTERN" 2>/dev/null || true)
    ((${#pids[@]})) || return 0

    for pid in "${pids[@]}"; do
        [[ "$pid" == "$$" ]] && continue
        comm=$(ps -o comm= -p "$pid" 2>/dev/null) || continue
        [[ "$comm" == python* ]] || continue
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ') || continue
        pcomm=$(ps -o comm= -p "$ppid" 2>/dev/null || true)
        [[ "$pcomm" == python* ]] && continue
        if [[ -n "$main" && "$main" != "$pid" ]]; then
            echo "Error: found multiple Aetherscan main processes ($main and $pid)." >&2
            echo "This utility assumes a single running instance; resolve manually." >&2
            exit 3
        fi
        main="$pid"
    done
    [[ -n "$main" ]] && echo "$main"
}

# Post-mortem sweep for producers orphaned by an ungraceful main-process death:
# read each leftover {round_data_root}/{tag}/producer.pid, and if that PID is
# still a live python process, kill its whole tree (SIGTERM first so the
# producer's own handler reaps its pool, unless --force) and remove the pidfile.
reap_orphan_producers() {
    local pidfile pid args found=0 waited p
    for pidfile in "$ROUND_DATA_ROOT"/*/producer.pid; do
        [[ -e "$pidfile" ]] || continue
        pid=$(tr -cd '0-9' <"$pidfile")
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            echo "Removing stale producer pidfile: $pidfile"
            [[ "$DRYRUN" == 1 ]] || rm -f "$pidfile"
            continue
        fi
        # PID-reuse guard: the recorded producer is a python process (spawn_main
        # argv); anything else holding the PID now is not it.
        args=$(ps -o args= -p "$pid" 2>/dev/null || true)
        if ! grep -qE 'python|spawn_main' <<<"$args"; then
            echo "Removing stale producer pidfile (PID $pid reused by: ${args:0:60}): $pidfile"
            [[ "$DRYRUN" == 1 ]] || rm -f "$pidfile"
            continue
        fi
        found=1
        echo "Orphaned producer : $pid  (${args:0:72}...)  [$pidfile]"
        mapfile -t TREE < <(descendants "$pid")
        TREE=("$pid" ${TREE[@]+"${TREE[@]}"})
        echo "Producer tree     : ${TREE[*]}"
        if [[ "$DRYRUN" == 1 ]]; then
            echo "[dry-run] would terminate ${#TREE[@]} process(es); no signals sent."
            continue
        fi
        if [[ "$FORCE" == 0 ]]; then
            # SIGTERM: the producer's handler terminates its own pool before dying.
            kill -TERM "$pid" 2>/dev/null || true
            waited=0
            while ((waited < TIMEOUT)); do
                any_alive "${TREE[@]}" || break
                sleep 1
                waited=$((waited + 1))
            done
        fi
        for p in "${TREE[@]}"; do
            kill -KILL "$p" 2>/dev/null || true
        done
        rm -f "$pidfile"
        echo "Orphaned producer tree reaped."
    done
    if ((!found)); then
        echo "No orphaned producers found under $ROUND_DATA_ROOT."
    fi
}

MAIN_PID=$(find_main_pid)
if [[ -z "${MAIN_PID:-}" ]]; then
    echo "No running Aetherscan pipeline found."
    reap_orphan_producers
    exit 0
fi

# Snapshot the whole tree now: once the main process dies its children reparent to
# init (PID 1) and a `pgrep -P $MAIN_PID` would no longer find them, so we record
# them up front to verify (and, if needed, force-kill) the complete set later.
mapfile -t WORKERS < <(descendants "$MAIN_PID")
ALL=("$MAIN_PID" ${WORKERS[@]+"${WORKERS[@]}"})

echo "Main process : $MAIN_PID  ($(ps -o args= -p "$MAIN_PID" 2>/dev/null | cut -c1-72)...)"
echo "Worker PIDs  : ${WORKERS[*]:-(none spawned yet)}"

if [[ "$DRYRUN" == 1 ]]; then
    echo "[dry-run] would terminate ${#ALL[@]} process(es); no signals sent."
    exit 0
fi

if [[ "$FORCE" == 0 ]]; then
    echo "Sending SIGTERM to main process $MAIN_PID (graceful ResourceManager cleanup)..."
    kill -TERM "$MAIN_PID" 2>/dev/null || true

    waited=0
    while ((waited < TIMEOUT)); do
        if ! any_alive "${ALL[@]}"; then
            echo "Pipeline shut down gracefully."
            exit 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "Grace period (${TIMEOUT}s) elapsed; escalating to SIGKILL."
fi

# Force path. Kill workers before the main process so nothing reparents away
# mid-sweep, and recompute the live child tree (union with the snapshot) in case
# workers were respawned.
REMAINING=()
if ((${#WORKERS[@]})); then
    REMAINING+=("${WORKERS[@]}")
fi
mapfile -t LIVE < <(descendants "$MAIN_PID")
if ((${#LIVE[@]})); then
    REMAINING+=("${LIVE[@]}")
fi
if ((${#REMAINING[@]})); then
    mapfile -t REMAINING < <(printf '%s\n' "${REMAINING[@]}" | sort -un)
    for pid in "${REMAINING[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
fi
kill -KILL "$MAIN_PID" 2>/dev/null || true

sleep 1
if any_alive "${ALL[@]}"; then
    echo "Warning: some processes survived SIGKILL (likely uninterruptible D-state):" >&2
    for pid in "${ALL[@]}"; do
        kill -0 "$pid" 2>/dev/null && ps -o pid=,stat=,comm= -p "$pid" >&2 || true
    done
    exit 1
fi

echo "Pipeline killed."
echo "Note: a forced kill skips ResourceManager cleanup. Check for orphaned shared"
echo "memory with 'ls /dev/shm' and remove stale 'psm_*' segments if present."
