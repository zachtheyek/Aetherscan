#!/usr/bin/env bash
set -euo pipefail

SESSION="aetherscan"

# If we're already inside tmux, switch; otherwise attach. Idempotent on reboot.
attach_or_switch() {
    if [ -n "${TMUX:-}" ]; then
        tmux switch-client -t "$SESSION"
    else
        exec tmux attach -t "$SESSION"
    fi
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
    attach_or_switch
fi

# Long python one-liner kept verbatim via single-quoted heredoc (no expansion).
# Keep this on a SINGLE line — the heredoc preserves any literal newline, and
# tmux send-keys passes it through as Enter, which breaks the `-c "..."` quoted
# arg mid-f-string ("SyntaxError: unterminated string literal").
PSUTIL_CMD=$(
    cat <<'EOF'
python -c "import psutil, time; [print(f'CPU: {psutil.cpu_percent()}%, MEM: {psutil.virtual_memory().percent}%') or time.sleep(1) for _ in iter(int,1)]"
EOF
)

# ───── Window 1: pipeline ─────
tmux new-session -d -s "$SESSION" -n pipeline
TOP=$(tmux display-message -p -t "$SESSION:pipeline" '#{pane_id}')

tmux send-keys -t "$TOP" 'conda activate aetherscan' C-m
tmux send-keys -t "$TOP" 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' C-m
tmux send-keys -t "$TOP" 'export TF_CPP_MIN_LOG_LEVEL=1' C-m
tmux send-keys -t "$TOP" 'cd Aetherscan/' C-m

# Single full-window working pane (the filesystem watches now live in the "data" window).

# ───── Window 2: htop ─────
tmux new-window -t "$SESSION" -n htop
# htop keeps the top 75%; the CPU/MEM ticker takes the bottom 25%.
C=$(tmux display-message -p -t "$SESSION:htop" '#{pane_id}')
D=$(tmux split-window -v -l 25% -t "$C" -P -F '#{pane_id}')

tmux send-keys -t "$C" 'htop' C-m
tmux send-keys -t "$D" 'conda activate aetherscan' C-m
tmux send-keys -t "$D" "$PSUTIL_CMD" C-m

# Put focus back on the htop pane so that's what's active when the user
# later switches to this window (split-window leaves the new pane focused).
tmux select-pane -t "$C"

# ───── Window 3: nvidia-smi ─────
tmux new-window -t "$SESSION" -n nvidia-smi
tmux send-keys -t "$SESSION:nvidia-smi" 'watch -n 1 nvidia-smi' C-m

# ───── Window 4: data ─────
# Four even-vertical panes (top→bottom): /dev/shm, then the data / models / outputs trees.
tmux new-window -t "$SESSION" -n data
E=$(tmux display-message -p -t "$SESSION:data" '#{pane_id}')
F=$(tmux split-window -v -t "$E" -P -F '#{pane_id}')
G=$(tmux split-window -v -t "$F" -P -F '#{pane_id}')
H=$(tmux split-window -v -t "$G" -P -F '#{pane_id}')
tmux select-layout -t "$SESSION:data" even-vertical

tmux send-keys -t "$E" 'watch -n 1 ls -lh /dev/shm' C-m
tmux send-keys -t "$F" 'watch -n 1 tree -L 3 /datax/scratch/zachy/data/aetherscan' C-m
tmux send-keys -t "$G" 'watch -n 1 tree -L 2 /datax/scratch/zachy/models/aetherscan' C-m
tmux send-keys -t "$H" 'watch -n 1 tree -L 2 /datax/scratch/zachy/outputs/aetherscan' C-m

# Land on the pipeline window, top pane (where you actually type).
tmux select-window -t "$SESSION:pipeline"
tmux select-pane -t "$TOP"

attach_or_switch
