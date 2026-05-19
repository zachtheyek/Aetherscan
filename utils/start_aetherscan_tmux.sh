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
PSUTIL_CMD=$(
    cat <<'EOF'
python -c "import psutil, time; [print(f'CPU: {psutil.cpu_percent()}%, MEM: {psutil.virtual_memo
ry().percent}%') or time.sleep(1) for _ in iter(int,1)]"
EOF
)

# ───── Window 1: pipeline ─────
tmux new-session -d -s "$SESSION" -n pipeline
TOP=$(tmux display-message -p -t "$SESSION:pipeline" '#{pane_id}')

tmux send-keys -t "$TOP" 'conda activate aetherscan' C-m
tmux send-keys -t "$TOP" 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' C-m
tmux send-keys -t "$TOP" 'export TF_CPP_MIN_LOG_LEVEL=1' C-m
tmux send-keys -t "$TOP" 'cd Aetherscan/' C-m
tmux send-keys -t "$TOP" 'source .env' C-m

# Top pane stays full-width; A is bottom-left, B is bottom-right.
A=$(tmux split-window -v -t "$TOP" -P -F '#{pane_id}')
B=$(tmux split-window -h -t "$A" -P -F '#{pane_id}')

tmux send-keys -t "$A" 'watch -n 1 tree -L 3 /datax/scratch/zachy/models/aetherscan/' C-m
tmux send-keys -t "$B" 'watch -n 1 tree -L 4 /datax/scratch/zachy/outputs/aetherscan/' C-m

# ───── Window 2: htop ─────
tmux new-window -t "$SESSION" -n htop
C=$(tmux display-message -p -t "$SESSION:htop" '#{pane_id}')
D=$(tmux split-window -v -t "$C" -P -F '#{pane_id}')

tmux send-keys -t "$C" 'htop' C-m
tmux send-keys -t "$D" 'conda activate aetherscan' C-m
tmux send-keys -t "$D" "$PSUTIL_CMD" C-m

# ───── Window 3: nvidia-smi ─────
tmux new-window -t "$SESSION" -n nvidia-smi
tmux send-keys -t "$SESSION:nvidia-smi" 'watch -n 1 nvidia-smi' C-m

# ───── Window 4: shm ─────
tmux new-window -t "$SESSION" -n shm
tmux send-keys -t "$SESSION:shm" 'watch -n 1 ls -lh /dev/shm' C-m

# Land on the pipeline window, top pane (where you actually type).
tmux select-window -t "$SESSION:pipeline"
tmux select-pane -t "$TOP"

attach_or_switch
