#!/bin/bash

# Script to run parallel distillation workers in tmux
# Usage: ./run_parallel_distill.sh [num_workers] [limit] [batch_size] [model]

# Configuration with defaults
NUM_WORKERS=${1:-8}
LIMIT=${2:-10}
BATCH_SIZE=${3:-10}
MODEL=${4:-GPT41}
DATASET=${5:-"UWV/wim-synthetic-data-rd"}
SESSION_NAME="distill_workers"

# Validate NUM_WORKERS is a positive integer
if ! [[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of workers must be a positive integer (got: '$NUM_WORKERS')"
    echo ""
    echo "Usage: $0 [num_workers] [limit] [batch_size] [model] [dataset]"
    echo ""
    echo "Examples:"
    echo "  $0              # Use defaults: 8 workers, 10 records"
    echo "  $0 4            # 4 workers, 10 records"
    echo "  $0 8 100        # 8 workers, 100 records"
    echo "  $0 4 50 25 GPT4O  # 4 workers, 50 records, batch size 25, GPT4O model"
    exit 1
fi

# Validate other numeric parameters
if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Limit must be a positive integer (got: '$LIMIT')"
    exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Batch size must be a positive integer (got: '$BATCH_SIZE')"
    exit 1
fi

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    echo ""
    echo "Do you want to kill it and start fresh? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Killing existing session..."
        tmux kill-session -t $SESSION_NAME
        sleep 1
    else
        echo "Exiting without changes."
        exit 1
    fi
fi

# Create new tmux session with larger window size
echo "Creating tmux session with $NUM_WORKERS workers..."
tmux new-session -d -s $SESSION_NAME -x 200 -y 50

# Create panes more carefully to avoid "no space" errors
# For many workers, we need to rebalance the layout periodically
for ((i=1; i<$NUM_WORKERS; i++)); do
    # Try to split the window
    if ! tmux split-window -t $SESSION_NAME 2>/dev/null; then
        # If split fails, rebalance and try again
        tmux select-layout -t $SESSION_NAME tiled
        sleep 0.1
        tmux split-window -t $SESSION_NAME
    fi
    
    # Rebalance every few panes to ensure space
    if (( i % 3 == 0 )); then
        tmux select-layout -t $SESSION_NAME tiled
        sleep 0.1
    fi
done

# Final tiled layout
tmux select-layout -t $SESSION_NAME tiled

# Run the command in each pane
echo "Starting workers (with staggered start to avoid database locks)..."
echo "Worker 0 starts immediately, worker 1 after 2s, worker 2 after 4s, etc."

# Get list of pane IDs
PANE_IDS=($(tmux list-panes -t $SESSION_NAME -F '#{pane_id}'))

# Send commands to each pane using pane IDs
for ((i=0; i<$NUM_WORKERS && i<${#PANE_IDS[@]}; i++)); do
    COMMAND="cd /home/yeb/Developer/linnaeus-ai/wim-signaalberichten && \
export PYTHONPATH=\$(pwd)/src && \
python src/scripts/distill_graph_all_nodes.py \
  --limit $LIMIT \
  --batch-size $BATCH_SIZE \
  --model $MODEL \
  --add-labels \
  --dataset $DATASET \
  --worker-id worker$i"
    
    tmux send-keys -t ${PANE_IDS[$i]} "$COMMAND" C-m
done

echo "All workers started. Attaching to session..."
echo ""
echo "Tmux controls:"
echo "  - Ctrl-b + arrow keys: Navigate between panes"
echo "  - Ctrl-b + d: Detach from session"
echo "  - Ctrl-b + z: Zoom in/out of current pane"
echo "  - Ctrl-b + x: Kill current pane"
echo ""
echo "To reattach later: tmux attach -t $SESSION_NAME"
echo "To kill session: tmux kill-session -t $SESSION_NAME"
echo ""

# Attach to the session
tmux attach-session -t $SESSION_NAME