#!/bin/bash
#SBATCH --job-name=dclgan_grumpycat
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --time=6:00:00
#SBATCH --account=def-lakahrs
#SBATCH --mem=32G
#SBATCH --output=slurm_output/%x-%j.out

# ───────────────────────── 0. Configuration ─────────────────────── #
DATAROOT=./datasets/grumpifycat
EXP_NAME=grumpycat_DCL
MODEL=dcl                    # dcl | simdcl | cut | fastcut | cycle_gan
N_EPOCHS=200
N_EPOCHS_DECAY=200
BATCH_SIZE=1

# Resume from checkpoint (leave empty for fresh run)
RESUME_CKPT=""

# ───────────────────────── 1. Environment ───────────────────────── #
module load python/3.11.5 cuda/12.6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load wandb API key from .env if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
  source "$SCRIPT_DIR/.env"
  export WANDB_API_KEY
fi

source "$HOME/envs/dclgan/bin/activate"

# ───────────────────────── 2. Provenance ────────────────────────── #
mkdir -p slurm_output

echo "========== JOB INFO =========="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname)"
echo "GPUs         : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Started      : $(date)"
echo "Experiment   : $EXP_NAME"
echo "Model        : $MODEL"
echo "Dataroot     : $DATAROOT"
echo "Git branch   : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "Git commit   : $(git log -1 --format='%h %s' 2>/dev/null || echo unknown)"
echo "=============================="

# ───────────────────────── 3. Training ──────────────────────────── #
RESUME_FLAG=""
if [ -n "$RESUME_CKPT" ]; then
  RESUME_FLAG="--continue_train --epoch latest"
  echo "Resuming from: $RESUME_CKPT"
fi

python train.py \
  --dataroot "$DATAROOT" \
  --name "$EXP_NAME" \
  --model "$MODEL" \
  --display_id 0 \
  --use_wandb \
  --n_epochs "$N_EPOCHS" \
  --n_epochs_decay "$N_EPOCHS_DECAY" \
  --batch_size "$BATCH_SIZE" \
  --gpu_ids 0 \
  $RESUME_FLAG

# ───────────────────────── 4. Footer ────────────────────────────── #
echo "Finished     : $(date)"
echo "Checkpoints  : ./checkpoints/$EXP_NAME/"
