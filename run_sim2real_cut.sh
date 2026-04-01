#!/bin/bash
#SBATCH --job-name=sim2real_CUT
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --time=10:00:00
#SBATCH --account=def-lakahrs
#SBATCH --mem=32G
#SBATCH --output=slurm_output/%x-%j.out

# ───────────────────────── 0. Configuration ─────────────────────── #
DATAROOT=./datasets/sim2real
EXP_NAME=sim2real_CUT
MODEL=cut
N_EPOCHS=200
N_EPOCHS_DECAY=200
BATCH_SIZE=4
LOAD_SIZE=96
CROP_SIZE=84
NET_G=resnet_6blocks
PRINT_FREQ=1000
DISPLAY_FREQ=1000

# Resume from checkpoint (leave empty for fresh run)
RESUME_CKPT=""

# ───────────────────────── 1. Environment ───────────────────────── #
module load python/3.11.5 cuda/12.6

PROJECT_DIR=/scratch/anoopreh/projects/DCLGAN
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env" ]; then
  source "$PROJECT_DIR/.env"
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
echo "Image size   : load=${LOAD_SIZE} crop=${CROP_SIZE}"
echo "Batch size   : $BATCH_SIZE"
echo "Generator    : $NET_G"
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
  --netG "$NET_G" \
  --load_size "$LOAD_SIZE" \
  --crop_size "$CROP_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --display_id 0 \
  --use_wandb \
  --print_freq "$PRINT_FREQ" \
  --display_freq "$DISPLAY_FREQ" \
  --n_epochs "$N_EPOCHS" \
  --n_epochs_decay "$N_EPOCHS_DECAY" \
  --gpu_ids 0 \
  $RESUME_FLAG

# ───────────────────────── 4. Footer ────────────────────────────── #
echo "Finished     : $(date)"
echo "Checkpoints  : ./checkpoints/$EXP_NAME/"
