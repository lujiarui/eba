#!/bin/bash
#SBATCH --job-name=bash          # avoid lightning auto-debug configuration
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=6      # shall >#GPU to avoid overtime thread distribution 
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem=192GB               
#SBATCH --time 47:59:59           # time limit (D-HH:MM:ss)
#SBATCH --gres=gpu:4
#SBATCH --constraint='80gb'       # 80GB GPU

#########################
####### Configs #########
#########################
CONDA_ENV_NAME=py311
CONDA_HOME=$(expr match $CONDA_PREFIX '\(.*miniconda\)')
WORKDIR=$(pwd)

#########################
####### Env loader ######
#########################
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}
module load cuda/12.1.1 # for deepspeed compilation

dt=$(date '+%d/%m/%Y-%H:%M:%S')
echo "[$0] | Start: ${dt}"

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export CUTLASS_PATH="./cutlass"

checkpoint_path="~/scratch/data/pdb/protenix/model_v0.2.0.pt"

python3 ./runner/train.py \
--run_name atlas_all \
--seed 42 \
--base_dir example/pairformer_emb \
--dtype bf16 \
--project protenix \
--use_wandb false \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--model.N_cycle 10 \
--data.train_sets example \
--data.test_sets example_test \
--dump_embeddings True