#!/bin/bash
#SBATCH --job-name=distributed_training   # Job name
#SBATCH --nodes=6                         # Number of nodes 
#SBATCH --ntasks-per-node=1               # Number of tasks per node (select the number of CPUs and threads in SLURM sbatch)
#SBATCH --cpus-per-task=96                # CPUs per task (adjust as needed) (select the number of CPUs and threads in SLURM sbatch)
#SBATCH --nodelist=node00[1-7]           # Specific nodes (could aslo be specified like node001,node002,...,node00n)
#SBATCH --exclude=node005
#SBATCH --gres=gpu:8                      # Number of GPUs per node
#SBATCH --time=72:00:00                   # Maximum runtime (adjust as needed)
#SBATCH --partition=defq                  # Partition name
#SBATCH --exclusive                       # Only you allcoate this node
#SBATCH --output=slurm-%N.%j.out          # Standard output log file
#SBATCH --error=slurm-%N.%j.err           # Standard error log file
# Load  environment variables
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
# export OMP_NUM_THREADS=96 # The reason why we had 1600 tok/sec when it's equal to 8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=0
# export NCCL_IB_TIMEOUT=22
# export NCCL_DEBUG=INFO


# Added variable in a hope to speed up training and memory.
# ulimit -n 102400
# export NCCL_IB_QPS_PER_CONNECTION=2 
# export NCCL_NSOCKS_PERTHREAD=3 
# export NCCL_SOCKET_NTHREADS=4
# export NCCL_IGNORE_CPU_AFFINITY=1

# To solve issue with wandb POST request, need to download wget https://curl.se/ca/cacert.pem and put path as env. variable
export SSL_CERT_FILE=/home/adal_abilbekov/cacert.pem 

# Spcify scratch directory, otherwise you will save everything within home/.cache
export HF_HOME=/local/adal_abilbekov/

export WANDB_API_KEY=_
export PYTHONPATH=torchtune_try_1/KazLLM_Bee
export WANDB_DISABLED=False

# nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
# head_node=${nodes[0]}  # First node is the master node
# MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | cut -d" " -f2)
# MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

TRAIN_SCRIPT="/home/adal_abilbekov/torchtune_try_1/KazLLM_Bee/recipes/full_finetune_distributed_loop_no_val.py"
CONFIG_PATH="/home/adal_abilbekov/torchtune_try_1/KazLLM_Bee/config_train/8B_3.1_inst_noval_loop.yaml"

echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "TRAIN_SCRIPT: ${TRAIN_SCRIPT}"

# echo "Allocated Nodes: ${nodes[@]}"
# echo "Master Node: $head_node"
# echo "Master Node IP: $head_node_ip"
echo "Master Node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
# Run distributed training using torchrun

conda activate aenv
srun torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node 8 \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_id $SLURM_JOB_ID \
    ${TRAIN_SCRIPT} \
    --config \
    ${CONFIG_PATH}
