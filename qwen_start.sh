export CUDA_VISIBLE_DEVICES=5,7
export PYTHONPATH=.
export NCCL_IB_GID_INDEX=1
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_IB_HCA=mlx5_0,mlx5_10,mlx5_11,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9
export NCCL_IB_TIMEOUT=22
export NCCL_IB_DISABLE=0
export NCCL_IB_RETRY_CNT=7
export NCCL_NET=IB
export PYTHONPATH=.
export WANDB_API_KEY=<KEY>
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000
# export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_DEBUG=INFO 

# torchrun --nproc_per_node=8 --nnodes=8 --master_addr=<ip_address> --master_port=1234 --node_rank=1  recipes/full_finetune_distributed_loop.py --config config_train/8B_3.1_base_val_loop.yaml
torchrun --nproc_per_node=2 --nnodes=1 --master_addr=10.12.190.166 --master_port=1234 --node_rank=0 recipes/async_full_finetune_distributed.py --config recipes/configs/qwen2_5/3B_full.yaml