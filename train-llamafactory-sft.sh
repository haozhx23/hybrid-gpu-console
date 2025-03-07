#!/bin/bash

# echo "##### train-llamafactory.sh#####"

# pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install deepspeed==0.14.4 -i https://pypi.tuna.tsinghua.edu.cn/simple

# cd /workspace/llamafactory-training/LLaMA-Factory
# pip install -e ".[torch,metrics]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 直接使用节点预定义环境变量
echo "ECS_NUM_NODES: $ECS_NUM_NODES"
echo "ECS_NODE_RANK: $ECS_NODE_RANK"
echo "ECS_MASTER_ADDR: $ECS_MASTER_ADDR"
echo "ECS_MASTER_PORT: $ECS_MASTER_PORT"

export NCCL_DEBUG=INFO

# 设置Llamafactory CLI 所需环境变量
export FORCE_TORCHRUN=1
export NNODES=${ECS_NUM_NODES}
export NODE_RANK=${ECS_NODE_RANK}
export MASTER_ADDR=${ECS_MASTER_ADDR}
export MASTER_PORT=${ECS_MASTER_PORT}

# 启动训练
llamafactory-cli train /workspace/llamafactory-training/LLaMA-Factory/examples/train_full/llama3_full_sft_perf.yaml
# llamafactory-cli train /workspace/llamafactory-training/LLaMA-Factory/examples/train_full/llama3_full_dpo_perf.yaml