#!/bin/bash

# 切换目录并安装依赖
cd /workspace/sample-ddp-training/
# pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# 直接使用节点预定义环境变量
echo "ECS_NUM_NODES: $ECS_NUM_NODES"
echo "ECS_NODE_RANK: $ECS_NODE_RANK"
echo "ECS_MASTER_ADDR: $ECS_MASTER_ADDR"
echo "ECS_MASTER_PORT: $ECS_MASTER_PORT"


echo "##### Trainign data copying from FSx for lustre to each instance local storage #####"
# cp -r /datafiles/large_audio_arrows /localdata/large_audio_arrows
echo "##### Training data copied from FSx for lustre to each instance local storage #####"

export NCCL_DEBUG=INFO

export DEV_SLEEP_SEC=600

torchrun \
    --nproc_per_node 4 \
    --nnodes ${ECS_NUM_NODES} \
    --node_rank ${ECS_NODE_RANK} \
    --master_addr "${ECS_MASTER_ADDR}" \
    --master_port "${ECS_MASTER_PORT}" \
    /workspace/sample-ddp-training/train.py

