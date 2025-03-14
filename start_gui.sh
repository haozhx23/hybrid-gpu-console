#!/bin/bash

export USER_NAME='a'
export USER_PASSWORD='a'

# export USER_NAME='cluster-user1'
# export USER_PASSWORD='abc123'

# Default prefill file
export CLUSTER_NAME="nwcd-l4-v1"
export JOB_MANAGE_TABLE="$CLUSTER_NAME-jobs"
export TASK_MANAGE_TABLE="$CLUSTER_NAME-tasks"


export ECS_CLUSTER_CONF_PATH="HYBRID_GPU_PRE_SETTINGS"
export ECS_TASK_DEF="$ECS_CLUSTER_CONF_PATH/ecs_task_def.json"
export TRAINING_CONTAINER_DEF="$ECS_CLUSTER_CONF_PATH/training_container_def.json"
export HEALTH_CONTAINER_DEF="$ECS_CLUSTER_CONF_PATH/healthcheck_container_def.json"


# export NODE_NAME_LIST="A800_node001,A800_node002,A800_node003,A800_node004"
export NODE_NAME_LIST="A800-10-204-9-8,A800-10-204-9-9"
export IB_DEV_LIST="mlx5_10,mlx5_11,mlx5_12,mlx5_13"

# Generate port number using hour and minute without leading zeros
# e.g., 06:09 -> 69, then add base port to ensure valid range
HOUR=$(date +%H | sed 's/^0*//')  # Remove leading zeros
MINUTE=$(date +%M | sed 's/^0*//')  # Remove leading zeros
BASE_PORT=6000  # Base port to ensure we're above privileged ports

# Combine hour and minute, then add base port
## Dynamic port number based on timestamp
# TIME_NUM="${HOUR}${MINUTE}"
# PORT=$((BASE_PORT + TIME_NUM))
## static port number
PORT=7789

# Export port for Gradio
export GRADIO_SERVER_PORT=$PORT

echo "Current time: $(date +%H:%M)"
echo "Starting Gradio interface on port: $PORT"
# echo "Access the interface at: http://localhost:$PORT"

# Choose interface version
echo "Using UI appUI.py"
# python gui/appui.py
python gui/appuiv4.py
# python gui/appui-stl.py