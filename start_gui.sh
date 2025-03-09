#!/bin/bash

# Default prefill file
export CLUSTER_NAME="nwcd-g5-v1"
export JOB_MANAGE_TABLE="$CLUSTER_NAME-jobs"
export TASK_MANAGE_TABLE="$CLUSTER_NAME-tasks"
# export NODE_MANAGE_TABLE="$CLUSTER_NAME-nodes"

export ECS_CLUSTER_CONF_PATH="HYBRID_GPU_PRE_SETTINGS"
export ECS_TASK_DEF="$ECS_CLUSTER_CONF_PATH/ecs_task_def.json"
export TRAINING_CONTAINER_DEF="$ECS_CLUSTER_CONF_PATH/training_container_def.json"
export HEALTH_CONTAINER_DEF="$ECS_CLUSTER_CONF_PATH/healthcheck_container_def.json"
export NODE_MAPPING_PATH="$ECS_CLUSTER_CONF_PATH/node_mapping_info.yaml"

# Export prefill file path for GUI
# export GUI_PREFILL_PATH="$ECS_CLUSTER_CONF_PATH/gui_prefill_4.yaml"


# Generate port number using hour and minute without leading zeros
# e.g., 06:09 -> 69, then add base port to ensure valid range
HOUR=$(date +%H | sed 's/^0*//')  # Remove leading zeros
MINUTE=$(date +%M | sed 's/^0*//')  # Remove leading zeros
BASE_PORT=6000  # Base port to ensure we're above privileged ports

# Combine hour and minute, then add base port
TIME_NUM="${HOUR}${MINUTE}"
# PORT=$((BASE_PORT + TIME_NUM))
PORT=7789

# Export port for Gradio
export GRADIO_SERVER_PORT=$PORT

echo "Current time: $(date +%H:%M)"
echo "Time number: $TIME_NUM"
echo "Starting Gradio interface on port: $PORT"
echo "Access the interface at: http://localhost:$PORT"

# Choose interface version
echo "Using classic UI (appUI.py) with prefill: $UI_PREFILL"
# python gui/appui.py
python gui/appuiv3.py
# python gui/appui-stl.py