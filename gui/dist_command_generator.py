from typing import Dict, List
import yaml
from node_manager import NodeManager
import os

class DistCommandGenerator:
    def __init__(self):
        # self.node_config = node_config
        self.node_manager = NodeManager()
        

    def generate_dist_setting(self, 
                                num_nodes: int,
                                submit_history_path: str,
                                health_check: bool
                                ) -> str:
        
        dist_vars = [
            "#!/bin/bash",
            "",
            "echo '#### Start Node IP assignment ####'",
            f"chmod +x /workspace/PortalScripts/dynamic_addr_assign.sh",
            f"export DIST_CONFIG_PATH=/workspace/{submit_history_path}/node_ips",
            f"mkdir -p $DIST_CONFIG_PATH",
            f"sh /workspace/PortalScripts/dynamic_addr_assign.sh -p $DIST_CONFIG_PATH -n {num_nodes} -w 1800",
            "export CURRENT_NODE_IP=$(hostname -i 2>/dev/null || ip route get 1 | awk '{print $NF;exit}')",
            "export MASTER_NODE_IP=$(cat $DIST_CONFIG_PATH/master_ip)",
            "echo 'Read Master IP from file: '$MASTER_NODE_IP",
            "echo '#### Finish Node IP assignment ####'",
            ""
        ]

        if health_check:
            ibdev_str = self.node_manager.node_ibdev_str

            health_vars = ["",
                f"chmod +x /workspace/PortalScripts/create_hostfile.sh",
                f"bash /workspace/PortalScripts/create_hostfile.sh $DIST_CONFIG_PATH",
                f"export IBDEV_STR={ibdev_str}",
                '''
if [ "$CURRENT_NODE_IP" = "$MASTER_NODE_IP" ]; then
    bash /workspace/PortalScripts/healthCheckMainDryrun.sh
else
    bash /workspace/PortalScripts/healthCheckWorkerDryrun.sh
fi
'''
            ]

            dist_vars = dist_vars + health_vars

        # return "\n".join(dist_vars)
        return dist_vars


    def generate_dist_wrapper_script(self, 
                                    num_nodes: int,
                                    master_port: str,
                                    entry_script_path: str,
                                    submit_history_path: str,
                                    health_check: bool
                                ):

        dist_vars = self.generate_dist_setting(
                                num_nodes,
                                submit_history_path,
                                health_check
                                )
        
        train_vars = [
            f"chmod +x /workspace/{entry_script_path}",
            # "export NCCL_SOCKET_IFNAME=e",
            # "export FI_PROVIDER=efa",
            # "export NCCL_PROTO=simple",
            # "export FI_EFA_USE_DEVICE_RDMA=1",
            "export NCCL_SOCKET_IFNAME=bond0",
            "export NCCL_IB_DISABLE=0",
            "export NCCL_DEBUG=INFO",
            f"export NCCL_IB_HCA=$IBDEV_STR",
            f"export ECS_NUM_NODES={num_nodes}",
            # f"export ECS_NODE_RANK={node_rank}",
            f"export ECS_MASTER_ADDR=$MASTER_NODE_IP",
            f"export ECS_MASTER_PORT={master_port}",
            f"/workspace/{entry_script_path}"
        ]

        train_vars = dist_vars + train_vars
        return "\n".join(train_vars)