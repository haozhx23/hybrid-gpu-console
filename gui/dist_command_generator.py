from typing import Dict, List
import yaml
from node_manager import NodeManager


class DistCommandGenerator:
    def __init__(self):
        # self.node_config = node_config
        self.node_manager = NodeManager()


    def generate_node_entry_script(self, 
                                node_rank: int,
                                num_nodes: int,
                                master_addr: str,
                                master_port: str,
                                entry_script_path: str,
                                node_name: str) -> str:

        # ibdev_list = self.node_config[node_name]['ibdev']
        
        ibdev_list = self.node_manager.get_ibdev_list(node_name)
        ibdev_str = ','.join(ibdev_list)

        node_vars = [
            "#!/bin/bash\n",
            "echo '### Execution of Portal wrapped entry ###'\n", 
            f"chmod +x /workspace/{entry_script_path}",
            # "export NCCL_SOCKET_IFNAME=e",
            "export NCCL_SOCKET_IFNAME=bond0",
            "export NCCL_IB_DISABLE=0",
            "export NCCL_DEBUG=INFO",
            f"export NCCL_IB_HCA={ibdev_str}",
            f"export ECS_NUM_NODES={num_nodes}",
            f"export ECS_NODE_RANK={node_rank}",
            f"export ECS_MASTER_ADDR={master_addr}",
            f"export ECS_MASTER_PORT={master_port}",
            f"/workspace/{entry_script_path}"
        ]

        return "\n".join(node_vars)

        # ]
        
        # torchrun_args = [
        #     "torchrun",
        #     "--nproc_per_node", "8",
        #     f"--nnodes", str(num_nodes),
        #     f"--node_rank={node_rank}",
        #     f"--master_addr='{master_addr}'",
        #     f"--master_port='{master_port}'",
        #     "/workspace/training/train.py"
        # ]

        # FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
        # FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

        llamafactory_torchrun_args = [
            "FORCE_TORCHRUN=1",
            f"NNODES={num_nodes}",
            f"NODE_RANK={node_rank}",
            f"MASTER_ADDR={master_addr}",
            f"MASTER_PORT={master_port}",
            "llamafactory-cli train /workspace/llamafactory-training/LLaMA-Factory/examples/train_full/llama3_full_sft_perf.yaml"
        ]

        whisper_torchrun_args = [
            "torchrun",
            "--nproc_per_node", "8",
            f"--nnodes", str(num_nodes),
            f"--node_rank={node_rank}",
            f"--master_addr='{master_addr}'",
            f"--master_port='{master_port}'",
            "/workspace/whisper-training/train.py"
        ]

        torchrun_args = whisper_torchrun_args

        # env_vars = []
        # torchrun_args = [
        #     "torchrun",
        #     "--nproc_per_node", "4",
        #     f"--nnodes", str(num_nodes),
        #     f"--node_rank={node_rank}",
        #     f"--master_addr='{master_addr}'",
        #     f"--master_port='{master_port}'",
        #     "/workspace/training/train.py"
        # ]

        
        config_args = []
        # for key, value in config.items():
        #     if isinstance(value, bool) and value:
        #         config_args.append(f"--{key}")
        #     else:
        #         config_args.extend([f"--{key}", str(value)])
        
        return " ".join(init_sets + env_vars + torchrun_args + config_args)

    def generate_script_content(self, command: str) -> str:
        return f"#!/bin/bash\n{command}\n"
