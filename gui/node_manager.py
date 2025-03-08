from typing import Dict, List, Optional
from dataclasses import dataclass
import copy
from file_manager import FileManager
import os
import datetime
import boto3
from ddb_handler import DynamoDBHandler

from enum import Enum, unique

@unique
class UserNodeStatus(Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    ASSIGNED = "ASSIGNED"
    UNKNOWN = "UNKNOWN"



@dataclass
class NodeInfo:
    name: str
    ip: str
    ibdev: List[str]
    num_gpus: int = 8
    status: bool = False
    container_inst_id: str = ""


def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


@singleton
class NodeManager:
    def __init__(self):
        print('Node Manager initiated..')
        self.node_config = FileManager.load_yaml(os.environ['NODE_MAPPING_PATH'])

        self.cluster_name = os.environ.get('CLUSTER_NAME', 'default-cluster')
        self.ecs_client = boto3.client('ecs')
        
        # Initialize STATIC node information from config
        self.nodes = {
            name: NodeInfo(
                name=name,
                ip=info['ip'],
                ibdev=info['ibdev']
            )
            for name, info in self.node_config.items()
        }

        # self.refresh_all_node_status()
        self.assigned_nodes = set()
        self.spare_nodes = set()
        physical_available_node_names = self.get_physical_available_node_names()
        self.spare_nodes.update(physical_available_node_names)

    
    def get_physical_available_node_names(self) -> List[str]:
        self.refresh_all_node_status()
        physical_available_node_names = list(filter(lambda key: self.nodes[key].status == True, self.nodes.keys()))
        return physical_available_node_names


    # Update DYNAMIC node information from physical node status
    def refresh_all_node_status(self):
        # self.cluster_name
        # self.ecs_client

        self.release_all_node_names()

        container_instance_arns = []
        paginator = self.ecs_client.get_paginator('list_container_instances')

        for page in paginator.paginate(cluster=self.cluster_name):
            container_instance_arns.extend(page['containerInstanceArns'])

        if container_instance_arns:
            desp_response = self.ecs_client.describe_container_instances(
                    cluster=self.cluster_name,
                    containerInstances=container_instance_arns,
                    # include=['TAGS']  # Include tags in the response
                )

            for i, inst_arn in enumerate(container_instance_arns):

                container_instance_id = inst_arn.split('/')[-1]
                node_usable = False

                for attrdict in desp_response['containerInstances'][i]['attributes']:
                    if attrdict['name'] == 'node_name':
                        node_name = attrdict['value']

                        self.nodes[node_name].container_inst_id = container_instance_id
                
                node_physical_status = desp_response['containerInstances'][i]['status']
                

                for item in desp_response['containerInstances'][i]['registeredResources']:
                    if item['name'] == 'GPU':
                        registered_gpu = len(item['stringSetValue'])

                        self.nodes[node_name].num_gpus = registered_gpu

                for item in desp_response['containerInstances'][i]['remainingResources']:
                    if item['name'] == 'GPU':
                        remain_gpu = len(item['stringSetValue'])

                if registered_gpu == remain_gpu and node_physical_status == 'ACTIVE':
                    node_usable = True
                    self.nodes[node_name].status = True
                else:
                    self.nodes[node_name].status = False
                    self.spare_nodes.remove(node_name)

                print(container_instance_id, node_name, node_physical_status, registered_gpu, remain_gpu, node_usable)
        

        

        return

    
    ## Node assignment during node assignment
    ## release above temperary status
    def release_all_node_names(self) -> None:
        self.assigned_nodes.clear()
        self.spare_nodes = set()
        # self.refresh_all_node_status()
        # physical_available_node_names = self.get_physical_available_node_names()
        self.spare_nodes.update(self.nodes.keys())
        return


    def assign_a_node_name(self) -> str:
        node_name = self.spare_nodes.pop()
        self.assigned_nodes.add(node_name)
        # self.update_node_status(node_name, UserNodeStatus.ASSIGNED.value)
        return node_name


    def get_ibdev_list(self, node_name: str) -> List[str]:
        return self.nodes.get(node_name).ibdev

    def get_node_address(self, node_name):
        return self.nodes.get(node_name).ip

    def get_node_status_display(self) -> List[List[str]]:
        """Get node status data for UI display, fetching from DDB"""
    
        data = []
        physical_available_node_names = self.get_physical_available_node_names()

        for node_name in self.nodes.keys():
            is_avl = False
            if node_name in physical_available_node_names:
                is_avl = True
            
            data.append([
                node_name,
                self.nodes[node_name].container_inst_id,
                self.get_node_address(node_name),
                f"✅ AVAILABLE" if is_avl else f"⬜ UNAVAILABLE"
            ])

        return data








    # def get_node_names(self) -> List[str]:
    #     return list(self.nodes.keys())

    # def get_spare_node_names(self) -> List[str]:
    #     self.refresh_all_node_status()
    #     return list(self.spare_nodes.keys())

    # def get_assigned_node_names(self) -> List[str]:
    #     self.refresh_all_node_status()
    #     return list(self.assigned_nodes.keys())

    # def assign_node_name(self, node_name: str) -> None:
    #     node = self.spare_nodes.pop(node_name)
    #     self.assigned_nodes[node_name] = node
    #     self.update_node_status(node_name, UserNodeStatus.ASSIGNED.value)



    # def release_node_name(self, node_name: str) -> None:
    #     node = self.assigned_nodes.pop(node_name)
    #     self.spare_nodes[node_name] = node
    #     self.update_node_status(node_name, UserNodeStatus.AVAILABLE.value)


    # def validate_node_count(self, requested_nodes: int) -> Optional[str]:
    #     if requested_nodes > len(self.nodes):
    #         return f"Error: Requested {requested_nodes} nodes but only {len(self.nodes)} available"
    #     return None