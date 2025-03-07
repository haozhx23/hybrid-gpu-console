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
    ASSIGNED = "ASSIGNED"
    UNKNOWN = "UNKNOWN"



@dataclass
class NodeInfo:
    name: str
    ip: str
    ibdev: List[str]


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
        
        # Initialize nodes from config
        self.nodes = {
            name: NodeInfo(
                name=name,
                ip=info['ip'],
                ibdev=info['ibdev']
            )
            for name, info in self.node_config.items()
        }
        
        # Initialize DynamoDB table if needed
        self.node_table_name = os.environ.get('NODE_MANAGE_TABLE', 'node_status_table')
        
        # Initialize ECS client and sync node status with DDB
        try:
            cluster_name = os.environ.get('CLUSTER_NAME', 'default-cluster')
            ecs_client = boto3.client('ecs')
            container_instance_arns = []
            paginator = ecs_client.get_paginator('list_container_instances')

            for page in paginator.paginate(cluster=cluster_name):
                container_instance_arns.extend(page['containerInstanceArns'])

            if container_instance_arns:
                desp_response = ecs_client.describe_container_instances(
                        cluster=cluster_name,
                        containerInstances=container_instance_arns,
                        include=['TAGS']  # Include tags in the response
                    )

                for i, inst_arn in enumerate(container_instance_arns):

                    for attrdict in desp_response['containerInstances'][i]['attributes']:
                        if attrdict['name'] == 'node_name':
                            node_name = attrdict['value']
                            if node_name in self.nodes:
                                item = DynamoDBHandler.get_item(
                                    table_name=self.node_table_name,
                                    key={'node_name': node_name}
                                )
                                if not item:
                                    # Use DynamoDBHandler to write item
                                    DynamoDBHandler.write_item(
                                        table_name=self.node_table_name,
                                        item={
                                            'node_name': node_name,
                                            'container_instance_id': inst_arn.split('/')[-1],
                                            'container_instance_arn': inst_arn,
                                            'cluster_name': cluster_name,
                                            'node_status': UserNodeStatus.AVAILABLE.value,
                                            'ip': self.nodes.get(node_name).ip,
                                            'ibdev': self.nodes.get(node_name).ibdev,
                                            'created_at': datetime.datetime.now().isoformat(),
                                            'updated_at': datetime.datetime.now().isoformat()
                                        }
                                    )
        except Exception as e:
            print(f"Error initializing ECS data: {str(e)}")

        # self.spare_nodes = copy.deepcopy(self.nodes)
        # self.assigned_nodes = {}


        
        # Sync with DDB on startup
        self.refresh_all_node_status()


    def refresh_all_node_status(self) -> None:
        """Refresh all node statuses from DynamoDB"""
        try:
            # Scan the table to get all node statuses
            print('## start scan table ##')
            
            items = DynamoDBHandler.scan_table(table_name=self.node_table_name)
            
            # Create a mapping of node names to statuses
            status_map = {item['node_name']: item['node_status'] for item in items if 'node_name' in item and 'node_status' in item}

            self.spare_nodes = {}
            self.assigned_nodes = {}
            # Update local tracking based on DDB status
            for node_name in status_map.keys():
                if status_map[node_name] == UserNodeStatus.AVAILABLE.value:
                    self.spare_nodes[node_name] = self.nodes[node_name]
                else:
                    self.assigned_nodes[node_name] = self.nodes[node_name]
            
            
            # Update local tracking based on DDB status
            for node_name in self.nodes:
                if node_name in status_map:
                    status = status_map[node_name]
                    if status == UserNodeStatus.ASSIGNED.value and node_name in self.spare_nodes:
                        node = self.spare_nodes.pop(node_name)
                        self.assigned_nodes[node_name] = node
                    elif status == UserNodeStatus.AVAILABLE.value and node_name in self.assigned_nodes:
                        node = self.assigned_nodes.pop(node_name)
                        self.spare_nodes[node_name] = node
        except Exception as e:
            print(f"Error refreshing node statuses: {str(e)}")


    def update_node_status(self, node_name: str, status: str) -> bool:
        """Update node status in DynamoDB"""
        try:
            success = DynamoDBHandler.update_item(
                table_name=self.node_table_name,
                key={'node_name': node_name},
                update_expression="SET node_status = :s, updated_at = :t",
                expression_values={
                    ':s': status,
                    ':t': datetime.datetime.now().isoformat()
                }
            )
            if success:
                print(f"Updated node {node_name} status to {status} in DDB")
                return True
            return False
        except Exception as e:
            print(f"Error updating node status in DDB: {str(e)}")
            return False
    
    def get_node_status(self, node_name: str) -> str:
        """Get node status from DynamoDB"""
        
        item = DynamoDBHandler.get_item(
            table_name=self.node_table_name,
            key={'node_name': node_name}
        )

        return item.get('node_status', UserNodeStatus.UNKNOWN.value)



    def get_node_names(self) -> List[str]:
        return list(self.nodes.keys())

    def get_spare_node_names(self) -> List[str]:
        self.refresh_all_node_status()  # Sync with DDB before returning
        return list(self.spare_nodes.keys())

    def get_assigned_node_names(self) -> List[str]:
        self.refresh_all_node_status()  # Sync with DDB before returning
        return list(self.assigned_nodes.keys())

    def assign_node_name(self, node_name: str) -> None:
        node = self.spare_nodes.pop(node_name)
        self.assigned_nodes[node_name] = node
        self.update_node_status(node_name, UserNodeStatus.ASSIGNED.value)

    def assign_a_node_name(self) -> str:
        node_name, node = self.spare_nodes.popitem()
        self.assigned_nodes[node_name] = node
        self.update_node_status(node_name, UserNodeStatus.ASSIGNED.value)
        return node_name

    def release_node_name(self, node_name: str) -> None:
        node = self.assigned_nodes.pop(node_name)
        self.spare_nodes[node_name] = node
        self.update_node_status(node_name, UserNodeStatus.AVAILABLE.value)

    def release_all_node_names(self) -> None:
        for node_name in list(self.assigned_nodes.keys()):
            self.update_node_status(node_name, UserNodeStatus.AVAILABLE.value)
        self.spare_nodes.update(self.assigned_nodes)
        self.assigned_nodes.clear()

    def get_ibdev_list(self, node_name: str) -> List[str]:
        return self.nodes.get(node_name).ibdev

    def get_node_address(self, node_name):
        return self.nodes.get(node_name).ip

    def get_node_status_display(self) -> List[List[str]]:
        """Get node status data for UI display, fetching from DDB"""
        data = []
        for node_name in self.nodes:
            # Get status from DDB
            status = self.get_node_status(node_name)
            is_available = status == UserNodeStatus.AVAILABLE.value
            
            data.append([
                node_name,
                self.get_node_address(node_name),
                f"✅ {status}" if is_available else f"⬜ {status}"
            ])
        return data

    def validate_node_count(self, requested_nodes: int) -> Optional[str]:
        if requested_nodes > len(self.nodes):
            return f"Error: Requested {requested_nodes} nodes but only {len(self.nodes)} available"
        return None
