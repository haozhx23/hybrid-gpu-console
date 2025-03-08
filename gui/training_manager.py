from datetime import datetime
import os
import re
from typing import List, Dict, Any, Tuple, Optional

from file_manager import FileManager
from dist_command_generator import DistCommandGenerator
from node_manager import NodeManager
from task_manager import TaskManager
from job_manager import JobManager
# from job_manager import Job
from health_manager import HealthManager
from ddb_handler import DynamoDBHandler

import boto3
from datetime import datetime
from decimal import Decimal


import subprocess
import json
import boto3


def _convert_floats_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))  # Convert float to string first for precision
    elif isinstance(obj, dict):
        return {k: _convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_decimal(item) for item in obj]
    return obj




class TrainingManager:
    def __init__(self):
        # self.ddb_handler = DynamoDBHandler()
        self.job_ddb_table_name = os.environ.get('JOB_MANAGE_TABLE')
        self.task_ddb_table_name = os.environ.get('TASK_MANAGE_TABLE')
        self.node_manager = NodeManager()
        self.health_manager = HealthManager()
        self.task_manager = TaskManager()
        self.command_generator = DistCommandGenerator()
        # self.nodes = self.node_manager.get_node_names()
        self.job_manager = JobManager()


    def generate_job_id(self, base_job_name):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # output_dir = f"training_output_{timestamp}"
        job_id = f"{base_job_name}-{timestamp}-{os.urandom(4).hex()}"
        exec_history_save_path = f'_submit_history/output-scripts-{job_id}'
        return job_id, exec_history_save_path, timestamp


    def assign_job_nodes(self, num_nodes):
        master_node_name = self.node_manager.assign_a_node_name()

        all_node_names = [master_node_name]

        for _ in range(num_nodes-1):
            node_name = self.node_manager.assign_a_node_name()
            all_node_names.append(node_name)

        return all_node_names



    def generate_node_scripts(self, 
                              all_node_names, 
                              master_port, 
                              user_script_path, 
                              exec_history_save_dir,
                              ui_task_config
                              ):
        
        num_nodes = len(all_node_names)
        master_node_addr = self.node_manager.get_node_address(all_node_names[0])

        all_task_def_paths = []
                    
        for i, node_name in enumerate(all_node_names):
            print(i, node_name)
            
            # assign node name and generate script
            node_wrap_script_path = self.generate_node_training_script(
                node_name, i, num_nodes, master_node_addr, master_port, user_script_path, exec_history_save_dir
            )
            print(i, node_wrap_script_path)
            # node_task_def_path = self.generate_node_task_def(
            #     task_def_template_path, node_name, node_wrap_script_path, ui_task_config, exec_history_save_dir
            # )
            
            node_task_def_path = self.construct_node_task_def(node_name, i, master_port, node_wrap_script_path, ui_task_config, exec_history_save_dir)

            # all_node_names.append(node_name)
            all_task_def_paths.append(node_task_def_path)

        return all_task_def_paths


    def run_all_tasks(self, 
                      job_id,
                      job_timestamp,
                      all_node_names, 
                      all_task_def_paths,
                      exec_history_save_dir
                      ):
        
        num_nodes = len(all_node_names)
        all_commands = []
        container_inst_ids = []
        ecs_task_ids = []
        

        for i, node_name in enumerate(all_node_names):
            reg_task_cmd, exec_task_cmd, container_inst_id, ecs_task_id, cluster_name = self.register_execute_and_record(
                                  job_id,
                                  job_timestamp,
                                  num_nodes,
                                  node_name,
                                  i,
                                  all_task_def_paths[i]
                                  )
            
            all_commands.append(reg_task_cmd)
            all_commands.append(exec_task_cmd)
            container_inst_ids.append(container_inst_id)
            ecs_task_ids.append(ecs_task_id)

        print('all_commands', all_commands)
        history_file = FileManager.create_execution_history(exec_history_save_dir, all_commands)
        print('history_file', history_file)


        ## if Each node is assigned a task, write to job
        if len(ecs_task_ids) == num_nodes:
            self.gather_task_and_record_job(
                job_id, job_timestamp, cluster_name, num_nodes, all_node_names, container_inst_ids, ecs_task_ids
            )

            # for node_name in all_node_names:
            #     self.node_manager.update_node_status(node_name, 'IN_PROGRESS')

        return ecs_task_ids, history_file
        

    def gather_task_and_record_job(self, job_id, job_timestamp, cluster_name, num_nodes, assigned_nodes, container_inst_ids, ecs_task_ids):
        DynamoDBHandler.write_item(table_name = self.job_ddb_table_name, 
                                    item = {
                                        'job_id': job_id,
                                        'job_timestamp': job_timestamp,
                                        'cluster_name': cluster_name,
                                        'num_nodes': num_nodes,
                                        'assigned_nodes': assigned_nodes,
                                        'submittd_container_inst_ids': container_inst_ids,
                                        'submittd_ecs_task_ids': ecs_task_ids,
                                        'updated_at': datetime.now().isoformat(),
                                        'created_at': datetime.now().isoformat(),
                                        'retry': 0,
                                        'job_status': 'IN_PROGRESS',
                                    }
                                )

    
    def register_execute_and_record(self, 
                                  job_id,
                                  job_timestamp,
                                  nnodes,
                                  node_name,
                                  nodei,
                                  task_def_path, 
                                  ):

        task_id, task_def_arn, cluster_name, container_inst_id, reg_result, exec_result, reg_task_cmd, exec_task_cmd = TaskManager.task_register_and_exec(task_def_path)

        resp = DynamoDBHandler.write_item(table_name = self.task_ddb_table_name,
                                    item = {
                                    'ecs_task_id': task_id,
                                    'node_name': node_name,
                                    'node_index_in_job': nodei, #Decimal(rank),
                                    'job_id': job_id,
                                    'job_timestamp': job_timestamp,
                                    'job_num_nodes': nnodes, #Decimal(nnodes),
                                    'task_def_arn': task_def_arn,
                                    'task_def_name': task_def_arn.split(':')[0],
                                    'task_def_revision': task_def_arn.split(':')[-1],
                                    'cluster_name': cluster_name,
                                    'container_inst_id': container_inst_id,
                                    # 'retry': 0,
                                    # 'task_status': 'IN_PROGRESS',
                                    'updated_at': datetime.now().isoformat(),
                                    'created_at': datetime.now().isoformat(),
                                    # 'metadata': _convert_floats_to_decimal({
                                    #     'task_reg_result': reg_result,
                                    #     'task_exec_result': exec_result
                                    # })
                                }
                            )
        
        print('record task resp: ', resp)

        return reg_task_cmd, exec_task_cmd, container_inst_id, task_id, cluster_name



    def generate_node_training_script(self, node_name: str, noderank: int, num_nodes: int, master_addr: str, master_port: str, entry_script_path: str, output_dir: str) -> List[str]:
        print('Assigned node name: ', node_name)
        script_content = self.command_generator.generate_node_entry_script(
            node_rank=noderank,
            num_nodes=num_nodes,
            master_addr=master_addr,
            master_port=master_port,
            entry_script_path=entry_script_path,
            node_name=node_name
        )

        # script_content = self.command_generator.generate_script_content(command)
        script_path = os.path.join(output_dir, f"training-{node_name}.sh")
        
        FileManager.write_script(script_path, script_content)
        return script_path



    def construct_node_task_def(self, node_name: str, node_index: int, master_port: int, train_script_path: str, task_config: Dict[str, str], output_dir: str):
        
        ecs_task_def = self.task_manager.get_ecs_task_def()
        # ecs_task_def['family'] = task_config['family']
        ecs_task_def['placementConstraints'] = [{"type": "memberOf","expression": f"attribute:node_name=={node_name}"}]
        

        training_container_def = self.task_manager.get_training_container_def()
        # training_container_def['image'] = task_config['image']
        training_container_def['portMappings'][0]['containerPort'] = int(master_port)
        training_container_def['portMappings'][0]['hostPort'] = int(master_port)
        # training_container_def['logConfiguration']['options']['awslogs-group'] = task_config['logGroup']
        training_container_def['command'] = ['/workspace/'+train_script_path]


        if task_config['traininghealth_check']:
            health_container_def = self.health_manager.generate_healthcheck_container_def(node_index, dependent=True)
            training_container_def['dependsOn'] = [{"containerName": health_container_def['name'], "condition": "COMPLETE"}]
            training_container_def['essential'] = False
            ecs_task_def['containerDefinitions'] = [health_container_def, training_container_def]
        else:
            ecs_task_def['containerDefinitions'] = [training_container_def]

        
        node_task_def_path = os.path.join(output_dir, f"task_def_{node_name}.json")
        FileManager.save_json(node_task_def_path, ecs_task_def)

        return node_task_def_path


    def get_summary(self, timestamp: str, num_nodes: int, master_port: str, 
                   output_dir: str, entry_script_path: str) -> Dict[str, Any]:
        return {
            "Timestamp": timestamp,
            "Number of Nodes": num_nodes,
            "Master Port": master_port,
            "Execution History Directory": output_dir,
            "User Entry Script Path": os.path.basename(entry_script_path)
        }
