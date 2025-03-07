import gradio as gr
import time
import os
import json
import subprocess
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from threading import Lock
from node_manager import NodeManager, UserNodeStatus
from training_manager import TrainingManager
from health_manager import HealthManager
from job_manager import Job, JobManager
from task_manager import TaskManager
from cloudwatch_manager import CloudWatchManager
from ui_components import (
    create_training_tab,
    create_health_check_tab,
    create_job_status_tab,
    get_custom_css
)

class EnhancedTrainingGUI:
    def __init__(self):
        self.job_manager = JobManager()
        self.health_manager = HealthManager()
        self.cloudwatch_manager = CloudWatchManager()
        self.task_manager = TaskManager()
        self.node_manager = NodeManager()
        self.submission_lock = Lock()
        self.training_manager = None  # Initialize as None
        


    def launch_training(self, 
                    base_job_name: str, 
                    num_nodes: int, 
                    master_port: str, 
                    user_script_path: str, 
                    ecsClusterName: str,
                    family: str,
                    containerName: str,
                    image: str,
                    # logGroup: str,
                    containerWorkdir: str,
                    hostWorkdir: str,
                    health_check_checkbox: bool,
                    progress=gr.Progress()) -> Tuple[gr.Markdown, Optional[Dict], Optional[List]]:

        # ecs_cli_path = 'ecs_task_register_and_launch.sh'

        if not self.submission_lock.acquire(blocking=False):
            return (
                gr.Markdown("⚠️ Another job submission is in progress. Please wait."),
                None,
                None
            )

        try:
            # Initialize manager with node mapping path from UI
            self.training_manager = TrainingManager()

            
            # Initialize progress tracking
            progress(0, desc="Initializing...")
            
            # Generate training scripts
            results = []

            # Collect task configuration from input parameters
            ui_task_config = {
                'family': f'{family}',
                'image': image,
                'traininghealth_check': health_check_checkbox
            }

            progress(0.2, desc="Generate Job ID...")
            # Create job ID
            job_id, exec_history_save_dir, job_timestamp = self.training_manager.generate_job_id(base_job_name)

            # Assign nodes for the job
            all_node_names = self.training_manager.assign_job_nodes(num_nodes)
            print('### all_node_names ### ', all_node_names)

            print(health_check_checkbox)

            if health_check_checkbox:
                print('### setup host file ### ')
                ## launch a dependency task
                self.health_manager.setup_connectivity_host_file(all_node_names)
                print('### gen host file ### ')


            ## trigger training directly
            progress(0.4, desc="Generating task definitions...")
            all_task_def_paths = self.training_manager.generate_node_scripts(
                            all_node_names, 
                            master_port, 
                            user_script_path, 
                            exec_history_save_dir,
                            ui_task_config
                            )

            print('### save exec scripts ### ')


            progress(0.6, desc="Launching training tasks...")
            training_task_ids, history_file_path = self.training_manager.run_all_tasks(
                      job_id,
                      job_timestamp,
                      all_node_names, 
                      all_task_def_paths,
                      exec_history_save_dir
                      )

            
            print(all_node_names, all_task_def_paths, training_task_ids, history_file_path)
            progress(0.8, desc="Training tasks submitted...")
            
            # Refresh node status after job submission
            self.node_manager.refresh_all_node_status()

            # Add command to list
            for i, node_task_def_path in enumerate(all_task_def_paths):
                # run_cmd = f"{ecs_cli_path} {node_task_def_path}"
                results.append(f"\n🔷 Node: {all_node_names[i]}")
                results.append(f"\n  └─ Register & Execute: `{node_task_def_path}`")


            results.append(f"\n📝 Execution history saved to: `{history_file_path}`")
            results.append(f"\n🔍 Job ID: {job_id}")
            if training_task_ids:
                results.append(f"\n  └─ Task IDs: {', '.join(training_task_ids)}")


            # # Generate summary and node status
            # summary = self.training_manager.get_summary(
            #     job_id, num_nodes, master_port, exec_history_save_dir, user_script_path
            # )

            node_data = self.node_manager.get_node_status_display()
            
            progress(1.0, desc="Complete!")
            return (
                gr.Markdown("\n".join(results)),
                # summary,
                node_data
            )
                
        except Exception as e:
            return (
                gr.Markdown(f"⚠️ Error: {str(e)}"),
                # None,
                None
            )
        finally:
            self.submission_lock.release()


    def launch_health_check(self, master_node_name: str, other_nodes_values: str):

        # self.health_manager = HealthManager()
        all_node_names = [master_node_name]

        print("------other_nodes_values: ", other_nodes_values)
        other_nodes_list = other_nodes_values.split(',')

        if len(other_nodes_list) >= 1:
            for i in other_nodes_list:
                all_node_names.append(i.strip())

        healthcheck_task_ids = self.health_manager.submit_health_check(all_node_names)
        print(healthcheck_task_ids)
        return None


    def refresh_job_status(self) -> List[List[str]]:
        """Callback for refreshing job status table."""
        return JobManager.get_jobs_data()

    def refresh_node_status(self) -> List[List[str]]:
        """Callback for refreshing node status without releasing nodes."""
        # First refresh all node statuses from DynamoDB
        self.node_manager.refresh_all_node_status()
        # Then get the updated status data for UI display
        return self.node_manager.get_node_status_display()

    def release_all_nodes(self) -> List[List[str]]:
        self.node_manager.release_all_node_names()
        self.node_manager.refresh_all_node_status()
        return self.node_manager.get_node_status_display()

    def view_task_logs(self, task_id: str, log_group_input: str, container_name_input: str) -> Tuple[str, str]:
        """Callback for viewing task logs."""
        try:
            if not task_id:
                return "", "No task ID provided"
            
            # Fetch and return logs
            logs = self.cloudwatch_manager.get_task_logs(task_id, log_group_input, container_name_input)
            # Escape any existing backticks in the logs and wrap in code block
            escaped_logs = logs.replace('`', '\\`')
            return task_id, f"```\n{escaped_logs}\n```"  # Format logs as code block

        except Exception as e:
            print(f"Log viewing error: {str(e)}")
            return "", f"Error fetching logs: {str(e)}"

def create_interface() -> gr.Blocks:
    gui = EnhancedTrainingGUI()
    
    with gr.Blocks(
        title="Hybrid-GPU Training Console",
        css=get_custom_css(),
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
        )
    ) as interface:
        gr.Markdown(
            """
            # 🚀 Hybrid-GPU Training Console
            ### Distributed Training Management Interface
            """,
            elem_classes=["title"]
        )
        
        with gr.Tabs():
            # Launch Training Tab
            with gr.TabItem("🚀 Training Job"):
                create_training_tab(gui.launch_training, gui.release_all_nodes, gui.refresh_node_status)

            # Health Check Tab
            with gr.TabItem("🏥 Health Check"):
                create_health_check_tab(gui.launch_health_check)

            # Job Status Tab
            with gr.TabItem("📋 Job Status"):
                job_status_components = create_job_status_tab(
                    refresh_callback=gui.refresh_job_status,
                    log_callback=gui.view_task_logs
                )
    
    return interface

if __name__ == "__main__":

    ## Move to CDK
    ##############################
    # from ddb_handler import DynamoDBHandler
    # DynamoDBHandler.create_table_if_not_exists(os.environ['JOB_MANAGE_TABLE'], 'job_id')
    # DynamoDBHandler.create_table_if_not_exists(os.environ['TASK_MANAGE_TABLE'], 'ecs_task_id')
    # time.sleep(10)
    ##############################

    interface = create_interface()
    port = int(os.environ.get('GRADIO_SERVER_PORT', 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
