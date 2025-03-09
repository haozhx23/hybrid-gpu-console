import gradio as gr
import time
import os
import json
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from threading import Lock

# Import managers
from node_manager import NodeManager
from training_manager import TrainingManager
from health_manager import HealthManager
from job_manager import Job, JobManager
from task_manager import TaskManager
from cloudwatch_manager import CloudWatchManager

# Import UI components
from ui_components import UIComponents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
APP_TITLE = "Hybrid-GPU Training Console"
DEFAULT_PORT = 7860

class EnhancedTrainingGUI:
    """
    Main class for the Enhanced Training GUI application.
    Handles business logic and interfaces with UI components.
    """
    
    def __init__(self):
        """Initialize the GUI application and its components."""
        # Initialize managers
        self.job_manager = JobManager()
        self.health_manager = HealthManager()
        self.cloudwatch_manager = CloudWatchManager()
        self.task_manager = TaskManager()
        self.node_manager = NodeManager()
        
        # Thread safety
        self.submission_lock = Lock()
        
        # Training manager is initialized on demand
        self.training_manager = None
        
        logger.info("EnhancedTrainingGUI initialized")

    def launch_training(self, 
                      base_job_name: str, 
                      num_nodes: int, 
                      master_port: str, 
                      user_script_path: str, 
                      ecs_cluster_name: str,
                      family: str,
                      container_name: str,
                      image: str,
                      container_workdir: str,
                      host_workdir: str,
                      health_check_checkbox: bool,
                      progress=gr.Progress()) -> Tuple[gr.Markdown, List[List[str]]]:
        """
        Launch a training job with the specified configuration.
        
        Args:
            base_job_name: Base name for the job
            num_nodes: Number of nodes to use
            master_port: Port for master node communication
            user_script_path: Path to the user script
            ecs_cluster_name: Name of the ECS cluster
            family: Task family name
            container_name: Container name
            image: Container image
            container_workdir: Working directory in the container
            host_workdir: Working directory on the host
            health_check_checkbox: Whether to perform health checks
            progress: Gradio progress indicator
            
        Returns:
            Tuple of (markdown output, node status data)
        """
        # Check if another submission is in progress
        if not self.submission_lock.acquire(blocking=False):
            logger.warning("Another job submission is in progress")
            return (
                gr.Markdown("⚠️ Another job submission is in progress. Please wait."),
                None
            )

        try:
            logger.info(f"Launching training job: {base_job_name} with {num_nodes} nodes")
            
            # Initialize training manager
            self.training_manager = TrainingManager()
            
            # Initialize progress tracking
            progress(0, desc="Initializing...")
            
            # Collect task configuration
            ui_task_config = {
                'family': family,
                'image': image,
                'traininghealth_check': health_check_checkbox
            }
            
            # Step 1: Generate job ID
            progress(0.2, desc="Generating Job ID...")
            job_id, exec_history_save_dir, job_timestamp = self._generate_job_id(base_job_name)
            
            # Step 2: Assign nodes for the job
            progress(0.3, desc="Assigning nodes...")
            all_node_names = self._assign_job_nodes(num_nodes)
            logger.info(f"Assigned nodes: {all_node_names}")
            
            # Step 3: Setup health check if requested
            if health_check_checkbox:
                progress(0.4, desc="Setting up health check...")
                self._setup_health_check(all_node_names)
            
            # Step 4: Generate task definitions
            progress(0.5, desc="Generating task definitions...")
            all_task_def_paths = self._generate_node_scripts(
                all_node_names,
                master_port,
                user_script_path,
                exec_history_save_dir,
                ui_task_config
            )
            
            # Step 5: Launch tasks
            progress(0.7, desc="Launching training tasks...")
            training_task_ids, history_file_path = self._run_all_tasks(
                job_id,
                job_timestamp,
                all_node_names,
                all_task_def_paths,
                exec_history_save_dir
            )
            
            # Step 6: Refresh node status
            progress(0.9, desc="Refreshing node status...")
            self.node_manager.refresh_all_node_status()
            
            # Step 7: Prepare results
            results = self._prepare_results(
                all_node_names,
                all_task_def_paths,
                training_task_ids,
                history_file_path,
                job_id
            )
            
            # Get node status data for display
            node_data = self.node_manager.get_node_status_display()
            
            progress(1.0, desc="Complete!")
            return (
                gr.Markdown("\n".join(results)),
                node_data
            )
                
        except Exception as e:
            logger.error(f"Error launching training: {str(e)}", exc_info=True)
            return (
                gr.Markdown(f"⚠️ Error: {str(e)}"),
                None
            )
        finally:
            self.submission_lock.release()

    def _generate_job_id(self, base_job_name: str) -> Tuple[str, str, str]:
        """
        Generate a job ID and related paths.
        
        Args:
            base_job_name: Base name for the job
            
        Returns:
            Tuple of (job_id, exec_history_save_dir, job_timestamp)
        """
        try:
            return self.training_manager.generate_job_id(base_job_name)
        except Exception as e:
            logger.error(f"Error generating job ID: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate job ID: {str(e)}")

    def _assign_job_nodes(self, num_nodes: int) -> List[str]:
        """
        Assign nodes for the job.
        
        Args:
            num_nodes: Number of nodes to assign
            
        Returns:
            List of assigned node names
        """
        try:
            return self.training_manager.assign_job_nodes(num_nodes)
        except Exception as e:
            logger.error(f"Error assigning job nodes: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to assign job nodes: {str(e)}")

    def _setup_health_check(self, node_names: List[str]) -> None:
        """
        Setup health check for the specified nodes.
        
        Args:
            node_names: List of node names
        """
        try:
            self.health_manager.setup_connectivity_host_file(node_names)
        except Exception as e:
            logger.error(f"Error setting up health check: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to setup health check: {str(e)}")

    def _generate_node_scripts(self, 
                             node_names: List[str],
                             master_port: str,
                             user_script_path: str,
                             exec_history_save_dir: str,
                             ui_task_config: Dict[str, Any]) -> List[str]:
        """
        Generate scripts for each node.
        
        Args:
            node_names: List of node names
            master_port: Port for master node communication
            user_script_path: Path to the user script
            exec_history_save_dir: Directory to save execution history
            ui_task_config: Task configuration from UI
            
        Returns:
            List of paths to generated task definition files
        """
        try:
            return self.training_manager.generate_node_scripts(
                node_names,
                master_port,
                user_script_path,
                exec_history_save_dir,
                ui_task_config
            )
        except Exception as e:
            logger.error(f"Error generating node scripts: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate node scripts: {str(e)}")

    def _run_all_tasks(self,
                     job_id: str,
                     job_timestamp: str,
                     node_names: List[str],
                     task_def_paths: List[str],
                     exec_history_save_dir: str) -> Tuple[List[str], str]:
        """
        Run all tasks for the job.
        
        Args:
            job_id: Job ID
            job_timestamp: Job timestamp
            node_names: List of node names
            task_def_paths: List of task definition file paths
            exec_history_save_dir: Directory to save execution history
            
        Returns:
            Tuple of (training_task_ids, history_file_path)
        """
        try:
            return self.training_manager.run_all_tasks(
                job_id,
                job_timestamp,
                node_names,
                task_def_paths,
                exec_history_save_dir
            )
        except Exception as e:
            logger.error(f"Error running tasks: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to run tasks: {str(e)}")

    def _prepare_results(self,
                       node_names: List[str],
                       task_def_paths: List[str],
                       training_task_ids: List[str],
                       history_file_path: str,
                       job_id: str) -> List[str]:
        """
        Prepare results for display.
        
        Args:
            node_names: List of node names
            task_def_paths: List of task definition file paths
            training_task_ids: List of training task IDs
            history_file_path: Path to history file
            job_id: Job ID
            
        Returns:
            List of result strings
        """
        results = []
        
        # Add command to list
        for i, node_task_def_path in enumerate(task_def_paths):
            results.append(f"\n🔷 Node: {node_names[i]}")
            results.append(f"\n  └─ Register & Execute: `{node_task_def_path}`")

        results.append(f"\n📝 Execution history saved to: `{history_file_path}`")
        results.append(f"\n🔍 Job ID: {job_id}")
        
        if training_task_ids:
            results.append(f"\n  └─ Task IDs: {', '.join(training_task_ids)}")
            
        return results

    def launch_health_check(self, master_node_name: str, other_nodes_values: str) -> Tuple[str, List[List[str]]]:
        """
        Launch a health check for the specified nodes.
        
        Args:
            master_node_name: Name of the master node
            other_nodes_values: Comma-separated list of other node names
            
        Returns:
            Tuple of (health check output, health check history)
        """
        try:
            logger.info(f"Launching health check for master node: {master_node_name}")
            
            # Parse node names
            all_node_names = [master_node_name]
            
            if other_nodes_values:
                other_nodes_list = other_nodes_values.split(',')
                for node in other_nodes_list:
                    node_name = node.strip()
                    if node_name:
                        all_node_names.append(node_name)
            
            logger.info(f"Health check nodes: {all_node_names}")
            
            # Submit health check
            healthcheck_task_ids = self.health_manager.submit_health_check(all_node_names)
            
            # Prepare output
            output = f"Health check submitted for nodes: {', '.join(all_node_names)}"
            output += f"\nTask IDs: {', '.join(healthcheck_task_ids)}"
            
            # Get health check history
            history = self.health_manager.get_health_check_history()
            
            return output, history
            
        except Exception as e:
            logger.error(f"Error launching health check: {str(e)}", exc_info=True)
            return f"⚠️ Error: {str(e)}", []

    def refresh_job_status(self) -> List[List[str]]:
        """
        Refresh job status data.
        
        Returns:
            List of job status data rows
        """
        try:
            return JobManager.get_jobs_data()
        except Exception as e:
            logger.error(f"Error refreshing job status: {str(e)}", exc_info=True)
            return [["Error", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"Error: {str(e)}", "", ""]]

    def refresh_node_status(self) -> List[List[str]]:
        """
        Refresh node status data.
        
        Returns:
            List of node status data rows
        """
        try:
            # First refresh all node statuses from DynamoDB
            self.node_manager.refresh_all_node_status()
            # Then get the updated status data for UI display
            return self.node_manager.get_node_status_display()
        except Exception as e:
            logger.error(f"Error refreshing node status: {str(e)}", exc_info=True)
            return [["Error", "", "", f"Error: {str(e)}"]]

    def release_all_nodes(self) -> List[List[str]]:
        """
        Release all nodes and refresh node status.
        
        Returns:
            List of node status data rows
        """
        try:
            self.node_manager.release_all_node_names()
            self.node_manager.refresh_all_node_status()
            return self.node_manager.get_node_status_display()
        except Exception as e:
            logger.error(f"Error releasing nodes: {str(e)}", exc_info=True)
            return [["Error", "", "", f"Error: {str(e)}"]]

    def view_task_logs(self, task_id: str, log_group: str, container_name: str) -> Tuple[str, str]:
        """
        View logs for a specific task.
        
        Args:
            task_id: Task ID
            log_group: Log group name
            container_name: Container name
            
        Returns:
            Tuple of (task ID, log output)
        """
        try:
            if not task_id:
                return "", "No task ID provided"
            
            # Fetch and return logs
            logs = self.cloudwatch_manager.get_task_logs(task_id, log_group, container_name)
            
            # Escape any existing backticks in the logs and wrap in code block
            escaped_logs = logs.replace('`', '\\`')
            return task_id, f"```\n{escaped_logs}\n```"  # Format logs as code block

        except Exception as e:
            logger.error(f"Error viewing task logs: {str(e)}", exc_info=True)
            return "", f"Error fetching logs: {str(e)}"

def create_interface() -> gr.Blocks:
    """
    Create the main interface for the application.
    
    Returns:
        Gradio Blocks interface
    """
    gui = EnhancedTrainingGUI()
    
    with gr.Blocks(
        title=APP_TITLE,
        css=UIComponents.get_custom_css(),
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
        )
    ) as interface:
        gr.Markdown(
            f"""
            # 🚀 {APP_TITLE}
            ### Distributed Training Management Interface
            """,
            elem_classes=["title"]
        )
        
        with gr.Tabs():
            # Launch Training Tab
            with gr.TabItem("🚀 Training Job"):
                UIComponents.create_training_tab(
                    launch_callback=gui.launch_training,
                    release_all_callback=gui.release_all_nodes,
                    refresh_node_callback=gui.refresh_node_status
                )

            # Health Check Tab
            with gr.TabItem("🏥 Health Check"):
                UIComponents.create_health_check_tab(
                    health_check_callback=gui.launch_health_check
                )

            # Job Status Tab
            with gr.TabItem("📋 Job Status"):
                UIComponents.create_job_status_tab(
                    refresh_callback=gui.refresh_job_status,
                    log_callback=gui.view_task_logs
                )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('GRADIO_SERVER_PORT', DEFAULT_PORT))
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share=True
    )
