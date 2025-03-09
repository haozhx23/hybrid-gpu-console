import gradio as gr
from typing import Callable, Any, Dict, List, Tuple, Optional
import os
from pathlib import Path

from node_manager import NodeManager
from file_manager import FileManager
from job_manager import JobManager
from task_manager import TaskManager

task_manager = TaskManager()

class UIComponents:
    """
    Class responsible for creating and managing UI components.
    Provides a clean interface between UI elements and business logic.
    """
    
    @staticmethod
    def create_training_tab(
        launch_callback: Callable, 
        refresh_node_callback: Callable, 
        release_all_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Creates the training job configuration tab.
        
        Args:
            launch_callback: Function to call when launching a training job
            refresh_node_callback: Function to call when refreshing node status
            release_all_callback: Optional function to call when releasing all nodes
            
        Returns:
            Dictionary of UI components for reference
        """
        # Main container
        with gr.Blocks() as demo:
            # Launch Configuration Section
            with gr.Column():
                gr.Markdown("## 🚀 Launch Configuration")
                
                # Two-column layout
                with gr.Row():
                    # Left column: Training Configuration and File Paths
                    with gr.Column(scale=1):
                        # Training Configuration
                        with gr.Group():
                            gr.Markdown("### 📊 Training Configs")
                            base_job_name = gr.Textbox(
                                label="Base Job Name",
                                placeholder="torch-job",
                                value="torch-job",
                                info="This will be used as prefix for the job ID",
                                container=False
                            )
                            num_nodes = gr.Number(
                                minimum=1,
                                label="Number of Nodes",
                                value=1,
                                info="Number of nodes to use for distributed training",
                                container=False
                            )
                            master_port = gr.Textbox(
                                label="Master Port",
                                placeholder="10000",
                                value="10000",
                                info="An exclusive port number for inter-node communication",
                                container=False
                            )
                            health_check_checkbox = gr.Checkbox(
                                label="Health check before training job",
                                value=False,
                                info="Compute Instance Health & Connectivity checks"
                            )

                        # File Paths
                        with gr.Group():
                            gr.Markdown("### 📁 File Paths")
                            
                            node_mapping_path = gr.Text(
                                value=UIComponents._get_env_var('ECS_CLUSTER_CONF_PATH', ''),
                                info="ECS Config file incl. task def. container def. and node info.",
                                label="ECS Config Files", 
                                interactive=False
                            )
                            
                            user_script_path = gr.Textbox(
                                label="Customized Entry Script Path",
                                placeholder="train-ddp.sh",
                                value="train-ddp.sh",
                                info="Path to user defined entry script, e.g. pip and torchrun train.py",
                                container=False
                            )

                    # Right column: ECS Task Definition Configs
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 📋 ECS Task Definition Configs")
                            
                            ecs_cluster_name = gr.Text(
                                value=UIComponents._get_env_var('CLUSTER_NAME', ''),
                                info="Name of ECS Cluster Control Plane",
                                label="ECS Cluster Name",
                                interactive=False
                            )
                            
                            # These values would be provided by the business logic layer
                            family = gr.Textbox(
                                label="Task Family",
                                placeholder="training-task-family",
                                value="",  # Will be set by business logic
                                info="`family` field in ECS Task Definition",
                                interactive=False,
                                container=False
                            )
                            
                            container_name = gr.Textbox(
                                label="Container Name",
                                placeholder="training-container",
                                value="",  # Will be set by business logic
                                info="`name` field in ECS Container Definition",
                                interactive=False,
                                container=False
                            )
                            
                            image = gr.Textbox(
                                label="Container Image",
                                placeholder="training-image:latest",
                                value="",  # Will be set by business logic
                                info="`image` field in ECS Container Definition",
                                interactive=False,
                                container=False
                            )
                            
                            container_workdir = gr.Textbox(
                                label="Container Working Directory",
                                placeholder="/workspace",
                                value="/workspace",
                                info="Fixed as `/workspace` dir in container",
                                interactive=False,
                                container=False
                            )

                            host_workdir = gr.Textbox(
                                label="Host Working Directory",
                                placeholder="/path/to/workspace",
                                value=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                info="Host workspace dir that maps to container `/workspace` dir",
                                interactive=False,
                                container=False
                            )

                # Launch button row
                with gr.Row():
                    with gr.Column(scale=5):
                        pass  # Empty column for spacing
                    with gr.Column(scale=1):
                        launch_btn = gr.Button("🚀 Launch Training", variant="primary", min_width=200)

            # Job Submit Status Row
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📊 Job Submit Status")
                    with gr.Tabs() as tabs:
                        with gr.TabItem("📝 Execution Trace"):
                            output_log = gr.Markdown("Click 'Launch Training' to begin.")
                        
                        with gr.TabItem("📡 Node Name Assigned") as node_assignment_tab:
                            with gr.Column():
                                with gr.Row():
                                    refresh_node_btn = gr.Button("🔄 Refresh", variant="secondary")
                                node_status = gr.HTML(
                                    label="Node Status Overview",
                                    # every=30,
                                    value=lambda: UIComponents._create_node_table(refresh_node_callback())
                                )
        
        # Define refresh function
        def refresh_and_format_nodes():
            data = refresh_node_callback()
            return UIComponents._create_node_table(data)

        # Connect event handlers
        launch_btn.click(
            fn=launch_callback,
            inputs=[
                base_job_name,
                num_nodes,
                master_port,
                user_script_path,
                ecs_cluster_name,
                family,
                container_name,
                image,
                container_workdir,
                host_workdir,
                health_check_checkbox
            ],
            outputs=[
                output_log,
                node_status
            ]
        ).then(
            fn=refresh_and_format_nodes,
            outputs=[node_status]
        )

        # Refresh node status button click handler
        refresh_node_btn.click(
            fn=refresh_and_format_nodes,
            outputs=[node_status]
        )

        # Tab selection handler
        node_assignment_tab.select(
            fn=refresh_and_format_nodes,
            outputs=[node_status]
        )

        return {
            "output_log": output_log,
            "node_status": node_status
        }

    @staticmethod
    def create_health_check_tab(health_check_callback: Callable) -> Dict[str, Any]:
        """
        Creates the health check tab.
        
        Args:
            health_check_callback: Function to call when submitting a health check
            
        Returns:
            Dictionary of UI components for reference
        """
        with gr.Row():
            gr.Markdown("Input Master Node Name")

        with gr.Row():
            master_node_name = gr.Textbox(
                label="Master Node Name",
                placeholder="Enter Master node Name",
                container=False,
            )
            
        # Add checkbox group
        with gr.Row():
            gr.Markdown('''Input Other Pairing Nodes (Separated by ",")''')
        
        with gr.Row():
            other_nodes_names = gr.Textbox(
                label="Other Pairing Nodes Names",
                placeholder='''Enter Pairing Nodes (Separated by ",")''',
                container=False,
            )
        
        with gr.Row():
            with gr.Column(scale=3):
                pass  # Empty space for alignment
            with gr.Column(scale=1):
                check_btn = gr.Button("Submit Health Check", variant="primary")
        
        health_output = gr.Markdown()
        health_check_history = gr.Dataframe(
            headers=["Node ID", "Timestamp", "Status"],
            label="Health Check History",
            value=[]
        )

        # Connect event handler
        check_btn.click(
            fn=health_check_callback,
            inputs=[master_node_name, other_nodes_names],
            outputs=[health_output, health_check_history]
        )

        return {
            "health_output": health_output,
            "health_check_history": health_check_history
        }

    @staticmethod
    def create_job_status_tab(refresh_callback: Callable, log_callback: Callable) -> Dict[str, Any]:
        """
        Creates the job status tab.
        
        Args:
            refresh_callback: Function to call when refreshing job status
            log_callback: Function to call when fetching logs
            
        Returns:
            Dictionary of UI components for reference
        """
        with gr.Blocks(css=UIComponents._get_jobtab_css()) as demo:
            with gr.Column():
                # Job status section
                with gr.Blocks(elem_classes="dashboard-card"):
                    with gr.Column():
                        with gr.Row():
                            gr.Markdown("## 📋 作业状态", elem_classes="card-title")
                            
                        with gr.Row():
                            refresh_btn = gr.Button("🔄 Refresh", variant="secondary", elem_classes="action-button")
                        
                        # Job status table
                        job_status = gr.HTML(
                            value=lambda: UIComponents._create_job_table(refresh_callback()),
                            every=30,  # Auto-refresh every 30 seconds
                            elem_classes="status-table"
                        )
                        
                        # Stop job area
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2):
                                job_id_input = gr.Textbox(
                                    label="作业 ID",
                                    placeholder="输入要停止的作业 ID",
                                    interactive=True,
                                    type="text"
                                )
                            with gr.Column(scale=2):
                                pass  # Empty column for spacing
                            with gr.Column(scale=2):
                                pass  # Empty column for spacing
                            with gr.Column(scale=1):
                                stop_job_btn = gr.Button("🛑 停止作业", variant="stop", size="lg")

                # Log viewing section
                with gr.Blocks(elem_classes="dashboard-card"):
                    with gr.Column():
                        gr.Markdown("## 📜 任务日志", elem_classes="card-title")
                        
                        with gr.Row(equal_height=True, variant="compact"):
                            with gr.Column(scale=2):
                                log_group_input = gr.Textbox(
                                    label="日志组",
                                    value=task_manager.training_container_def['logConfiguration']['options']['awslogs-group'],
                                    interactive=False,
                                    type="text"
                                )
                            
                            with gr.Column(scale=2):
                                container_name_input = gr.Textbox(
                                    label="容器名称",
                                    value=task_manager.training_container_def['name'],
                                    interactive=False,
                                    type="text"
                                )
                            
                            with gr.Column(scale=2):
                                task_id_input = gr.Textbox(
                                    label="任务 ID",
                                    placeholder="上表中的任务 ID",
                                    interactive=True,
                                    type="text"
                                )
                            
                            with gr.Column(scale=1):
                                log_refresh_btn = gr.Button("📋 获取日志", variant="primary", size="lg")
                        
                        with gr.Row():
                            log_output = gr.Markdown(elem_classes="log-viewer")

        # Handle manual refresh
        def refresh_and_format():
            return UIComponents._create_job_table(refresh_callback())

        refresh_btn.click(
            fn=refresh_and_format,
            outputs=[job_status]
        )

        # # Handle stop job button click
        # def stop_job_and_refresh(job_id: str):
        #     if not job_id or not job_id.strip():
        #         return "Please enter a job ID", UIComponents._create_job_table(refresh_callback())
            
        #     try:
        #         # This would be handled by the business logic
        #         # For now, we'll just refresh the table
        #         return "", UIComponents._create_job_table(refresh_callback())
        #     except Exception as e:
        #         return f"Error stopping job: {str(e)}", UIComponents._create_job_table(refresh_callback())

        def stop_job_and_refresh(job_id: str):
            if not job_id or not job_id.strip():
                return "Please enter a job ID", UIComponents._create_job_table(refresh_callback())
            try:
                print("job_id to stop: ", job_id)
                success = JobManager.stop_job(job_id.strip())
                if success:
                    return "", UIComponents._create_job_table(refresh_callback())
                return "Failed to stop job", UIComponents._create_job_table(refresh_callback())
            except Exception as e:
                return f"Error stopping job: {str(e)}", UIComponents._create_job_table(refresh_callback())

        stop_job_btn.click(
            fn=stop_job_and_refresh,
            inputs=[job_id_input],
            outputs=[job_id_input, job_status]
        )

        # Handle log fetching
        def fetch_logs(task_id: str, log_group: str, container_name: str) -> Tuple[str, str]:
            if not task_id or not task_id.strip():
                return "", "Please enter a task ID"
            if not log_group or not log_group.strip():
                return "", "Please enter a log group"
            if not container_name or not container_name.strip():
                return "", "Please enter a container name"
            
            # Clean task ID
            task_id = task_id.strip().replace('/', '').replace('\\', '')
            if not task_id:
                return "", "Invalid task ID"
                
            return log_callback(task_id, log_group, container_name)

        log_refresh_btn.click(
            fn=fetch_logs,
            inputs=[task_id_input, log_group_input, container_name_input],
            outputs=[task_id_input, log_output]
        )

        return {
            "job_status": job_status,
            "refresh_btn": refresh_btn,
            "task_id_input": task_id_input,
            "log_output": log_output,
            "log_refresh_btn": log_refresh_btn,
            "log_group_input": log_group_input,
            "container_name_input": container_name_input
        }

    @staticmethod
    def get_custom_css() -> str:
        """Returns custom CSS for the UI."""
        return """
        .container {
            max-width: 1200px !important;
            margin: auto;
        }
        .title {
            text-align: center;
            margin-bottom: 2em;
        }
        .status-ready {
            color: #28a745;
        }
        .status-error {
            color: #dc3545;
        }
        .interactive-table {
            margin: 1em 0;
            width: 100%;
            overflow-x: auto;
        }
        .interactive-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .interactive-table th, .interactive-table td {
            padding: 8px 12px;
            border: 1px solid #ddd;
            text-align: left;
        }
        .interactive-table th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        .selectable-row {
            cursor: pointer;
            user-select: text;
        }
        .selectable-row:hover {
            background-color: #f8f9fa;
        }
        .selectable-row td {
            white-space: pre-wrap;
            word-break: break-word;
        }
        """

    @staticmethod
    def _get_jobtab_css() -> str:
        """Returns CSS specific to the job tab."""
        return """
        /* Card layout */
        .dashboard-card {
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            background-color: white;
            margin-bottom: 20px;
        }
        
        .card-title {
            margin: 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #eaeaea;
            color: #333;
        }
        
        /* Table styles */
        .status-table table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        .status-table th {
            background-color: #f5f7fa;
            color: #506690;
            font-weight: 600;
            text-align: left;
            padding: 12px 15px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .status-table td {
            padding: 10px 15px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .status-table tr:hover {
            background-color: #f8f9fa;
        }
        
        /* Status badges */
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .status-running {
            background-color: #e3f2fd;
            color: #0d47a1;
        }
        
        .status-stopped {
            background-color: #ffebee;
            color: #c62828;
        }
        
        /* Button styles */
        .action-button {
            margin-left: auto;
        }
        
        /* Log viewer */
        .log-viewer {
            height: 300px;
            overflow-y: auto;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            white-space: pre-wrap;
        }

        .task-id-cell {
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .log-error {
            color: #d32f2f;
            background-color: #ffebee;
            padding: 8px;
            border-left: 4px solid #d32f2f;
            margin: 8px 0;
        }
        """

    @staticmethod
    def _create_job_table(data: List[List[str]]) -> str:
        """
        Convert job data into an interactive HTML table.
        
        Args:
            data: List of job data rows
            
        Returns:
            HTML string representing the table
        """
        table_html = """
        <div class="interactive-table">
            <table>
                <thead>
                    <tr>
                        <th>Job ID</th>
                        <th>Timestamp</th>
                        <th>Status</th>
                        <th>Nodes</th>
                        <th>ECS Task ID</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for row in data:
            table_html += f"""
                <tr class="selectable-row">
                    <td>{row[0]}</td>
                    <td>{row[1]}</td>
                    <td>{row[2]}</td>
                    <td>{row[3]}</td>
                    <td>{row[4]}</td>
                </tr>
            """
            
        table_html += """
                </tbody>
            </table>
        </div>
        """
        return table_html

    @staticmethod
    def _create_node_table(data: List[List[str]]) -> str:
        """
        Convert node status data into an interactive HTML table.
        
        Args:
            data: List of node status data rows
            
        Returns:
            HTML string representing the table
        """
        table_html = """
        <div class="interactive-table">
            <table>
                <thead>
                    <tr>
                        <th>Node Name</th>
                        <th>Container Inst. ID</th>
                        <th>IP Address</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for row in data:
            table_html += f"""
                <tr class="selectable-row">
                    <td>{row[0]}</td>
                    <td>{row[1]}</td>
                    <td>{row[2]}</td>
                    <td>{row[3]}</td>
                </tr>
            """
            
        table_html += """
                </tbody>
            </table>
        </div>
        """
        return table_html

    @staticmethod
    def _get_env_var(var_name: str, default: str = "") -> str:
        """
        Safely get an environment variable with a default value.
        
        Args:
            var_name: Name of the environment variable
            default: Default value if not found
            
        Returns:
            Value of the environment variable or default
        """
        import os
        return os.environ.get(var_name, default)
