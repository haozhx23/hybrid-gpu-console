echo '------------------------------------------------'
echo `pwd`

pip install -U pip
pip install -r requirements.txt
# pip install flash-attn --no-build-isolation

# git clone https://github.com/hiyouga/LLaMA-Factory.git
# cd LLaMA-Factory && pip install -e ".[torch,metrics]"
# cd ..

curl -L https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar -xz s5cmd
chmod +x ./s5cmd

# ./s5cmd cp $MODEL_ID_OR_S3_PATH /tmp/initial-model-path/

# # Copy data from s3 to GPU instance
./s5cmd cp $DATA_S3_PATH /tmp/data-path/

# Run customized script
# os.system('chmod +x compatible_script.sh')
# FORCE_TORCHRUN=1 llamafactory-cli train $CONF_YAML_NAME
# FORCE_TORCHRUN=1 NNODES=$SM_HOST_COUNT NODE_RANK=$SM_CURRENT_HOST_RANK MASTER_ADDR=$SM_MASTER_ADDR MASTER_PORT=$SM_MASTER_PORT llamafactory-cli train $CONF_YAML_NAME

# # # Copy model from 1 GPU instance (if "stage3_gather_16bit_weights_on_model_save": true)
# trained_s3_uri = os.environ['MODEL_SAVE_PATH_S3']
# if 0 == host_rank:
# os.system(f'./s5cmd cp /tmp/tuned-model-path/ {trained_s3_uri}')



CURRENT_NODE_RANK=$((${SM_CURRENT_HOST##*-} - 1))

torchrun --nnodes $NNODES --node_rank $CURRENT_NODE_RANK --nproc_per_node $SM_NUM_GPUS --master_addr algo-1 --master_port 7777 train.py





