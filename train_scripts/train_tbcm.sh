#/bin/bash
set -e

work_dir=output/debug_sCM_ladd
np=8
master_addr="127.0.0.1"
master_port="29830" 
nodes=4
node_rank=1
python_script=train_scripts/train_tbcm.py

while [[ $# -gt 0 ]]; do
    case $1 in
        --np=*)
            np="${1#*=}"
            shift
            ;;
        --nodes=*)
            nodes="${1#*=}"
            shift
            ;;
        --node_rank=*)
            node_rank="${1#*=}"
            shift
            ;;
        --master_addr=*)
            master_addr="${1#*=}"
            shift
            ;;
        --work_dir=*)
            work_dir="${1#*=}"
            shift
            ;;
        *.yaml)
            config="$1"
            shift
            ;;
        *)
            other_args+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$config" ]]; then
    config=configs/600M_1024px_tbcm.yaml
    echo "Only support .yaml files, use default config=$config"
fi


DATE=$(date '+%Y-%m-%d_%H-%M-%S')
mkdir -p "$work_dir/logs"
log_file="$work_dir/logs/log_${DATE}_rank_${node_rank}.txt"

cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc_per_node=$np --nnodes=$nodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port \
        $python_script \
        --config_path=$config \
        --work_dir=$work_dir \
        --name=tmp \
        --resume_from=latest \
        --report_to=tensorboard \
        --debug=false \
        ${other_args[@]} \
        2>&1 | tee -a $log_file"

echo "$cmd"
eval "$cmd"
