#!/bin/bash

#SBATCH --requeue
#SBATCH --job-name="train"
#SBATCH --partition=a40x
#SBATCH --time=7-00:00:00
#SBATCH --nodes=16
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8         
#SBATCH --cpus-per-task=10
#SBATCH --output=/admin/home-jinwooahn/repos/exp/logs/train.out  # 본인 경로에 맞게 수정
#SBATCH --error=/admin/home-jinwooahn/repos/exp/logs/train.err  # 본인 경로에 맞게 수정
#SBATCH --account=oslo
#SBATCH --exclusive

echo "##############################################################"
lscpu
grep 'cpu cores' /proc/cpuinfo | uniq
echo "##############################################################"

ds_report

# load cuda version 11.8 to use with fused_kernel
module load openmpi
module load cuda/11.8

# For Jinu (본인에게 맞게 변경)
export AWS_CONFIG_FILE=/admin/home-jinwooahn/repos/.aws/config

# NCCL stuff
export NCCL_DEBUG=WARNING
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without the following two NCCL vars set; See https://github.com/NVIDIA/nccl/issues/676
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export PYTHONFAULTHANDLER=1

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"

# Some hostfile stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# setup hostfile
./generate_hostfile.sh
export DLTS_HOSTFILE=/admin/home-jinwooahn/repos/exp/hostfiles/hosts_$SLURM_JOBID  # 본인 경로에 맞게 수정

# cd to gpt-neox dir
TRAIN_DIR='/admin/home-jinwooahn/repos/gpt-neox/'  # 보닌 경로에 맞게 수정
cd $TRAIN_DIR

#TODO: set this up automatically
NUM_NODES=16
NUM_GPUS_PER_NODE=8
export SLURM_NTASKS=$(($NUM_NODES*$NUM_GPUS_PER_NODE))

python ./deepy.py train.py -d configs polyglot-v2/6-9B.yml polyglot-v2/train-polyglot.yml
