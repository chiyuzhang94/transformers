#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=64G
#SBATCH --job-name=pre-train
#SBATCH --output=pre-train.out
#SBATCH --account=ctb-mageed
#SBATCH --mail-user=zcy94@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
module load gcc/7.3.0
module load cuda/10.0.130 cudnn
module load openmpi nccl/2.5.6

source ~/roberta/bin/activate
export NCCL_DEBUG=INFO
export NPROC_PER_NODE=2
export HDF5_USE_FILE_LOCKING='FALSE'
export PARENT=`/bin/hostname -s`
export MPORT=13001
export CHILDREN=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $PARENT`
export HOSTLIST="$PARENT $CHILDREN"
echo $HOSTLIST
export WORLD_SIZE=$SLURM_NTASKS

srun distributed_runner.sh
