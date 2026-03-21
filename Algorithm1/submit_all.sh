#!/bin/bash
#SBATCH --job-name=algo1_all_frameworks
#SBATCH --output=algo1_%j.log
#SBATCH --error=algo1_%j.err
#SBATCH --partition=gpu                  # ARC: use 'gpu' Delta: use 'gpuA40x4'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2                     # Request 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marisa.munoz@my.utsa.edu

echo "Algorithm 1: All 3 Frameworks Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

#Load modules
module load python/3.10
module load cuda/11.8
module load openmpi # needed for AxoNN

# Activate environment if you have one
# source ~/envs/hpml/bin/activate

cd $SLURM_SUBMIT_DIR

# Install AxoNN if not already installed
pip install axonn --quiet 2>/dev/null || echo "AxoNN install failed will use fallback"

echo ""
echo "GPU Info"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv

# Start power logging (all runs)
nvidia-smi --query-gpu=timestamp,name,power.draw,temperature.gpu,utilization.gpu,memory.used \
           --format=csv --loop=5 > power_log_all.csv &
POWER_PID=$!

# METHOD 1: PyTorch DDP
echo ""
echo "Running PyTorch DDP (1 GPU)"
torchrun --nproc_per_node=1 train_ddp.py \
    --epochs 10 --batch_size 128 \
    --output_csv results_pytorch_1gpu.csv

echo ""
echo "Running PyTorch DDP (2 GPUs)"
torchrun --nproc_per_node=2 train_ddp.py \
    --epochs 10 --batch_size 128 \
    --output_csv results_pytorch_2gpu.csv

# METHOD 2: TensorFlow MirroredStrategy
echo ""
echo "Running TensorFlow MirroredStrategy"
python train_tf.py \
    --epochs 10 --batch_size 128 \
    --output_csv results_tensorflow.csv

# METHOD 3: AxoNN
echo ""
echo "Running AxoNN (2 GPUs)"
mpirun -np 2 python train_axonn.py \
    --epochs 10 --batch_size 128 \
    --output_csv results_axonn.csv

# Stop power logging
kill $POWER_PID

echo ""
echo "All experiments complete!"
echo "End: $(date)"
echo ""
echo "Output files:"
ls -lh results_*.csv power_log_*.csv 2>/dev/null
