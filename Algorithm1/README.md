Algorithm 1: Data Parallelism

This folder contains the code and results for Algorithm 1 of the term project, which compares data parallelism across three frameworks using ResNet18 trained on CIFAR-10. All experiments were run on NVIDIA A40 GPUs on NCSA Delta. Each framework was tested at 1 GPU and 2 GPUs, with 3 runs per configuration to account for variance.

The three frameworks compared are PyTorch Distributed Data Parallel, TensorFlow MirroredStrategy, and AxoNN.

Code

train_ddp.py trains ResNet18 using PyTorch Distributed Data Parallel with NCCL all-reduce.
train_tf.py trains using TensorFlow MirroredStrategy with automatic GPU distribution.
train_axonn.py trains using AxoNN asynchronous MPI-based communication.
submit_all.sh is the SLURM job script used to run all experiments on Delta.

Results

CSV results and power logs are in the Results folder, organized by framework.
Full write-up is the Implementation Paper at the root of this repository.
