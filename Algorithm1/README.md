# Algorithm 1: Data Parallelism

Baseline comparison of data parallelism across three frameworks using ResNet18 on CIFAR-10.
All experiments run on NVIDIA A40 GPUs on NCSA Delta.

Frameworks compared: PyTorch DDP, TensorFlow MirroredStrategy, and AxoNN.
Each framework was tested at 1 GPU and 2 GPUs, with 3 runs per configuration.

## Code
 train_ddp.py — PyTorch DDP training script
 train_tf.py — TensorFlow MirroredStrategy training script
 train_axonn.py — AxoNN training script
 submit_all.sh — SLURM job script used on Delta

## Results
CSV results and power logs are in the Results folder, organized by framework.

