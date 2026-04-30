Algorithm 3: ZeRO Memory Optimization

This folder contains the code and results for Algorithm 3 of the term project, which tests DeepSpeed ZeRO optimizer stages 1, 2, and 3 on ResNet18 trained on CIFAR-10. All experiments were run on 2 NVIDIA A40 GPUs on NCSA Delta. Each stage was run 3 times to account for variance.

ZeRO (Zero Redundancy Optimizer) eliminates redundant memory storage in distributed training by partitioning optimizer states, gradients, and model parameters across GPUs instead of replicating them on every GPU.

Stage 1 partitions optimizer states only.
Stage 2 partitions optimizer states and gradients.
Stage 3 partitions optimizer states, gradients, and model parameters.

Code

train_zero.py trains ResNet18 using DeepSpeed ZeRO at the specified stage.
plot_algo3_results.py generates the result figures from the CSV files.

Results

CSV result files and figures are in the Results folder.
Full write-up is the Implementation Paper at the root of this repository.
