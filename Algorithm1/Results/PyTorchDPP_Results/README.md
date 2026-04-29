PyTorch DDP Results

ResNet18 trained on CIFAR-10 using PyTorch Distributed Data Parallel.
Hardware: NVIDIA A40 GPUs on NCSA Delta.
Each configuration was run 3 times to account for variance.

1 GPU runs:
results_pytorch_1gpu.csv
results_pytorch_1gpu_r2.csv
results_pytorch_1gpu_r3.csv

2 GPU runs:
results_pytorch_2gpu.csv
results_pytorch_2gpu_r2.csv
results_pytorch_2gpu_r3.csv

Metrics: epoch time, training loss, test loss, test accuracy, memory usage, GPU count, batch size.
