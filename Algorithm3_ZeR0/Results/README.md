Algorithm 3 Results

ResNet18 trained on CIFAR-10 using DeepSpeed ZeRO optimization.
Hardware: 2 NVIDIA A40 GPUs on NCSA Delta.
Each configuration was run 3 times to account for variance.

ZeRO Stage 1 runs:
results_zero_stage1_r1.csv
results_zero_stage1_r2.csv
results_zero_stage1_r3.csv

ZeRO Stage 2 runs:
results_zero_stage2_r1.csv
results_zero_stage2_r2.csv
results_zero_stage2_r3.csv

ZeRO Stage 3 runs:
results_zero_stage3_r1.csv
results_zero_stage3_r2.csv
results_zero_stage3_r3.csv

Metrics: epoch time, training loss, test accuracy, memory usage, GPU count, batch size, ZeRO stage.
