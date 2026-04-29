# Algorithm 2 Results

ResNet18 trained on CIFAR-10 using reduced-precision techniques.
Hardware: single NVIDIA A40 GPU on NCSA Delta.
Each method was run 3 times to account for variance.

FP16 runs:
results_fp16_r1.csv
results_fp16_r2.csv
results_fp16_r3.csv

BF16 runs:
results_bf16_r1.csv
results_bf16_r2.csv
results_bf16_r3.csv

INT8 post-training quantization runs:
results_int8_r1.csv
results_int8_r2.csv
results_int8_r3.csv

Knowledge distillation runs:
results_distillation_r1.csv
results_distillation_r2.csv
results_distillation_r3.csv

Metrics: epoch time, training loss, test accuracy, memory usage, precision type, batch size.
