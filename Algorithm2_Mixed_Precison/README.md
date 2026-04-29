# Algorithm 2: Mixed Precision and Quantization

Comparison of reduced-precision training techniques on ResNet18/CIFAR-10.
All experiments run on a single NVIDIA A40 GPU on NCSA Delta.
GPU count is held constant at 1 to isolate the effect of precision format,
consistent with the 1 GPU baseline from Algorithm 1.

Each method was run 3 times to account for variance.

## Methods

FP16: Half-precision training using PyTorch AMP with GradScaler.
BF16: Brain Float 16 training, numerically more stable than FP16.
INT8: Post-training dynamic quantization applied after FP32 training.
Distillation: ResNet18 student trained with soft labels from ResNet50 teacher.

Note: Quantization-Aware Training (QAT) was attempted but encountered a
compatibility issue with PyTorch 2.6 fake quantization on the cluster
(zero_point range error). This is a known issue with the fbgemm/x86 backend
on this PyTorch version and is documented in the error logs.

## Code

train_fp16.py — FP16 mixed precision training script
train_bf16.py — BF16 mixed precision training script
train_int8.py — INT8 post-training dynamic quantization
train_distillation.py — Knowledge distillation training script

## Results

CSV results are in the Results folder, organized by method.
Full write-up: see Implementation Paper in Algorithm1 folder.
