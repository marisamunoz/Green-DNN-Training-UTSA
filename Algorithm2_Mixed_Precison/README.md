Algorithm 2: Mixed Precision and Quantization

This folder contains the code and results for Algorithm 2 of the term project,
which compares reduced-precision training techniques on ResNet18 trained on
CIFAR-10. All experiments were run on a single NVIDIA A40 GPU on NCSA Delta.
GPU count is held constant at 1 so results are directly comparable to the 1 GPU
baseline from Algorithm 1. Each method was run 3 times to account for variance.

The four methods are FP16 mixed precision, BF16 mixed precision, INT8
post-training quantization, and knowledge distillation. QAT was attempted but
encountered a compatibility issue with PyTorch 2.6 on the cluster.

Code

train_fp16.py trains ResNet18 using PyTorch AMP with FP16 and GradScaler.
train_bf16.py trains using BF16 precision, which does not need GradScaler.
train_int8.py trains in FP32 then applies dynamic INT8 quantization.
train_distillation.py trains a ResNet18 student from a ResNet50 teacher.

Results

CSV result files and plots are in the Results folder.
Full write-up is the Implementation Paper at the root of this repository.
