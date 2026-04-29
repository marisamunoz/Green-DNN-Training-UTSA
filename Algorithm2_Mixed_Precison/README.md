# Algorithm 2: Mixed Precision and Quantization

This folder contains the code and results for Algorithm 2 of the term project, which compares reduced-precision training techniques on ResNet18 trained on CIFAR-10. All experiments were run on a single NVIDIA A40 GPU on NCSA Delta. GPU count is held constant at 1 so results are directly comparable to the 1 GPU baseline from Algorithm 1.

Each method was run 3 times to account for variance from cluster scheduling.

The four methods implemented are FP16 mixed precision, BF16 mixed precision, INT8 post-training quantization, and knowledge distillation. Quantization-Aware Training (QAT) was also attempted but ran into a compatibility issue with PyTorch 2.6 fake quantization on the cluster. The error is documented in the job logs.

## Code

train_fp16.py trains ResNet18 using PyTorch AMP with automatic FP16 casting and GradScaler for gradient stability.

train_bf16.py trains using BF16 precision, which has the same exponent range as FP32 and does not need GradScaler.

train_int8.py trains in FP32 then applies dynamic INT8 quantization to the final model.

train_distillation.py trains a ResNet18 student using soft label supervision from a ResNet50 teacher.

## Results

CSV result files and a README are in the Results folder.
