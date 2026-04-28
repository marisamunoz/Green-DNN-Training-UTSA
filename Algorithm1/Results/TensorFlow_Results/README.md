# TensorFlow MirroredStrategy Results

ResNet18 trained on CIFAR-10 using TensorFlow MirroredStrategy.
Hardware: NVIDIA A40 GPUs on NCSA Delta.

Each configuration was run 3 times to account for variance.

1 GPU: results_tensorflow_1gpu_r1.csv, results_tensorflow_1gpu_r2.csv, results_tensorflow_1gpu_r3.csv
2 GPU: results_tensorflow.csv, results_tensorflow_r2.csv, results_tensorflow_r3.csv

Metrics: epoch time, training loss, validation loss, validation accuracy, memory usage, GPU count, batch size.
