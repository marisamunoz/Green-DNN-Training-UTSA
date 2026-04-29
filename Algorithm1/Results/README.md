# Algorithm 1 Results

All experimental results from Algorithm 1: Data Parallelism comparison across PyTorch DDP, TensorFlow MirroredStrategy, and AxoNN. Experiments run on NVIDIA A40 GPUs on NCSA Delta. Each configuration was run 3 times to account for variance.

## Figures

fig1_training_time.png — total 10-epoch training time per framework and GPU count
fig2_energy.png — estimated energy consumption (avg power draw x training time)
fig3_accuracy.png — final test accuracy after 10 epochs
fig4_learning_curves.png — test accuracy over training for 2 GPU configurations

## Power Logs

nvidia-smi power logs collected every 5 seconds during training. Used to estimate energy consumption. Named by framework and GPU configuration.

## Result CSVs

Organized by framework in subfolders. Each CSV contains per-epoch metrics: training time, loss, accuracy, memory usage, GPU count, and batch size.
