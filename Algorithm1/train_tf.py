"""
Algorithm 1 Method 2: Data Parallelism with TensorFlow MirroredStrategy
CS 4953 High Performance Machine Learning
Student: Marisa Munoz
UTSA Spring 2026

Description:
    Trains ResNet18 equivalent on CIFAR-10 using TensorFlow's MirroredStrategy,
    which is TensorFlow's built-in data parallelism across multiple GPUs.
    Mirrors the same experiment as train_ddp.py for direct comparison.

Usage:
    python train_tf.py --epochs 10 --batch_size 128
"""

import os
import time
import argparse
import csv
import tensorflow as tf
import numpy as np

def get_dataset(batch_size):
    """Load and preprocess CIFAR-10 using tf.data pipeline."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    x_train = (x_train - mean) / std
    x_test  = (x_test  - mean) / std

    # Build tf.data datasets
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(50000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds


def build_resnet18(num_classes=10):
    """
    Build a ResNet18-equivalent model using Keras.
    TF doesn't have a native ResNet18 in this case I used ResNet50 scaled down,
    or a simple ResNet-like stack for fair comparison.
    """
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Initial conv
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Residual blocks (simplified ResNet18 structure)
    def res_block(x, filters, stride=1):
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    x = res_block(x, 64)
    x = res_block(x, 64)
    x = res_block(x, 128, stride=2)
    x = res_block(x, 128)
    x = res_block(x, 256, stride=2)
    x = res_block(x, 256)
    x = res_block(x, 512, stride=2)
    x = res_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def main():
    parser = argparse.ArgumentParser(description="TF MirroredStrategy: Algorithm 1")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--output_csv", type=str,   default="results_tf.csv")
    args = parser.parse_args()

    # MirroredStrategy: TF's built-in data parallelism
    # Automatically detects all available GPUs and mirrors the model on each
    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    print(f"Number of GPUs: {num_gpus}")

    # Scale batch size by number of GPUs (global batch size)
    global_batch_size = args.batch_size * num_gpus
    train_ds, test_ds = get_dataset(global_batch_size)

    # Build and compile model inside strategy scope
    with strategy.scope():
        model = build_resnet18(num_classes=10)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=args.lr, momentum=0.9, weight_decay=5e-4
        )
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # Training loop
    results = []
    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        history = model.fit(
            train_ds,
            epochs=1,
            validation_data=test_ds,
            verbose=1
        )
        epoch_time = time.time() - epoch_start

        train_loss = history.history['loss'][0]
        val_acc    = history.history['val_accuracy'][0] * 100
        val_loss   = history.history['val_loss'][0]

        # GPU memory via TF
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        mem_mb = mem_info['current'] / (1024 ** 2)

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Mem: {mem_mb:.1f}MB"
        )

        results.append({
            "epoch":        epoch + 1,
            "epoch_time_s": round(epoch_time, 4),
            "train_loss":   round(train_loss, 4),
            "val_loss":     round(val_loss, 4),
            "val_acc":      round(val_acc, 2),
            "mem_mb":       round(mem_mb, 2),
            "num_gpus":     num_gpus,
            "batch_size":   global_batch_size,
            "framework":    "TensorFlow_MirroredStrategy"
        })

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.2f}s")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
