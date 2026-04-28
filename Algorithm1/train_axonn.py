"""
Algorithm 1 Method 3: Data Parallelism with AxoNN
CS 4953: High Performance Machine Learning
Student: Marisa Munoz
UTSA Spring 2026

Description:
    Trains ResNet18 on CIFAR-10 using AxoNN's data parallel interface.
    AxoNN is an asynchronous, message-driven parallel framework for
    extreme-scale deep learning developed at Georgia Tech/UTSA.

Installation (To be ran ONCE on cluster before submitting job):
    pip install axonn
    or git clone https://github.com/axonn-ai/axonn.git
    cd axonn && pip install -e .

    AxoNN requires MPI on ARC/Delta:
    module load openmpi
    module load cuda

Usage:
    mpirun -np NUM_GPUS python train_axonn.py --epochs 10 --batch_size 128
"""

import os
import time
import argparse
import csv
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

# AxoNN imports
try:
    from axonn import axonn as ax
    from axonn.intra_layer import Auto_Parallel
    AXONN_AVAILABLE = True
except ImportError:
    print("WARNING: AxoNN not installed. Run: pip install axonn")
    print("Falling back to standard PyTorch for testing purposes.")
    AXONN_AVAILABLE = False


def get_dataloader(batch_size, rank, world_size):
    """Load CIFAR-10 with manual sharding for AxoNN."""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # Manual data sharding across ranks
    train_size = len(train_dataset) // world_size
    start_idx  = rank * train_size
    end_idx    = start_idx + train_size
    indices    = list(range(start_idx, end_idx))
    train_subset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="AxoNN Data Parallel Algorithm 1")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--output_csv", type=str,   default="results_axonn.csv")
    args = parser.parse_args()

    if AXONN_AVAILABLE:
        # AxoNN Initialization
        # AxoNN uses a 3D process grid: (data_parallel, row_tensor, col_tensor)
        # For pure data parallelism, set row and col tensor parallel = 1
        ax.init(
            G_data=int(os.environ.get("WORLD_SIZE", 1)),  # data parallel degree
            G_row_tensor=1,
            G_col_tensor=1
        )
        rank       = ax.config.data_parallel_rank
        world_size = ax.config.G_data
    else:
        # Fallback: plain PyTorch single GPU
        rank       = 0
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)

    # Model
    model = models.resnet18(weights=None, num_classes=10).to(device)

    if AXONN_AVAILABLE:
        # Wrap model with AxoNN's Auto_Parallel for automatic parallelism
        model = Auto_Parallel(model)

    # Loss/Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Data
    train_loader, test_loader = get_dataloader(args.batch_size, rank, world_size)

    # Training loop
    results = []
    total_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss   = total_loss / len(train_loader)
        scheduler.step()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total   += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100.0 * correct / total

        mem_mb = torch.cuda.memory_allocated(rank) / (1024 ** 2)

        if rank == 0:
            print(
                f"Epoch {epoch+1:02d}/{args.epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {accuracy:.2f}% | "
                f"Mem: {mem_mb:.1f}MB"
            )
            results.append({
                "epoch":        epoch + 1,
                "epoch_time_s": round(epoch_time, 4),
                "train_loss":   round(avg_loss, 4),
                "test_acc":     round(accuracy, 2),
                "mem_alloc_mb": round(mem_mb, 2),
                "num_gpus":     world_size,
                "batch_size":   args.batch_size,
                "framework":    "AxoNN"
            })

    total_time = time.time() - total_start

    if rank == 0:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
