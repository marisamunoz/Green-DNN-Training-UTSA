"""
Algorithm 1: Data Parallelism (Baseline)
CS 4953 High Performance Machine Learning
Student: Marisa Munoz
UTSA Spring 2026

Description:
    Trains ResNet18 on CIFAR-10 using PyTorch Distributed Data Parallel (DDP).
    Measures training time, memory usage, and logs power draw via nvidia-smi.

Usage:
    torchrun --nproc_per_node=NUM_GPUS train_ddp.py --epochs 10 --batch_size 128
"""

import os
import time
import argparse
import csv
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


def setup(rank, world_size):
    """Initialize the distributed process group."""
    dist.init_process_group(
        backend="nccl",# NCCL is fastest for GPU-to-GPU communication
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    """Destroy the process group after training."""
    dist.destroy_process_group()


def get_dataloaders(rank, world_size, batch_size):
    """
    Load CIFAR-10 dataset.
    DistributedSampler ensures each GPU gets a non-overlapping subset of data.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # DistributedSampler splits data across GPUs automatically
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, test_loader, train_sampler


def train_one_epoch(model, loader, sampler, optimizer, criterion, epoch, rank):
    """Run one training epoch and return average loss and epoch time."""
    model.train()
    sampler.set_epoch(epoch)# Ensures different shuffling each epoch
    total_loss = 0.0
    start = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(rank), targets.cuda(rank)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_time = time.time() - start
    avg_loss = total_loss / len(loader)
    return avg_loss, epoch_time


def evaluate(model, loader, criterion, rank):
    """Evaluate model on test set, return accuracy and loss."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss


def log_gpu_memory(rank):
    """Log current GPU memory usage in MB."""
    allocated = torch.cuda.memory_allocated(rank) / (1024 ** 2)
    reserved  = torch.cuda.memory_reserved(rank)  / (1024 ** 2)
    return allocated, reserved


def main():
    parser = argparse.ArgumentParser(description="DDP Training: Algorithm 1")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--output_csv", type=str,   default="results_ddp.csv")
    args = parser.parse_args()

    # torchrun sets these environment variables automatically
    rank       = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(rank, world_size)

    # Model
    # ResNet18 pretrained=False training from scratch on CIFAR-10
    model = models.resnet18(weights=None, num_classes=10).cuda(rank)
    model = DDP(model, device_ids=[rank])  # Wrap in DDP

    # Loss/Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Data
    train_loader, test_loader, train_sampler = get_dataloaders(
        rank, world_size, args.batch_size
    )

    # Logging (only rank 0 writes CSV)
    results = []
    total_start = time.time()

    for epoch in range(args.epochs):
        train_loss, epoch_time = train_one_epoch(
            model, train_loader, train_sampler, optimizer, criterion, epoch, rank
        )
        test_acc, test_loss = evaluate(model, test_loader, criterion, rank)
        scheduler.step()

        mem_alloc, mem_reserved = log_gpu_memory(rank)

        if rank == 0:
            print(
                f"Epoch {epoch+1:02d}/{args.epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Mem Alloc: {mem_alloc:.1f}MB"
            )
            results.append({
                "epoch":        epoch + 1,
                "epoch_time_s": round(epoch_time, 4),
                "train_loss":   round(train_loss, 4),
                "test_loss":    round(test_loss, 4),
                "test_acc":     round(test_acc, 2),
                "mem_alloc_mb": round(mem_alloc, 2),
                "mem_reserved_mb": round(mem_reserved, 2),
                "num_gpus":     world_size,
                "batch_size":   args.batch_size,
            })

    total_time = time.time() - total_start

    # Write CSV on rank 0 only
    if rank == 0:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nTotal training time: {total_time:.2f}s")
        print(f"Results saved to {args.output_csv}")

    cleanup()


if __name__ == "__main__":
    main()
