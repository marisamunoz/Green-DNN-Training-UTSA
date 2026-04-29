"""
Algorithm 2 Method 2: BF16 Mixed Precision Training
CS 4953 High Performance Machine Learning
Marisa Munoz, UTSA Spring 2026

Description:
    Trains ResNet18 on CIFAR-10 using BF16 (Brain Float 16) precision.
    BF16 has the same exponent range as FP32 but fewer mantissa bits,
    making it numerically more stable than FP16 without needing GradScaler.
    Supported natively on NVIDIA Ampere GPUs (A40 included).

Usage:
    torchrun --nproc_per_node=1 train_bf16.py --epochs 10 --batch_size 128
"""

import os
import time
import argparse
import csv
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


def setup(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def get_dataloaders(rank, world_size, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return train_loader, test_loader, train_sampler


def main():
    parser = argparse.ArgumentParser(description="BF16 Mixed Precision Training")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--output_csv", type=str,   default="results_bf16.csv")
    args = parser.parse_args()

    rank       = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)

    model = models.resnet18(weights=None, num_classes=10).cuda(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader, test_loader, train_sampler = get_dataloaders(rank, world_size, args.batch_size)

    results = []
    total_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        epoch_start = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)
            optimizer.zero_grad()

            # BF16 does not need GradScaler — its exponent range matches FP32
            with autocast(dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(rank), targets.cuda(rank)
                with autocast(dtype=torch.bfloat16):
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100.0 * correct / total

        mem_mb = torch.cuda.memory_allocated(rank) / (1024 ** 2)

        if rank == 0:
            print(f"Epoch {epoch+1:02d}/{args.epochs} | Time: {epoch_time:.2f}s | "
                  f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Mem: {mem_mb:.1f}MB")
            results.append({
                "epoch":        epoch + 1,
                "epoch_time_s": round(epoch_time, 4),
                "train_loss":   round(avg_loss, 4),
                "test_acc":     round(accuracy, 2),
                "mem_alloc_mb": round(mem_mb, 2),
                "num_gpus":     world_size,
                "batch_size":   args.batch_size,
                "precision":    "BF16",
            })

    if rank == 0:
        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time:.2f}s")
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {args.output_csv}")

    cleanup()


if __name__ == "__main__":
    main()
