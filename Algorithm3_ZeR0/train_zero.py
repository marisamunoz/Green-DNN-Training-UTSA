"""
Algorithm 3: ZeRO Memory Optimization
CS 4953 High Performance Machine Learning
Marisa Munoz, UTSA Spring 2026

Description:
    Trains ResNet18 on CIFAR-10 using DeepSpeed ZeRO optimizer stages 1, 2, and 3.
    ZeRO partitions optimizer states (Stage 1), gradients (Stage 2), and model
    parameters (Stage 3) across GPUs, eliminating redundant memory storage.

    Stage 1: Optimizer state partitioning
    Stage 2: Stage 1 + gradient partitioning
    Stage 3: Stage 2 + parameter partitioning

Usage:
    deepspeed --num_gpus=2 train_zero.py --zero_stage 1 --epochs 10 --output_csv results_zero_stage1_r1.csv
    deepspeed --num_gpus=2 train_zero.py --zero_stage 2 --epochs 10 --output_csv results_zero_stage2_r1.csv
    deepspeed --num_gpus=2 train_zero.py --zero_stage 3 --epochs 10 --output_csv results_zero_stage3_r1.csv
"""

import os
import time
import argparse
import csv
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, DistributedSampler


def get_dataloaders(batch_size, rank, world_size):
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


def get_ds_config(zero_stage, batch_size, lr):
    """Build DeepSpeed config for the requested ZeRO stage."""
    config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": lr,
                "momentum": 0.9,
                "weight_decay": 5e-4
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": lr,
                "warmup_num_steps": 100,
                "total_num_steps": 5000
            }
        },
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "zero_allow_untested_optimizer": True,
        "fp16": {
            "enabled": False
        },
        "steps_per_print": 1000,
        "wall_clock_breakdown": False
    }

    if zero_stage == 3:
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e7
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6

    return config


def main():
    parser = argparse.ArgumentParser(description="ZeRO Memory Optimization Training")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--output_csv", type=str,   default="results_zero.csv")
    parser.add_argument("--local_rank", type=int,   default=-1)

    # DeepSpeed adds its own args
    import deepspeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed()
    rank       = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)

    # Build model
    model = models.resnet18(weights=None, num_classes=10)
    criterion = nn.CrossEntropyLoss()

    # Build DeepSpeed config
    ds_config = get_ds_config(args.zero_stage, args.batch_size, args.lr)

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    train_loader, test_loader, train_sampler = get_dataloaders(
        args.batch_size, rank, world_size
    )

    results = []
    total_start = time.time()

    for epoch in range(args.epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        epoch_start = time.time()

        for inputs, targets in train_loader:
            inputs = inputs.to(model_engine.local_rank)
            targets = targets.to(model_engine.local_rank)

            outputs = model_engine(inputs)
            loss = criterion(outputs, targets)

            model_engine.backward(loss)
            model_engine.step()
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss   = total_loss / len(train_loader)

        # Evaluate
        model_engine.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs  = inputs.to(model_engine.local_rank)
                targets = targets.to(model_engine.local_rank)
                outputs = model_engine(inputs)
                _, predicted = outputs.max(1)
                total   += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100.0 * correct / total

        mem_mb = torch.cuda.memory_allocated(model_engine.local_rank) / (1024 ** 2)

        if rank == 0:
            print(f"Epoch {epoch+1:02d}/{args.epochs} | Time: {epoch_time:.2f}s | "
                  f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Mem: {mem_mb:.1f}MB | "
                  f"ZeRO Stage: {args.zero_stage}")
            results.append({
                "epoch":        epoch + 1,
                "epoch_time_s": round(epoch_time, 4),
                "train_loss":   round(avg_loss, 4),
                "test_acc":     round(accuracy, 2),
                "mem_alloc_mb": round(mem_mb, 2),
                "num_gpus":     world_size,
                "batch_size":   args.batch_size,
                "zero_stage":   args.zero_stage,
            })

    if rank == 0:
        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time:.2f}s")
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
