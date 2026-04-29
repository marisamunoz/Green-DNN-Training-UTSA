"""
Algorithm 2 Method 5: Knowledge Distillation
CS 4953 High Performance Machine Learning
Marisa Munoz, UTSA Spring 2026

Description:
    Trains a smaller student model (ResNet18) using soft labels from a
    pre-trained teacher model (ResNet50). The student learns from both
    the hard ground-truth labels and the teacher's probability distribution,
    which encodes richer information about class relationships.

    Loss = alpha * CrossEntropy(student, labels)
         + (1 - alpha) * KLDiv(student_soft, teacher_soft)

    Temperature T softens both distributions — higher T reveals more
    information in the teacher's low-confidence predictions.

Usage:
    torchrun --nproc_per_node=1 train_distillation.py --epochs 10 --batch_size 128
"""

import os
import time
import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
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
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader, train_sampler


def distillation_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.7):
    """
    Combined distillation loss.
    T: temperature — higher values soften probability distributions
    alpha: weight on the hard label loss vs soft label loss
    """
    hard_loss = F.cross_entropy(student_logits, targets)

    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha",       type=float, default=0.7)
    parser.add_argument("--output_csv",  type=str,   default="results_distillation.csv")
    args = parser.parse_args()

    rank       = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)

    # Teacher: ResNet50, trained on CIFAR-10 first (frozen during student training)
    teacher = models.resnet50(weights=None, num_classes=10).cuda(rank)

    # Train teacher briefly to get reasonable soft labels
    # In practice a pre-trained teacher would be loaded from disk
    print(f"[Rank {rank}] Pre-training teacher model...")
    t_optimizer = torch.optim.SGD(teacher.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    t_criterion = nn.CrossEntropyLoss()
    teacher.train()

    temp_loader, _, temp_sampler = get_dataloaders(rank, world_size, args.batch_size)
    for t_epoch in range(5):
        temp_sampler.set_epoch(t_epoch)
        for inputs, targets in temp_loader:
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)
            t_optimizer.zero_grad()
            with autocast():
                out = teacher(inputs)
                loss = t_criterion(out, targets)
            loss.backward()
            t_optimizer.step()
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"[Rank {rank}] Teacher pre-training complete.")

    # Student: ResNet18
    student = models.resnet18(weights=None, num_classes=10).cuda(rank)
    student = DDP(student, device_ids=[rank])

    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()

    train_loader, test_loader, train_sampler = get_dataloaders(rank, world_size, args.batch_size)

    results = []
    total_start = time.time()

    for epoch in range(args.epochs):
        student.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        epoch_start = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)
            optimizer.zero_grad()

            with autocast():
                student_logits  = student(inputs)
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                loss = distillation_loss(
                    student_logits, teacher_logits, targets,
                    T=args.temperature, alpha=args.alpha
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss   = total_loss / len(train_loader)
        scheduler.step()

        # Evaluate student
        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(rank), targets.cuda(rank)
                with autocast():
                    outputs = student(inputs)
                _, predicted = outputs.max(1)
                total   += targets.size(0)
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
                "precision":    "Distillation",
                "temperature":  args.temperature,
                "alpha":        args.alpha,
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
