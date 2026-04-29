"""
Algorithm 2 Method 3: INT8 Post-Training Quantization
CS 4953 High Performance Machine Learning
Marisa Munoz, UTSA Spring 2026

Description:
    Trains ResNet18 in FP32 on GPU, then applies INT8 post-training static
    quantization using dynamic quantization which is more compatible across
    PyTorch versions than static quantization.

Usage:
    /usr/bin/python3 train_int8.py --epochs 10 --batch_size 128
"""

import time
import argparse
import csv
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader


def get_dataloaders(batch_size):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader


def evaluate_gpu(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def evaluate_cpu(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.cpu())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.cpu()).sum().item()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="INT8 Post-Training Quantization")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--output_csv", type=str,   default="results_int8_r1.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    train_loader, test_loader = get_dataloaders(args.batch_size)

    model = models.resnet18(weights=None, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    results = []
    total_start = time.time()

    # Phase 1: FP32 training on GPU
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

        accuracy = evaluate_gpu(model, test_loader, device)
        mem_mb   = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        print(f"Epoch {epoch+1:02d}/{args.epochs} | Time: {epoch_time:.2f}s | "
              f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Mem: {mem_mb:.1f}MB")
        results.append({
            "epoch":        epoch + 1,
            "epoch_time_s": round(epoch_time, 4),
            "train_loss":   round(avg_loss, 4),
            "test_acc":     round(accuracy, 2),
            "mem_alloc_mb": round(mem_mb, 2),
            "phase":        "FP32_training",
            "precision":    "INT8_PTQ",
            "batch_size":   args.batch_size,
        })

    fp32_acc = results[-1]["test_acc"]
    print(f"\nFP32 training complete. Final accuracy: {fp32_acc:.2f}%")

    # Phase 2: Dynamic INT8 quantization
    # Dynamic quantization quantizes weights to INT8 statically and activations
    # dynamically at runtime. More compatible than static quantization across
    # PyTorch versions and does not require a calibration dataset.
    print("\nApplying INT8 dynamic quantization...")
    quant_start = time.time()

    model_cpu = model.cpu()
    model_int8 = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    quant_time = time.time() - quant_start
    print(f"Quantization complete in {quant_time:.2f}s")

    int8_acc = evaluate_cpu(model_int8, test_loader)
    print(f"INT8 accuracy: {int8_acc:.2f}%  (FP32 was {fp32_acc:.2f}%)")

    results.append({
        "epoch":        args.epochs,
        "epoch_time_s": round(quant_time, 4),
        "train_loss":   results[-1]["train_loss"],
        "test_acc":     round(int8_acc, 2),
        "mem_alloc_mb": 0.0,
        "phase":        "INT8_quantized",
        "precision":    "INT8_PTQ",
        "batch_size":   args.batch_size,
    })

    total_time = time.time() - total_start
    print(f"\nTotal pipeline time: {total_time:.2f}s")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
