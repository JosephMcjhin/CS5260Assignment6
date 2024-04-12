import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader, distributed
from pathlib import Path
from tqdm import tqdm
import os
import time
import psutil

def main(training_config):
    
    start_time = time.time()

    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss / (1024 * 1024)
    
    checkpoint_path = Path(training_config['checkpoint_path'])
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model = torchvision.models.resnet18(num_classes=10).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

    train_dataloader, test_dataloader = build_dataloaders(training_config['batch_size'])

    if training_config['resume']:
        load_checkpoint(model, optimizer, scheduler, training_config['checkpoint_path'])

    train_model(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, training_config)
    
    final_mem = process.memory_info().rss / (1024 * 1024)
    total_time = time.time() - start_time
    
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Initial memory usage: {initial_mem:.2f} MB")
    print(f"Final memory usage: {final_mem:.2f} MB")
    print(f"Memory used: {final_mem - initial_mem:.2f} MB")

def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, config):
    for epoch in range(config['start_epoch'], config['num_epochs']):
        model.train()
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}") as pbar:
            for images, labels in pbar:
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        scheduler.step()

        if (epoch + 1) % config['save_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, config['checkpoint_path'], epoch)

        evaluate(model, test_dataloader)

def build_dataloaders(batch_size):
    transform_train = transforms.Compose([
        transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()
    data_path = './data'
    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform_train, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform_test, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dataloader, test_dataloader

def evaluate(model, test_dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

def save_checkpoint(model, optimizer, scheduler, path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, f"{path}/benchmark_{epoch + 1}.pth")

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

if __name__ == "__main__":
    training_config = {
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'batch_size': 100,
        'plugin': 'torch_ddp',
        'resume': False,
        'checkpoint_path': './checkpoint',
        'start_epoch': 0,
        'save_interval': 5,
        'target_acc': None
    }
    main(training_config)
