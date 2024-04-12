import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

def main(training_config):
    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    model = torchvision.models.resnet18(num_classes=10)
    model.to(get_accelerator().get_current_device())
    
    optimizer = HybridAdam(model.parameters(), lr=training_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

    train_dataloader, test_dataloader = build_dataloaders(training_config['batch_size'], coordinator)
    booster = setup_booster(model, optimizer, criterion, training_config['plugin'])

    if training_config['resume']:
        load_checkpoint(model, optimizer, lr_scheduler, training_config['checkpoint_path'])

    train_model(model, train_dataloader, test_dataloader, optimizer, criterion, lr_scheduler, booster, coordinator, training_config)

def setup_booster(model, optimizer, criterion, plugin_type):
    plugin = {
        "torch_ddp": TorchDDPPlugin(),
        "low_level_zero": LowLevelZeroPlugin(initial_scale=2**5)
    }[plugin_type]

    booster = Booster(plugin=plugin)
    model, optimizer, criterion = booster.boost(model, optimizer, criterion)
    return booster

def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, lr_scheduler, booster, coordinator, config):
    for epoch in range(config['start_epoch'], config['num_epochs']):
        model.train()
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}") as pbar:
            for images, labels in pbar:
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                booster.backward(loss)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        lr_scheduler.step()

        if (epoch + 1) % config['save_interval'] == 0:
            save_checkpoint(model, optimizer, lr_scheduler, config['checkpoint_path'], epoch)

        evaluate(model, test_dataloader, coordinator)

def build_dataloaders(batch_size, coordinator):
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

def evaluate(model, test_dataloader, coordinator):
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
    if coordinator.is_master():
        print(f"Accuracy: {accuracy * 100:.2f}%")

def save_checkpoint(model, optimizer, lr_scheduler, path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch
    }, f"{path}/model_{epoch + 1}.pth")

def load_checkpoint(model, optimizer, lr_scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

if __name__ == "__main__":
    training_config = {
        'num_epochs': 80,
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
