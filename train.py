import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    print(type(train_dataset))
    print(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    print(type(train_loader))
    print(train_loader)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # print number of parameters in the model   
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    if total_params < 25000:
        # Save model with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'model_mnist_{timestamp}.pth'
        torch.save(model.state_dict(), save_path)
        print(f'Model saved as {save_path}')
    
if __name__ == '__main__':
    train() 