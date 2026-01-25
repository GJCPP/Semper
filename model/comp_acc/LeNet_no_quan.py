import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random

# ============================================================
# Define CNN
# ============================================================

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)

        # Fully connected layers
        # Compute flattened size after conv+pool layers
        self.fc1 = nn.Linear(1024, 640)
        self.fc2 = nn.Linear(640, 128)
        self.fc3 = nn.Linear(128, 10)

        # Initialize weights using Kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # Conv
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ============================================================
# Training function
# ============================================================

def train_cnn(batch_size=64, epochs=6, lr=0.01, patience=10, min_delta=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST mean/std
        ])
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST mean/std
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = LeNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # Early stopping state
    # ============================================================
    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    for epoch in range(epochs):
        # ------------------------
        # Train
        # ------------------------
        model.train()
        total = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total

        # ------------------------
        # Validate
        # ------------------------
        model.eval()
        val_loss = 0.0
        total = correct = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= total
        val_acc = 100.0 * correct / total

        print(
            f"Epoch {epoch+1}: "
            f"Train Acc = {train_acc:.2f}% | "
            f"Val Acc = {val_acc:.2f}% | "
            f"Val Loss = {val_loss:.4f}"
        )

        # ============================================================
        # Early stopping logic
        # ============================================================
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. "
                    f"Best val loss = {best_val_loss:.4f}"
                )
                break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model

if __name__ == "__main__":
    train_cnn(batch_size=64, epochs=6, lr=0.01)

# Epoch 1: Train Acc = 93.97% | Val Acc = 97.82% | Val Loss = 0.0666
# Epoch 2: Train Acc = 98.35% | Val Acc = 98.25% | Val Loss = 0.0520
# Epoch 3: Train Acc = 98.89% | Val Acc = 98.74% | Val Loss = 0.0379
# Epoch 4: Train Acc = 99.21% | Val Acc = 98.37% | Val Loss = 0.0484
# Epoch 5: Train Acc = 99.40% | Val Acc = 98.89% | Val Loss = 0.0368
# Epoch 6: Train Acc = 99.59% | Val Acc = 98.70% | Val Loss = 0.0395
