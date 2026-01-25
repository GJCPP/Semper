import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random


# ============================================================
# Fixed-point configuration
# ============================================================

SCALE_BITS = 14
SCALE = 1 << SCALE_BITS


def quantize(x):
    """
    Quantize tensor to 14-bit fixed-point.
    """
    return torch.round(x * SCALE) / SCALE


class QuantizeSTE(torch.autograd.Function):
    """
    Straight-through estimator for fixed-point quantization.
    """
    @staticmethod
    def forward(ctx, x):
        return quantize(x)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradients unchanged
        return grad_output
    
# ============================================================
# Quantized layers
# ============================================================

class QConv2d(nn.Conv2d):
    def forward(self, x):
        x_q = QuantizeSTE.apply(x)
        w_q = QuantizeSTE.apply(self.weight)
        b_q = QuantizeSTE.apply(self.bias) if self.bias is not None else None
        out = F.conv2d(x_q, w_q, b_q, self.stride, self.padding)
        return QuantizeSTE.apply(out)


class QLinear(nn.Linear):
    def forward(self, x):
        x_q = QuantizeSTE.apply(x)
        w_q = QuantizeSTE.apply(self.weight)
        b_q = QuantizeSTE.apply(self.bias) if self.bias is not None else None
        out = F.linear(x_q, w_q, b_q)
        return QuantizeSTE.apply(out)

# ============================================================
# Define CNN
# ============================================================

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = QConv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = QConv2d(6, 16, kernel_size=3, padding=1)

        # Fully connected layers
        # Compute flattened size after conv+pool layers
        self.fc1 = QLinear(1024, 640)
        self.fc2 = QLinear(640, 128)
        self.fc3 = QLinear(128, 10)

        # Initialize weights using Kaiming
        for m in self.modules():
            if isinstance(m, QConv2d) or isinstance(m, QLinear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # Quantize input
        x = QuantizeSTE.apply(x)

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

def train_cnn(batch_size=64, epochs=100, lr=0.01, patience=10, min_delta=1e-4):
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

# Epoch 1: Train Acc = 94.00% | Val Acc = 97.74% | Val Loss = 0.0701
# Epoch 2: Train Acc = 98.34% | Val Acc = 98.28% | Val Loss = 0.0502
# Epoch 3: Train Acc = 98.82% | Val Acc = 98.43% | Val Loss = 0.0429
# Epoch 4: Train Acc = 99.20% | Val Acc = 98.50% | Val Loss = 0.0439
# Epoch 5: Train Acc = 99.42% | Val Acc = 98.89% | Val Loss = 0.0350
# Epoch 6: Train Acc = 99.54% | Val Acc = 98.32% | Val Loss = 0.0534
