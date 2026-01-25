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
# Fixed-point CNN
# ============================================================

class FixedPointAlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = QConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = QConv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = QConv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = QConv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = QConv2d(128, 256, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = QLinear(4096, 1024)
        self.fc2 = QLinear(1024, 10)

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Quantize input
        x = QuantizeSTE.apply(x)

        # Block 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# Training & evaluation
# ============================================================

def train_cnn(batch_size=64, epochs=100, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = FixedPointAlexNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    optimal_acc = 0.0

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

        if val_acc > optimal_acc:
            optimal_acc = val_acc
        print(f"Optimal Val Acc so far: {optimal_acc:.2f}%")

    return model

if __name__ == "__main__":
    train_cnn(batch_size=64, epochs=100, lr=0.01)
    
# Optimal Val Acc so far: 77.15%
