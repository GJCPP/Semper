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
    return torch.round(x * SCALE) / SCALE


class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return quantize(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def q(x):
    return QuantizeSTE.apply(x)

# ============================================================
# Quantized layers
# ============================================================

class QConv2d(nn.Conv2d):
    def forward(self, x):
        x = q(x)
        w = q(self.weight)
        b = q(self.bias) if self.bias is not None else None
        out = F.conv2d(x, w, b, self.stride, self.padding)
        return q(out)


class QLinear(nn.Linear):
    def forward(self, x):
        x = q(x)
        w = q(self.weight)
        b = q(self.bias) if self.bias is not None else None
        out = F.linear(x, w, b)
        return q(out)

# ============================================================
# Quantized VGG11 for CIFAR-10
# ============================================================

class QuantizedVGG11CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            QConv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            QConv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            QConv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            QConv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            QConv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            QConv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5
            QConv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            QConv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            QLinear(512, 512),
            nn.ReLU(inplace=True),
            QLinear(512, 512),
            nn.ReLU(inplace=True),
            QLinear(512, num_classes),
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = q(x)                      # quantize input
        x = self.features(x)
        x = q(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================
# Training
# ============================================================

def train_cnn(batch_size=128, epochs=200, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261)),
    ])

    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = QuantizedVGG11CIFAR().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    optimal_acc = 0.0

    for epoch in range(epochs):
        model.train()
        correct = total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total

        model.eval()
        correct = total = 0
        val_loss = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_loss /= total
        val_acc = 100.0 * correct / total

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}% | Val Loss={val_loss:.4f}")

        if val_acc > optimal_acc:
            optimal_acc = val_acc
        print(f"Optimal Val Acc so far: {optimal_acc:.2f}%")

        scheduler.step()

    return model

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    train_cnn(batch_size=128, epochs=200, lr=0.01)

# Optimal Val Acc so far: 90.10%
