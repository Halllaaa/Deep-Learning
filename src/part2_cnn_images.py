from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# Global paths and class names
# ============================================================
# Part 2 saves every artifact in the same project output structure:
# models, figures, and CSV/text tables are separated for clarity.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODEL_DIR = OUTPUTS / "models"
FIGURE_DIR = OUTPUTS / "figures"
TABLE_DIR = OUTPUTS / "tables"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# ============================================================
# Device management
# ============================================================
def get_device() -> torch.device:
    # Use CUDA automatically if the laptop has a compatible NVIDIA GPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Manual convolution and pooling operations
# ============================================================
def cross_correlation_2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    # This implements the basic 2D cross-correlation operation by hand.
    # It slides the kernel over the image and computes element-wise products.
    kernel_h, kernel_w = kernel.shape
    out_h = x.shape[0] - kernel_h + 1
    out_w = x.shape[1] - kernel_w + 1
    output = torch.zeros((out_h, out_w), dtype=x.dtype)

    for row in range(out_h):
        for col in range(out_w):
            patch = x[row : row + kernel_h, col : col + kernel_w]
            output[row, col] = torch.sum(patch * kernel)

    return output


def manual_pooling_2d(x: torch.Tensor, kernel_size: int = 2, stride: int = 2, mode: str = "max") -> torch.Tensor:
    # Pooling reduces spatial size. Max pooling keeps the strongest activation;
    # average pooling keeps the mean activation in each local window.
    out_h = (x.shape[0] - kernel_size) // stride + 1
    out_w = (x.shape[1] - kernel_size) // stride + 1
    output = torch.zeros((out_h, out_w), dtype=x.dtype)

    for row in range(out_h):
        for col in range(out_w):
            patch = x[
                row * stride : row * stride + kernel_size,
                col * stride : col * stride + kernel_size,
            ]
            if mode == "max":
                output[row, col] = torch.max(patch)
            elif mode == "avg":
                output[row, col] = torch.mean(patch)
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return output


def output_size(input_size: int, kernel_size: int, padding: int = 0, stride: int = 1) -> int:
    # Standard CNN formula for the output height/width after convolution.
    return ((input_size + 2 * padding - kernel_size) // stride) + 1


def demonstrate_manual_operations() -> None:
    # This function compares our manual calculations with PyTorch operations.
    # It proves that the mathematical explanation and the framework behavior match.
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    kernel = torch.tensor([[1.0, 0.0], [0.0, -1.0]])

    manual_corr = cross_correlation_2d(x, kernel)
    pytorch_corr = torch.nn.functional.conv2d(
        x.view(1, 1, 4, 4),
        kernel.view(1, 1, 2, 2),
    ).squeeze()

    manual_max_pool = manual_pooling_2d(x, mode="max")
    pytorch_max_pool = torch.nn.functional.max_pool2d(x.view(1, 1, 4, 4), kernel_size=2, stride=2).squeeze()

    manual_avg_pool = manual_pooling_2d(x, mode="avg")
    pytorch_avg_pool = torch.nn.functional.avg_pool2d(x.view(1, 1, 4, 4), kernel_size=2, stride=2).squeeze()

    report_path = TABLE_DIR / "part2_manual_operations.txt"
    report_path.write_text(
        "\n".join(
            [
                "Manual 2D cross-correlation and pooling demonstration",
                "",
                f"Input matrix:\n{x}",
                f"Kernel:\n{kernel}",
                f"Manual cross-correlation:\n{manual_corr}",
                f"PyTorch conv2d result:\n{pytorch_corr}",
                f"Manual max-pooling:\n{manual_max_pool}",
                f"PyTorch max-pooling:\n{pytorch_max_pool}",
                f"Manual average-pooling:\n{manual_avg_pool}",
                f"PyTorch average-pooling:\n{pytorch_avg_pool}",
                "",
                f"Convolution output size example for 28x28, kernel=5, padding=0, stride=1: {output_size(28, 5, 0, 1)}",
                f"Convolution output size example for 28x28, kernel=5, padding=2, stride=1: {output_size(28, 5, 2, 1)}",
                f"Convolution output size example for 28x28, kernel=3, padding=1, stride=2: {output_size(28, 3, 1, 2)}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Manual operation comparison saved at: {report_path}")


# ============================================================
# Fashion-MNIST dataset and DataLoaders
# ============================================================
def make_data_loaders(batch_size: int = 128, train_limit: int = 4000, val_limit: int = 1000, test_limit: int = 1000):
    # Fashion-MNIST images are 28x28 grayscale clothing images.
    # ToTensor converts them to tensors; Normalize centers values around 0.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)

    # A fixed seed gives the same train/validation subsets every run.
    # The limits keep training fast enough for a laptop while still meaningful.
    generator = torch.Generator().manual_seed(42)
    shuffled_indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_indices = shuffled_indices[:train_limit]
    val_indices = shuffled_indices[train_limit : train_limit + val_limit]
    test_indices = list(range(min(test_limit, len(test_data))))

    train_loader = DataLoader(Subset(full_train, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_train, val_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(test_data, test_indices), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ============================================================
# Baseline model: MLP on flattened images
# ============================================================
class ImageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # The image is flattened from 28x28 pixels to 784 numbers.
        # This baseline ignores spatial structure, unlike a CNN.
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# CNN model: LeNet-style architecture
# ============================================================
class LeNetCNN(nn.Module):
    def __init__(self, filters: int = 16, padding: int = 2, pooling: str = "max", use_1x1: bool = False):
        super().__init__()
        # The pooling argument lets us compare max-pooling and average-pooling.
        pool_layer = nn.MaxPool2d if pooling == "max" else nn.AvgPool2d
        layers = [
            nn.Conv2d(1, filters, kernel_size=5, padding=padding),
            nn.ReLU(),
            pool_layer(kernel_size=2, stride=2),
        ]

        if use_1x1:
            # A 1x1 convolution mixes channels without changing image size.
            layers.extend([nn.Conv2d(filters, filters, kernel_size=1), nn.ReLU()])

        layers.extend(
            [
                nn.Conv2d(filters, filters * 2, kernel_size=5),
                nn.ReLU(),
                pool_layer(kernel_size=2, stride=2),
            ]
        )
        self.features = nn.Sequential(*layers)

        # A dummy image is passed once to calculate the flatten size automatically.
        # This avoids hard-coding a dimension that changes when padding/filter choices change.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            flattened_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# CNN variant: strided convolutions
# ============================================================
class StridedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # This model reduces image size using stride=2 convolutions instead of pooling.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ============================================================
# Training loop
# ============================================================
def train_model(model, train_loader, val_loader, device, epochs: int = 3, lr: float = 1e-3):
    # CrossEntropyLoss is the standard loss for multi-class classification.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_state = None
    best_val_accuracy = 0.0
    model.to(device)
    start = perf_counter()

    for epoch in range(1, epochs + 1):
        # Training mode enables layers such as Dropout.
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Standard training cycle: forward pass, loss, backward pass, update.
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, device, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Keep the best validation model so final testing uses the best epoch.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
        )

    elapsed = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val_accuracy, elapsed


# ============================================================
# Evaluation loop
# ============================================================
def evaluate_model(model, loader, device, criterion=None):
    # Evaluation collects predictions and true labels without updating weights.
    model.eval()
    predictions = []
    targets = []
    losses = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if criterion is not None:
                losses.append(criterion(logits, labels).item() * images.size(0))
            # The predicted class is the class with the highest logit.
            predicted = torch.argmax(logits, dim=1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

    avg_loss = sum(losses) / len(loader.dataset) if losses else 0.0
    accuracy = accuracy_score(targets, predictions)
    return avg_loss, accuracy, np.array(predictions), np.array(targets)


# ============================================================
# Visualization helpers
# ============================================================
def plot_history(history, name: str):
    # Training curves show whether learning improves or overfits across epochs.
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"part2_{name}_training_curves.png", dpi=160)
    plt.close()


def plot_confusion(y_true, y_pred, name: str):
    # Confusion matrices show which clothing categories are mixed up.
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Confusion matrix - {name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"part2_{name}_confusion_matrix.png", dpi=160)
    plt.close()


def visualize_feature_maps(model, test_loader, device, name: str):
    # Feature maps make CNN internals visible: they show what filters activate on.
    if not hasattr(model, "features"):
        return

    model.eval()
    images, labels = next(iter(test_loader))
    image = images[0:1].to(device)
    label = labels[0].item()

    with torch.no_grad():
        feature_maps = model.features(image).cpu().squeeze(0)

    maps_to_show = min(8, feature_maps.shape[0])
    plt.figure(figsize=(12, 3))
    plt.subplot(1, maps_to_show + 1, 1)
    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f"Input\n{CLASS_NAMES[label]}")
    plt.axis("off")

    for idx in range(maps_to_show):
        plt.subplot(1, maps_to_show + 1, idx + 2)
        plt.imshow(feature_maps[idx], cmap="viridis")
        plt.title(f"Map {idx + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"part2_{name}_feature_maps.png", dpi=160)
    plt.close()


# ============================================================
# Main experiment orchestration
# ============================================================
def run_experiments():
    # Seeds make the comparison between models reproducible.
    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()
    print(f"Using device: {device}")
    demonstrate_manual_operations()
    train_loader, val_loader, test_loader = make_data_loaders()
    print(f"Train/validation/test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    # Each entry is one experiment required for comparing image classifiers.
    experiments = {
        "mlp_flattened_images": ImageMLP,
        "cnn_lenet_max_padding": lambda: LeNetCNN(filters=16, padding=2, pooling="max", use_1x1=False),
        "cnn_avg_pooling": lambda: LeNetCNN(filters=16, padding=2, pooling="avg", use_1x1=False),
        "cnn_more_filters": lambda: LeNetCNN(filters=32, padding=2, pooling="max", use_1x1=False),
        "cnn_with_1x1": lambda: LeNetCNN(filters=16, padding=2, pooling="max", use_1x1=True),
        "cnn_strided": StridedCNN,
    }

    results = []
    best_model = None
    best_name = None
    best_f1 = -1.0

    for name, builder in experiments.items():
        print(f"\n=== Training {name} ===")
        model = builder()
        # CNN variants train longer than the MLP baseline because they have spatial filters.
        epochs = 3 if name == "mlp_flattened_images" else 6
        history, best_val_accuracy, elapsed = train_model(model, train_loader, val_loader, device, epochs=epochs)
        criterion = nn.CrossEntropyLoss()
        test_loss, test_accuracy, y_pred, y_true = evaluate_model(model, test_loader, device, criterion)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        # Save all figures and model checkpoints for the report and reproducibility.
        plot_history(history, name)
        plot_confusion(y_true, y_pred, name)
        visualize_feature_maps(model, test_loader, device, name)

        model_path = MODEL_DIR / f"part2_{name}.pt"
        torch.save({"model_name": name, "state_dict": model.state_dict(), "classes": CLASS_NAMES}, model_path)

        results.append(
            {
                "model": name,
                "best_val_accuracy": best_val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "train_time_seconds": elapsed,
                "model_path": str(model_path),
            }
        )
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

        # Macro F1 treats all 10 classes equally, so it is a fair model ranking metric.
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    # Save the final comparison table used in the report.
    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_path = TABLE_DIR / "part2_cnn_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n=== Results summary ===")
    print(results_df)
    print(f"\nBest model by macro F1: {best_name} ({best_f1:.4f})")
    print(f"Results table saved at: {results_path}")
    print(f"Figures saved in: {FIGURE_DIR}")


if __name__ == "__main__":
    run_experiments()
