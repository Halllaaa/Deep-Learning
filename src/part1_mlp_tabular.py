from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Global paths
# ============================================================
# These folders keep the experiment outputs organized:
# - trained models go to outputs/models
# - figures go to outputs/figures
# - result tables go to outputs/tables
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MODEL_DIR = OUTPUTS / "models"
FIGURE_DIR = OUTPUTS / "figures"
TABLE_DIR = OUTPUTS / "tables"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Device management
# ============================================================
def get_device() -> torch.device:
    # Use the GPU if CUDA is available; otherwise use the CPU.
    # All tensors and models must be moved to the same device.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Dataset loading and preprocessing
# ============================================================
def load_and_prepare_data(random_state: int = 42):
    # Breast Cancer Wisconsin is a real tabular binary-classification dataset.
    # Each row has 30 numerical features, and the target is malignant/benign.
    dataset = load_breast_cancer()
    x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    # First split: keep 15% as a final test set.
    # stratify=y preserves the class distribution in each split.
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=random_state,
        stratify=y,
    )

    # Second split: create a validation set from the remaining data.
    # The chosen ratio gives approximately 70% train, 15% validation, 15% test.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.1765,
        random_state=random_state,
        stratify=y_train_full,
    )

    # StandardScaler is fitted only on the training set to avoid data leakage.
    # The same scaler is then applied to validation and test sets.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return {
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "x_train": x_train_scaled.astype(np.float32),
        "x_val": x_val_scaled.astype(np.float32),
        "x_test": x_test_scaled.astype(np.float32),
        "y_train": y_train.to_numpy(dtype=np.int64),
        "y_val": y_val.to_numpy(dtype=np.int64),
        "y_test": y_test.to_numpy(dtype=np.int64),
        "raw_x": x,
        "raw_y": y,
    }


# ============================================================
# PyTorch DataLoaders
# ============================================================
def make_loader(x, y, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    # Neural networks in PyTorch work with tensors.
    # Features are float32; labels are integer class IDs.
    features = torch.tensor(x, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=shuffle)


# ============================================================
# MLP model definitions
# ============================================================
def build_sequential_mlp(input_dim: int, hidden_dim: int = 64) -> nn.Sequential:
    # Version 1 required by the project: a concise MLP using nn.Sequential.
    # Architecture: input -> 64 -> 32 -> 2 logits.
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 2),
    )


class CustomMLP(nn.Module):
    # Version 2 required by the project: a custom class inheriting from nn.Module.
    # This version exposes the forward pass explicitly.
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim // 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass: define how one batch moves through the network.
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation2(x)
        return self.output(x)


# ============================================================
# Weight initialization strategies
# ============================================================
def initialize_model(model: nn.Module, strategy: str) -> None:
    # The assignment asks for at least three initialization strategies.
    # We apply them only to Linear layers because they contain trainable weights.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if strategy == "gaussian":
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                nn.init.zeros_(module.bias)
            elif strategy == "constant":
                nn.init.constant_(module.weight, 0.01)
                nn.init.zeros_(module.bias)
            elif strategy == "xavier":
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            else:
                raise ValueError(f"Unknown initialization strategy: {strategy}")


# ============================================================
# Model inspection
# ============================================================
def inspect_model(model: nn.Module) -> None:
    # named_parameters() shows the trainable weights and biases.
    print("\nNamed parameters:")
    for name, parameter in model.named_parameters():
        print(f"{name:20s} shape={tuple(parameter.shape)} requires_grad={parameter.requires_grad}")

    # state_dict() is what PyTorch saves and reloads for a trained model.
    print("\nState dict keys:")
    for key, value in model.state_dict().items():
        print(f"{key:20s} shape={tuple(value.shape)}")


# ============================================================
# Training loop
# ============================================================
def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_state = None
    best_val_accuracy = 0.0

    model.to(device)
    start = perf_counter()

    for epoch in range(1, epochs + 1):
        # Training mode activates layers such as Dropout.
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            # Standard PyTorch training steps:
            # 1. reset old gradients
            # 2. forward pass
            # 3. compute loss
            # 4. backpropagation
            # 5. optimizer update
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Keep the best validation model, not necessarily the last epoch.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_accuracy={val_accuracy:.4f}"
            )

    elapsed = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_accuracy, elapsed


# ============================================================
# Evaluation loop
# ============================================================
def evaluate_model(model, data_loader, device, criterion=None):
    # Evaluation mode disables Dropout and avoids training-specific behavior.
    model.eval()
    losses = []
    predictions = []
    targets = []

    # no_grad() saves memory because no backpropagation is needed for evaluation.
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)

            if criterion is not None:
                loss = criterion(logits, labels)
                losses.append(loss.item() * features.size(0))

            # The class with the highest logit is the predicted class.
            predicted = torch.argmax(logits, dim=1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

    avg_loss = sum(losses) / len(data_loader.dataset) if losses else 0.0
    accuracy = accuracy_score(targets, predictions)
    return avg_loss, accuracy, np.array(predictions), np.array(targets)


# ============================================================
# Visualization helpers
# ============================================================
def plot_history(history, name: str) -> None:
    # Save loss and validation accuracy curves for the experimental appendix.
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss curves")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.title("Validation accuracy")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}_training_curves.png", dpi=160)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, target_names, name: str) -> None:
    # A confusion matrix shows which classes are correctly or incorrectly predicted.
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}_confusion_matrix.png", dpi=160)
    plt.close()


# ============================================================
# Main experiment orchestration
# ============================================================
def run_experiments():
    # Seeds improve reproducibility of splits, initialization, and training.
    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print(f"Using device: {device}")

    data = load_and_prepare_data()
    input_dim = data["x_train"].shape[1]
    print(f"Dataset: Breast Cancer Wisconsin")
    print(f"Input features: {input_dim}")
    print(f"Train/validation/test sizes: {len(data['y_train'])}/{len(data['y_val'])}/{len(data['y_test'])}")

    train_loader = make_loader(data["x_train"], data["y_train"], shuffle=True)
    val_loader = make_loader(data["x_val"], data["y_val"])
    test_loader = make_loader(data["x_test"], data["y_test"])

    results = []
    histories = {}
    model_builders = {
        "sequential": build_sequential_mlp,
        "custom": CustomMLP,
    }

    # Train every combination:
    # 2 model styles x 3 initialization strategies = 6 experiments.
    for model_type, builder in model_builders.items():
        for init_strategy in ["gaussian", "constant", "xavier"]:
            experiment_name = f"{model_type}_{init_strategy}"
            print(f"\n=== Training {experiment_name} ===")

            model = builder(input_dim)
            initialize_model(model, init_strategy)

            if experiment_name == "custom_xavier":
                inspect_model(model)

            history, best_val_accuracy, elapsed = train_one_model(
                model,
                train_loader,
                val_loader,
                device,
            )
            criterion = nn.CrossEntropyLoss()
            test_loss, test_accuracy, y_pred, y_true = evaluate_model(
                model,
                test_loader,
                device,
                criterion,
            )
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="binary",
                zero_division=0,
            )

            # Save figures for each experiment.
            histories[experiment_name] = history
            plot_history(history, experiment_name)
            plot_confusion_matrix(y_true, y_pred, data["target_names"], experiment_name)

            # Save model checkpoint for later reuse.
            model_path = MODEL_DIR / f"{experiment_name}.pt"
            torch.save(
                {
                    "model_type": model_type,
                    "initialization": init_strategy,
                    "input_dim": int(input_dim),
                    "state_dict": model.state_dict(),
                    "target_names": [str(name) for name in data["target_names"]],
                },
                model_path,
            )

            results.append(
                {
                    "model": model_type,
                    "initialization": init_strategy,
                    "best_val_accuracy": best_val_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "train_time_seconds": elapsed,
                    "model_path": str(model_path),
                }
            )

            print(classification_report(y_true, y_pred, target_names=data["target_names"]))

    # Save all metrics into a CSV table for the report.
    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    results_path = TABLE_DIR / "part1_mlp_results.csv"
    results_df.to_csv(results_path, index=False)

    # Reload the best model to prove that save/reload works correctly.
    best = results_df.iloc[0]
    best_checkpoint = torch.load(best["model_path"], map_location=device)
    best_model = (
        build_sequential_mlp(best_checkpoint["input_dim"])
        if best_checkpoint["model_type"] == "sequential"
        else CustomMLP(best_checkpoint["input_dim"])
    )
    best_model.load_state_dict(best_checkpoint["state_dict"])
    best_model.to(device)
    _, reloaded_accuracy, _, _ = evaluate_model(best_model, test_loader, device)

    print("\n=== Results summary ===")
    print(results_df)
    print(f"\nBest model saved at: {best['model_path']}")
    print(f"Reloaded best model test accuracy: {reloaded_accuracy:.4f}")
    print(f"Results table saved at: {results_path}")
    print(f"Figures saved in: {FIGURE_DIR}")


if __name__ == "__main__":
    run_experiments()
