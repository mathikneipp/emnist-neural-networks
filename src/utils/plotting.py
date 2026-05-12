import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from ..evaluation.metrics import f1_score


def plot_random_images(
    images, seed=42, n=4, figsize=(8, 8), cmap=None, image_shape=None
):

    images = np.array(images)
    assert len(images) >= n, "No hay suficientes imágenes"

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=n, replace=False)

    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, indices):
        img = images[idx]

        # Caso 1: imagen flattened
        if img.ndim == 1:
            if image_shape is None:
                raise ValueError("Tenés imágenes flatten. Pasá image_shape=(H, W)")
            img = img.reshape(image_shape)

        # Caso 2: (H, W, 1) → lo simplifico
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(img)

        ax.set_title(f"Index {idx}", fontsize=10)
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_training_history(train_loss: list[float], val_loss: list[float]) -> None:
    train_loss = np.asarray(train_loss, dtype=float).reshape(-1)
    val_loss = np.asarray(val_loss, dtype=float).reshape(-1)

    if train_loss.size == 0 and val_loss.size == 0:
        raise ValueError("train_loss y val_loss no pueden estar vacíos")

    max_len = max(train_loss.size, val_loss.size)
    train_x = np.linspace(1, max_len, train_loss.size) if train_loss.size else np.array([])
    val_x = np.linspace(1, max_len, val_loss.size) if val_loss.size else np.array([])

    def smooth(values: np.ndarray) -> np.ndarray:
        if values.size < 10:
            return values
        window = max(3, min(15, values.size // 8))
        kernel = np.ones(window) / window
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    plt.figure(figsize=(11, 6))

    if train_loss.size:
        plt.plot(train_x, train_loss, color="#1f77b4", alpha=0.25, linewidth=1.5)
        plt.plot(train_x, smooth(train_loss), label="Training Loss", color="#1f77b4", linewidth=2.5)

    if val_loss.size:
        plt.plot(val_x, val_loss, color="#d62728", alpha=0.25, linewidth=1.5)
        plt.plot(val_x, smooth(val_loss), label="Validation Loss", color="#d62728", linewidth=2.5)

        best_idx = int(np.argmin(val_loss))
        plt.scatter(
            val_x[best_idx],
            val_loss[best_idx],
            color="#d62728",
            s=55,
            zorder=5,
            label=f"Best Val ({val_loss[best_idx]:.4f})",
        )

    same_length = train_loss.size == val_loss.size
    plt.title("Training History", fontsize=18, pad=14, fontweight="bold")
    plt.xlabel("Epoch" if same_length else "Training Step", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=11, frameon=True)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Evaluates a multiclass classification model and plots:
    - Cross-Entropy Loss
    - Overall Accuracy
    - Macro F1-Score using one-vs-all binary F1
    - Multiclass Confusion Matrices
    """

    def to_numpy(array_like):
        if isinstance(array_like, np.ndarray):
            return array_like
        if torch.is_tensor(array_like):
            return array_like.detach().cpu().numpy()
        return np.asarray(array_like)

    def normalize_targets(y):
        y = to_numpy(y).reshape(-1)
        if np.issubdtype(y.dtype, np.floating):
            y = y.astype(int)
        return y

    def predict_outputs(X):
        """
        Returns model outputs as a NumPy array for both torch and custom models.
        """
        if hasattr(model, "predict"):
            return to_numpy(model.predict(to_numpy(X)))

        if isinstance(model, torch.nn.Module):
            device = next(model.parameters()).device
            X_tensor = X if torch.is_tensor(X) else torch.as_tensor(X, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                return to_numpy(model(X_tensor.to(device)))

        raise TypeError("El modelo debe implementar `predict` o ser un `torch.nn.Module`.")

    def predict_classes(X):
        """
        Converts model outputs into class predictions.
        Supports binary and multiclass outputs.
        """
        y_pred = np.asarray(predict_outputs(X))

        if y_pred.ndim == 1:
            return (y_pred >= 0.5).astype(int)

        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            return (y_pred[:, 0] >= 0.5).astype(int)

        return np.argmax(y_pred, axis=1)

    def compute_macro_f1_ova(y_true, y_pred):
        """
        Computes Macro F1-Score using one-vs-all binary evaluation.
        """
        classes = np.union1d(y_true, y_pred)
        f1_scores = []

        for cls in classes:
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)

            f1_scores.append(f1_score(y_true_binary, y_pred_binary))

        return float(np.mean(f1_scores)), classes

    def compute_accuracy(y_true, y_pred):
        """
        Computes overall multiclass accuracy.
        """
        return float(np.mean(y_true == y_pred))

    def compute_multiclass_confusion_matrix(y_true, y_pred):
        """
        Computes a single multiclass confusion matrix.
        """
        classes = np.union1d(y_true, y_pred)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)), dtype=int)

        for true_label, pred_label in zip(y_true, y_pred):
            cm[class_to_idx[true_label], class_to_idx[pred_label]] += 1

        return cm, classes

    def plot_bar_metric(train_value, val_value, title, ylabel):
        """
        Plots a bar chart comparing train and validation metrics.
        """
        values = [train_value, val_value]
        labels = ["Train", "Validation"]

        plt.figure(figsize=(6.5, 5))

        bars = plt.bar(labels, values, color=["#1f77b4", "#d62728"], width=0.6)

        plt.ylim(0, 1)

        plt.title(title, fontsize=15, fontweight="bold")
        plt.ylabel(ylabel, fontsize=12)

        plt.grid(axis="y", linestyle="--", alpha=0.35)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.4f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(cm, classes, title):
        """
        Plots a single multiclass confusion matrix.
        """
        plt.figure(figsize=(8.5, 7))
        plt.imshow(cm, cmap="Blues")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.xticks(np.arange(len(classes)), classes, rotation=90)
        plt.yticks(np.arange(len(classes)), classes)
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    y_train = normalize_targets(y_train)
    y_val = normalize_targets(y_val)

    # Generate predictions
    y_train_pred = predict_classes(X_train)
    y_val_pred = predict_classes(X_val)

    # Compute overall accuracy
    train_accuracy = compute_accuracy(y_train, y_train_pred)
    val_accuracy = compute_accuracy(y_val, y_val_pred)

    # Compute one-vs-all macro F1
    train_f1, _ = compute_macro_f1_ova(y_train, y_train_pred)
    val_f1, _ = compute_macro_f1_ova(y_val, y_val_pred)

    # Compute multiclass confusion matrices
    train_cm, train_classes = compute_multiclass_confusion_matrix(y_train, y_train_pred)
    val_cm, val_classes = compute_multiclass_confusion_matrix(y_val, y_val_pred)

    # Plot all metrics
    plot_bar_metric(train_accuracy, val_accuracy, "Overall Accuracy", "Accuracy")

    plot_bar_metric(train_f1, val_f1, "Macro F1-Score", "F1-Score")

    plot_confusion_matrix(train_cm, train_classes, "Train Confusion Matrix")

    plot_confusion_matrix(val_cm, val_classes, "Validation Confusion Matrix")


def compare_models(models: list, model_names: list[str] = None):
    """
    Prints and returns a comparison table for trained models.
    """

    rows = []

    for i, model in enumerate(models):

        train_loss = np.array(model.train_loss)
        val_loss = np.array(model.val_loss)

        best_epoch = np.argmin(val_loss)

        rows.append({
            "Model": (
                model_names[i]
                if model_names is not None
                else f"Model_{i}"
            ),

            "Best Val Loss": round(val_loss[best_epoch], 6),
            "Best Train Loss": round(train_loss[best_epoch], 6),
            "Best Epoch": best_epoch + 1,

            "Final Train Loss": round(train_loss[-1], 6),
            "Final Val Loss": round(val_loss[-1], 6),

            "Epochs Trained": len(train_loss)
        })

    df = (
        pd.DataFrame(rows)
        .sort_values(by="Best Val Loss", ascending=True)
        .reset_index(drop=True)
    )

    print(df.to_string(index=False))
