import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from ..evaluation.metrics import (
    compute_accuracy,
    compute_macro_f1_ova,
    compute_multiclass_confusion_matrix,
)
from ..evaluation.predictions import get_predictions

EMNIST_CLASSES = [
    "0",  # 0
    "1",  # 1
    "2",  # 2
    "3",  # 3
    "4",  # 4
    "5",  # 5
    "6",  # 6
    "7",  # 7
    "8",  # 8
    "9",  # 9
    "A",  # 10
    "B",  # 11
    "C",  # 12
    "D",  # 13
    "E",  # 14
    "F",  # 15
    "G",  # 16
    "H",  # 17
    "I",  # 18
    "J",  # 19
    "K",  # 20
    "L",  # 21
    "M",  # 22
    "N",  # 23
    "O",  # 24
    "P",  # 25
    "Q",  # 26
    "R",  # 27
    "S",  # 28
    "T",  # 29
    "U",  # 30
    "V",  # 31
    "W",  # 32
    "X",  # 33
    "Y",  # 34
    "Z",  # 35
    "a",  # 36
    "b",  # 37
    "d",  # 38
    "e",  # 39
    "f",  # 40
    "g",  # 41
    "h",  # 42
    "n",  # 43
    "q",  # 44
    "r",  # 45
    "t",  # 46
]


def _to_numpy(array_like):
    """
    Convert array-like inputs to NumPy arrays.

    Args:
        array_like: Input object to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    if isinstance(array_like, np.ndarray):
        return array_like
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _normalize_targets(y):
    """
    Convert target labels to a flat integer NumPy array.

    Args:
        y: Target labels.

    Returns:
        np.ndarray: Normalized target labels.
    """
    y = _to_numpy(y).reshape(-1)
    if np.issubdtype(y.dtype, np.floating):
        y = y.astype(int)
    return y


def plot_random_images(
    images, seed=42, n=4, figsize=(8, 8), cmap=None, image_shape=None
):
    """
    Display a random subset of images in a grid.

    Args:
        images: Collection of images to display.
        seed: Random seed used to sample the images.
        n: Number of images to display.
        figsize: Figure size for the plot.
        cmap: Colormap used for grayscale images.
        image_shape: Shape used to unflatten one-dimensional images.
    """

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
    """
    Plot training and validation loss curves.

    Args:
        train_loss (list[float]): Training loss history.
        val_loss (list[float]): Validation loss history.
    """
    train_loss = np.asarray(train_loss, dtype=float).reshape(-1)
    val_loss = np.asarray(val_loss, dtype=float).reshape(-1)

    if train_loss.size == 0 and val_loss.size == 0:
        raise ValueError("train_loss y val_loss no pueden estar vacíos")

    max_len = max(train_loss.size, val_loss.size)
    train_x = (
        np.linspace(1, max_len, train_loss.size) if train_loss.size else np.array([])
    )
    val_x = np.linspace(1, max_len, val_loss.size) if val_loss.size else np.array([])

    def smooth(values: np.ndarray) -> np.ndarray:
        """
        Smooth a one-dimensional curve with a moving average.

        Args:
            values (np.ndarray): Values to smooth.

        Returns:
            np.ndarray: Smoothed values.
        """
        if values.size < 10:
            return values
        window = max(3, min(15, values.size // 8))
        kernel = np.ones(window) / window
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    plt.figure(figsize=(11, 6))

    if train_loss.size:
        plt.plot(train_x, train_loss, color="#1f77b4", alpha=0.25, linewidth=1.5)
        plt.plot(
            train_x,
            smooth(train_loss),
            label="Training Loss",
            color="#1f77b4",
            linewidth=2.5,
        )

    if val_loss.size:
        plt.plot(val_x, val_loss, color="#d62728", alpha=0.25, linewidth=1.5)
        plt.plot(
            val_x,
            smooth(val_loss),
            label="Validation Loss",
            color="#d62728",
            linewidth=2.5,
        )

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


def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    dataset_name: str = "emnist_bymerge",
    val_name: str = "Validation",
    device=None,
):
    """
    Evaluate a multiclass classification model and plot summary metrics.

    Args:
        model: Trained model to evaluate.
        X_train: Training input samples.
        y_train: Training target labels.
        X_val: Validation input samples.
        y_val: Validation target labels.
        dataset_name (str, optional): Dataset name used to select class labels.
            Defaults to `"emnist_bymerge"`.
        val_name (str, optional): Name displayed for the validation split. Defaults to
            `"Validation"`.
        device (optional): Device used for PyTorch-based models. Defaults to None.
    """

    def plot_bar_metric(train_value, val_value, title, ylabel):
        """
        Plot a bar chart comparing training and validation metrics.

        Args:
            train_value: Metric value for the training split.
            val_value: Metric value for the validation split.
            title: Plot title.
            ylabel: Label for the y-axis.
        """
        values = [train_value, val_value]
        labels = ["Train", val_name]

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
        Plot a single multiclass confusion matrix.

        Args:
            cm: Confusion matrix values.
            classes: Class labels represented in the matrix.
            title: Plot title.
        """
        if dataset_name == "emnist_bymerge":
            ticks = EMNIST_CLASSES
        plt.figure(figsize=(8.5, 7))
        plt.imshow(cm, cmap="Blues")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        if ticks:
            tick_positions = np.arange(len(classes))

            plt.xticks(tick_positions, ticks)
            plt.yticks(tick_positions, ticks)

        else:
            plt.xticks(np.arange(len(classes)), classes)
            plt.yticks(np.arange(len(classes)), classes)

        plt.colorbar()

        plt.tight_layout()
        plt.show()

    y_train = _normalize_targets(y_train)
    y_val = _normalize_targets(y_val)

    # Generate predictions
    y_train_pred = get_predictions(model, X_train, device)
    y_val_pred = get_predictions(model, X_val, device)

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

    plot_confusion_matrix(val_cm, val_classes, f"{val_name} Confusion Matrix")


def plot_model_metric_comparison(
    models,
    X,
    y,
    model_names: list[str] = None,
    device=None,
    split_name: str = "Test",
    sort_by: str = "f1_score",
):
    """
    Plot a final comparison of accuracy and macro F1-score across models.

    Args:
        models: Either a list of trained models or a dictionary mapping names to models.
        X: Input samples used for evaluation.
        y: Ground-truth labels.
        model_names (list[str], optional): Names used when `models` is a list.
            Defaults to None.
        device (optional): Device used for PyTorch-based models. Defaults to None.
        split_name (str, optional): Name displayed in the plot title.
            Defaults to `"Test"`.
        sort_by (str, optional): Column used to sort the comparison table.
            Must be `"accuracy"` or `"f1_score"`. Defaults to `"f1_score"`.
    """
    if sort_by not in {"accuracy", "f1_score"}:
        raise ValueError("sort_by debe ser 'accuracy' o 'f1_score'")

    y = _normalize_targets(y)

    if isinstance(models, dict):
        model_items = list(models.items())
    else:
        if model_names is None:
            model_names = [f"Model_{i}" for i in range(len(models))]
        if len(model_names) != len(models):
            raise ValueError("model_names debe tener el mismo largo que models")
        model_items = list(zip(model_names, models))

    rows = []

    for model_name, model in model_items:
        y_pred = get_predictions(model, X, device)
        accuracy = compute_accuracy(y, y_pred)
        f1_score, _ = compute_macro_f1_ova(y, y_pred)

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "f1_score": f1_score,
            }
        )

    results_df = (
        pd.DataFrame(rows)
        .sort_values(by=[sort_by, "accuracy"], ascending=False)
        .reset_index(drop=True)
    )

    x = np.arange(len(results_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(9, len(results_df) * 1.4), 6))

    accuracy_bars = ax.bar(
        x - width / 2,
        results_df["accuracy"],
        width=width,
        label="Accuracy",
        color="#1f77b4",
        edgecolor="white",
        linewidth=1.2,
    )
    f1_bars = ax.bar(
        x + width / 2,
        results_df["f1_score"],
        width=width,
        label="Macro F1-Score",
        color="#ff7f0e",
        edgecolor="white",
        linewidth=1.2,
    )

    for bars in (accuracy_bars, f1_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(
        f"{split_name} Metrics Comparison",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"], ha="right")
    ax.set_ylim(
        0, min(1.08, results_df[["accuracy", "f1_score"]].to_numpy().max() + 0.12)
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=11)

    best_model = results_df.iloc[0]
    ax.text(
        1.,
        1.,
        (f"Best by {sort_by}: {best_model['model']} " f"({best_model[sort_by]:.4f})"),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#444444",
    )

    plt.tight_layout()
    plt.show()


def compare_models(models: list, model_names: list[str] = None):
    """
    Build and print a comparison table for trained models.

    Args:
        models (list): Trained models with stored loss histories.
        model_names (list[str], optional): Names used to identify each model.
            Defaults to None.

    Returns:
        pd.DataFrame: Comparison table sorted by best validation loss.
    """

    rows = []

    for i, model in enumerate(models):

        train_loss = np.array(model.train_loss)
        val_loss = np.array(model.val_loss)

        best_epoch = np.argmin(val_loss)

        rows.append(
            {
                "Model": (model_names[i] if model_names is not None else f"Model_{i}"),
                "Best Val Loss": round(val_loss[best_epoch], 6),
                "Best Train Loss": round(train_loss[best_epoch], 6),
                "Best Epoch": best_epoch + 1,
                "Final Train Loss": round(train_loss[-1], 6),
                "Final Val Loss": round(val_loss[-1], 6),
                "Epochs Trained": len(train_loss),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(by="Best Val Loss", ascending=True)
        .reset_index(drop=True)
    )

    return df
