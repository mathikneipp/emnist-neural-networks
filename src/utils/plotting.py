import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from ..evaluation.metrics import (
    compute_accuracy,
    compute_macro_f1_ova,
    compute_multiclass_confusion_matrix,
)
from ..evaluation.predictions import get_predictions
from ..models.custom.neural_network import SecuentialNeuralNetwork
from ..models.torch.mlp import MLP

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


def _advance_fig_num(fig_num):
    """
    Advance the figure number in place when numbering is enabled.

    Args:
        fig_num: Mutable container with the current figure number, or None.
    """
    if fig_num is None:
        return
    if not isinstance(fig_num, list) or len(fig_num) != 1:
        raise ValueError(
            "fig_num debe ser None o una lista con un único entero, por ejemplo [1]"
        )
    fig_num[0] += 1


def _current_fig_num(fig_num):
    """
    Return the current figure number from a mutable container.

    Args:
        fig_num: Mutable container with the current figure number, or None.

    Returns:
        int | None: Current figure number if enabled.
    """
    if fig_num is None:
        return None
    if not isinstance(fig_num, list) or len(fig_num) != 1:
        raise ValueError(
            "fig_num debe ser None o una lista con un único entero, por ejemplo [1]"
        )
    return fig_num[0]


def _annotate_figure_number(fig, fig_num) -> None:
    """
    Draw the figure number inside the rendered plot.

    Args:
        fig: Matplotlib figure.
        fig_num: Current figure number or None.
    """
    if fig_num is None:
        return
    fig.text(
        0.015,
        0.985,
        f"Figura {fig_num}",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#444444",
    )


def _get_model_probabilities(model, X, device) -> np.ndarray:
    """
    Generate class probabilities for either a custom or PyTorch model.

    Args:
        model: Trained model instance.
        X: Input samples.
        device: Device used for PyTorch-based models.

    Returns:
        np.ndarray: Predicted class probabilities.
    """
    X = _to_numpy(X)

    if isinstance(model, SecuentialNeuralNetwork):
        return model.predict(X)

    if isinstance(model, MLP):
        X_tensor = torch.from_numpy(X).float().to(device)
        model.eval()

        with torch.no_grad():
            logits = model(X_tensor)
            return F.softmax(logits, dim=1).cpu().numpy()

    raise TypeError("Tipo de modelo no soportado")


def _compute_cross_entropy(model, X, y, device) -> float:
    """
    Compute multiclass cross-entropy for either a custom or PyTorch model.

    Args:
        model: Trained model instance.
        X: Input samples.
        y: Ground-truth labels.
        device: Device used for PyTorch-based models.

    Returns:
        float: Mean cross-entropy loss.
    """
    y = _normalize_targets(y)

    if isinstance(model, SecuentialNeuralNetwork):
        y_prob = _get_model_probabilities(model, X, device)
        return float(model.loss_function.fn(y, y_prob))

    if isinstance(model, MLP):
        X_tensor = torch.from_numpy(_to_numpy(X)).float().to(device)
        y_tensor = torch.from_numpy(y).long().to(device)
        model.eval()

        with torch.no_grad():
            logits = model(X_tensor)
            loss = F.cross_entropy(logits, y_tensor)
            return float(loss.item())

    raise TypeError("Tipo de modelo no soportado")


def _compute_split_metrics(model, X, y, device) -> dict[str, float]:
    """
    Compute accuracy, macro F1 and cross-entropy for one dataset split.

    Args:
        model: Trained model instance.
        X: Input samples.
        y: Ground-truth labels.
        device: Device used for PyTorch-based models.

    Returns:
        dict[str, float]: Metrics for the provided split.
    """
    y_true = _normalize_targets(y)
    y_pred = get_predictions(model, _to_numpy(X), device)
    macro_f1, _ = compute_macro_f1_ova(y_true, y_pred)

    return {
        "acc": compute_accuracy(y_true, y_pred),
        "f1": macro_f1,
        "cross_entropy": _compute_cross_entropy(model, X, y_true, device),
    }


def compare_models_table(
    model1, model2, X_train, y_train, X_val, y_val, device, fig_num=None
) -> pd.DataFrame:
    """
    Build a comparative metrics table for two classification models.

    Args:
        model1: First trained model.
        model2: Second trained model.
        X_train: Training inputs.
        y_train: Training labels.
        X_val: Validation inputs.
        y_val: Validation labels.
        device: Device used for PyTorch-based models.
        fig_num: Lista mutable con el número de figura actual, por ejemplo `[1]`.

    Returns:
        pd.DataFrame: Comparative table with train and validation metrics.
    """
    metrics_model1 = {
        "train": _compute_split_metrics(model1, X_train, y_train, device),
        "val": _compute_split_metrics(model1, X_val, y_val, device),
    }
    metrics_model2 = {
        "train": _compute_split_metrics(model2, X_train, y_train, device),
        "val": _compute_split_metrics(model2, X_val, y_val, device),
    }

    comparison_df = pd.DataFrame(
        [
            {
                "model": "model1",
                "train_acc": metrics_model1["train"]["acc"],
                "train_f1": metrics_model1["train"]["f1"],
                "train_cross_entropy": metrics_model1["train"]["cross_entropy"],
                "val_acc": metrics_model1["val"]["acc"],
                "val_f1": metrics_model1["val"]["f1"],
                "val_cross_entropy": metrics_model1["val"]["cross_entropy"],
            },
            {
                "model": "model2",
                "train_acc": metrics_model2["train"]["acc"],
                "train_f1": metrics_model2["train"]["f1"],
                "train_cross_entropy": metrics_model2["train"]["cross_entropy"],
                "val_acc": metrics_model2["val"]["acc"],
                "val_f1": metrics_model2["val"]["f1"],
                "val_cross_entropy": metrics_model2["val"]["cross_entropy"],
            },
        ]
    ).set_index("model")

    return comparison_df


def plot_random_images(
    images, fig_num=None, seed=42, n=4, figsize=(8, 8), cmap=None, image_shape=None
):
    """
    Display a random subset of images in a grid.

    Args:
        images: Collection of images to display.
        fig_num: Lista mutable con el número de figura actual, por ejemplo `[1]`.
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

    figure_num = _current_fig_num(fig_num)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, num=figure_num, clear=True)
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

    _annotate_figure_number(fig, figure_num)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    _advance_fig_num(fig_num)


def plot_training_history(
    train_loss: list[float], val_loss: list[float], fig_num=None
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        train_loss (list[float]): Training loss history.
        val_loss (list[float]): Validation loss history.
        fig_num: Lista mutable con el número de figura actual, por ejemplo `[1]`.
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

    figure_num = _current_fig_num(fig_num)
    fig = plt.figure(figsize=(11, 6), num=figure_num, clear=True)

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
    _annotate_figure_number(fig, figure_num)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    _advance_fig_num(fig_num)


def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    dataset_name: str = "emnist_bymerge",
    val_name: str = "Validation",
    device=None,
    fig_num=None,
) -> None:
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
        fig_num: Lista mutable con el número de figura actual, por ejemplo `[1]`.
    """

    def plot_bar_metric(ax, train_value, val_value, title, ylabel, ylim=None):
        """
        Plot a bar chart comparing training and validation metrics on a given axis.

        Args:
            ax: Matplotlib axis where the plot is drawn.
            train_value: Metric value for the training split.
            val_value: Metric value for the validation split.
            title: Plot title.
            ylabel: Label for the y-axis.
            ylim: Optional y-axis limits.
        """
        values = [train_value, val_value]
        labels = ["Train", val_name]

        bars = ax.bar(labels, values, color=["#1f77b4", "#d62728"], width=0.6)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12)

        ax.grid(axis="y", linestyle="--", alpha=0.35)
        y_min, y_max = ax.get_ylim()
        label_offset = (y_max - y_min) * 0.02

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offset,
                f"{value:.4f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

    def plot_confusion_matrix(ax, cm, classes, title):
        """
        Plot a single multiclass confusion matrix on a given axis.

        Args:
            ax: Matplotlib axis where the plot is drawn.
            cm: Confusion matrix values.
            classes: Class labels represented in the matrix.
            title: Plot title.
        """
        ticks = None
        if dataset_name == "emnist_bymerge":
            ticks = EMNIST_CLASSES[: len(classes)]

        image = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)

        if ticks:
            tick_positions = np.arange(len(classes))
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(ticks)
            ax.set_yticklabels(ticks)
        else:
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)

        return image

    y_train = _normalize_targets(y_train)
    y_val = _normalize_targets(y_val)

    metric_classes = None
    if dataset_name == "emnist_bymerge":
        metric_classes = np.arange(len(EMNIST_CLASSES))

    # Generate predictions
    y_train_pred = get_predictions(model, X_train, device)
    y_val_pred = get_predictions(model, X_val, device)

    # Compute overall accuracy
    train_accuracy = compute_accuracy(y_train, y_train_pred)
    val_accuracy = compute_accuracy(y_val, y_val_pred)

    train_loss_history = None
    val_loss_history = None

    if hasattr(model, "train_loss") and hasattr(model, "val_loss"):
        train_loss_history = np.asarray(model.train_loss, dtype=float).reshape(-1)
        val_loss_history = np.asarray(model.val_loss, dtype=float).reshape(-1)

    if (
        train_loss_history is not None
        and val_loss_history is not None
        and train_loss_history.size > 0
        and val_loss_history.size > 0
    ):
        train_cross_entropy = float(np.min(train_loss_history))
        val_cross_entropy = float(np.min(val_loss_history))
    else:
        # Fallback for models re-trained without stored validation history.
        train_cross_entropy = _compute_cross_entropy(model, X_train, y_train, device)
        val_cross_entropy = _compute_cross_entropy(model, X_val, y_val, device)
    # Compute one-vs-all macro F1
    train_f1, _ = compute_macro_f1_ova(y_train, y_train_pred, classes=metric_classes)
    val_f1, _ = compute_macro_f1_ova(y_val, y_val_pred, classes=metric_classes)

    # Compute multiclass confusion matrices
    train_cm, train_classes = compute_multiclass_confusion_matrix(y_train, y_train_pred)
    val_cm, val_classes = compute_multiclass_confusion_matrix(y_val, y_val_pred)

    cross_entropy_max = max(train_cross_entropy, val_cross_entropy)
    cross_entropy_ylim = (0, cross_entropy_max * 1.08 if cross_entropy_max > 0 else 0.1)

    # Plot all metrics with 3 charts on top and 2 full-width confusion matrices below
    figure_num = _current_fig_num(fig_num)
    fig = plt.figure(figsize=(22, 18), num=figure_num, clear=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.8])
    metric_axes = [fig.add_subplot(grid[0, i * 2 : (i + 1) * 2]) for i in range(3)]
    train_cm_ax = fig.add_subplot(grid[1, :3])
    val_cm_ax = fig.add_subplot(grid[1, 3:])

    plot_bar_metric(
        metric_axes[0],
        train_accuracy,
        val_accuracy,
        "Overall Accuracy",
        "Accuracy",
        ylim=(0, 1),
    )
    plot_bar_metric(
        metric_axes[1],
        train_cross_entropy,
        val_cross_entropy,
        f"Cross-Entropy",
        "Cross-Entropy",
        ylim=cross_entropy_ylim,
    )
    plot_bar_metric(
        metric_axes[2],
        train_f1,
        val_f1,
        "Macro F1-Score",
        "F1-Score",
        ylim=(0, 1),
    )

    train_image = plot_confusion_matrix(
        train_cm_ax, train_cm, train_classes, "Train Confusion Matrix"
    )
    val_image = plot_confusion_matrix(
        val_cm_ax, val_cm, val_classes, f"{val_name} Confusion Matrix"
    )

    fig.colorbar(train_image, ax=train_cm_ax, fraction=0.046, pad=0.04)
    fig.colorbar(val_image, ax=val_cm_ax, fraction=0.046, pad=0.04)

    _annotate_figure_number(fig, figure_num)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    _advance_fig_num(fig_num)


def plot_model_metric_comparison(
    models,
    X,
    y,
    model_names: list[str] = None,
    device=None,
    split_name: str = "Test",
    sort_by: str = "f1_score",
    fig_num=None,
) -> None:
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
            Must be `"accuracy"`, `"f1_score"` or `"cross_entropy"`.
            Defaults to `"f1_score"`.
        fig_num: Lista mutable con el número de figura actual, por ejemplo `[1]`.
    """
    if sort_by not in {"accuracy", "f1_score", "cross_entropy"}:
        raise ValueError("sort_by debe ser 'accuracy', 'f1_score' o 'cross_entropy'")

    y = _normalize_targets(y)
    metric_classes = None

    if split_name.lower() == "test":
        metric_classes = np.arange(len(EMNIST_CLASSES))

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
        f1_score, _ = compute_macro_f1_ova(y, y_pred, classes=metric_classes)
        val_loss = None

        if hasattr(model, "val_loss"):
            val_loss = np.asarray(model.val_loss, dtype=float).reshape(-1)

        if val_loss is not None and val_loss.size > 0:
            cross_entropy = float(np.min(val_loss))
        else:
            cross_entropy = _compute_cross_entropy(model, X, y, device)

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "f1_score": f1_score,
                "cross_entropy": cross_entropy,
            }
        )

    ascending = sort_by == "cross_entropy"
    results_df = (
        pd.DataFrame(rows)
        .sort_values(
            by=[sort_by, "accuracy"],
            ascending=[ascending, False],
        )
        .reset_index(drop=True)
    )

    group_spacing = 1.
    x = np.arange(len(results_df), dtype=float) * group_spacing
    width = 0.26
    metric_offset = 0.28
    label_x_offsets = {
        "accuracy": -0.02,
        "f1_score": 0.0,
        "cross_entropy": 0.02,
    }

    figure_num = _current_fig_num(fig_num)
    fig, ax = plt.subplots(
        figsize=(max(10, len(results_df) * 1.75), 6.4), num=figure_num, clear=True
    )

    accuracy_bars = ax.bar(
        x - metric_offset,
        results_df["accuracy"],
        width=width,
        label="Accuracy",
        color="#1f77b4",
        edgecolor="white",
        linewidth=1.2,
    )
    f1_bars = ax.bar(
        x,
        results_df["f1_score"],
        width=width,
        label="Macro F1-Score",
        color="#ff7f0e",
        edgecolor="white",
        linewidth=1.2,
    )
    cross_entropy_bars = ax.bar(
        x + metric_offset,
        results_df["cross_entropy"],
        width=width,
        label="Best Cross-Entropy",
        color="#2ca02c",
        edgecolor="white",
        linewidth=1.2,
    )

    label_offsets = {
        "accuracy": 0.018,
        "f1_score": 0.045,
        "cross_entropy": 0.018,
    }

    for bars, metric_name in (
        (accuracy_bars, "accuracy"),
        (f1_bars, "f1_score"),
        (cross_entropy_bars, "cross_entropy"),
    ):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2 + label_x_offsets[metric_name],
                height + label_offsets[metric_name],
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_title(
        f"{split_name} Metrics Comparison",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Score / Loss", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"], ha="center")
    ax.margins(x=0.08)
    ax.set_ylim(
        0,
        results_df[["accuracy", "f1_score", "cross_entropy"]].to_numpy().max() + 0.12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=11)

    best_model = results_df.iloc[0]
    ax.text(
        1.0,
        1.0,
        (f"Best by {sort_by}: {best_model['model']} " f"({best_model[sort_by]:.4f})"),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#444444",
    )

    _annotate_figure_number(fig, figure_num)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    _advance_fig_num(fig_num)


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
