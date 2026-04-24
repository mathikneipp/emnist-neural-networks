import numpy as np
import matplotlib.pyplot as plt


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
