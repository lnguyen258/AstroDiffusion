from typing import List, Dict, Optional

import matplotlib.pyplot as plt


def plot_history(history: Dict[str, List[float]], save_fig: Optional[str] = None) -> None:
    plt.figure(figsize=(12, 5))
    epochs = history['epoch']

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['val_loss'], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot training and validation metric (accuracy, for example)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_metric'], label="Train Accuracy")
    plt.plot(epochs, history['val_metric'], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=300)
    else:
        plt.show()