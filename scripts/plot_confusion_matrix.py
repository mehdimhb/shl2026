import argparse
import importlib
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from src import data, task


def plot_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, save_path: str = "confusion_matrix.png", title=''):
    """
    Calculates and plots a confusion matrix using torchmetrics.

    Args:
        preds: Tensor of model predictions (either logits or raw class indices).
        targets: Tensor of ground truth labels.
        save_path: File path to save the generated image.
    """
    class_names = ["Still", "Walking", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
    num_classes = len(class_names)

    # Initialize the metric
    # Note: normalize='true' converts raw counts to percentages based on the true labels (rows).
    # Change to normalize='none' if you want to see the raw sample counts.
    confmat = MulticlassConfusionMatrix(num_classes=num_classes, normalize='true')
    acc_cal = MulticlassAccuracy(num_classes=num_classes, average='micro')
    f1_cal = MulticlassF1Score(num_classes=num_classes, average='macro')

    # Calculate the matrix tensor
    matrix = confmat(preds, targets)
    acc = acc_cal(preds, targets)
    f1 = f1_cal(preds, targets)

    # --- Visualizing the Matrix ---
    plt.figure(figsize=(10, 8))

    # Using seaborn to create a heatmap from the tensor
    sns.heatmap(
        matrix.cpu().numpy(),
        annot=True,
        fmt=".2f",      # Use ".0f" if you switch to normalize='none'
        cmap="Blues",   # A clean, standard color map
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )

    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.title(title + f' - Acc={acc*100:.4f} - F1={f1*100:.4f}', fontsize=14, pad=15)

    # Rotate tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix successfully saved to '{save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a confusion matrix for a trained head or ensemble.")
    parser.add_argument("--config", default="ef-unishape-mantis-nutime-tirex", help="Folder name under configs/task")
    parser.add_argument("--checkpoint", required=True, help="Path to a Lightning .ckpt file")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--lmdb-path", default="dataset/lmdb_features_all")
    parser.add_argument("--output", default="confusion_matrix.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_dir = Path("configs/task") / args.config
    model_config = config_dir / "model.json"
    modalities = json.loads((config_dir / "data.json").read_text())["parameters"]["models"]
    model_spec = json.loads(model_config.read_text())
    model_cls = getattr(importlib.import_module(model_spec["module"]), model_spec["class"])

    dataset = data.HARFeatureData(
        lmdb_path=args.lmdb_path,
        models=modalities,
        task='all',
    )

    data_module = task.DataModule(
        data=dataset,
        batch_size=128,
        num_workers=0,
        shuffle=False
    )

    data_module.setup(stage="train")
    target_loader = data_module.train_dataloader() if args.split == "train" else data_module.val_dataloader()

    learned_classifier = model_cls.load(model_config, args.checkpoint)
    learned_classifier.to(device)

    all_preds = []
    all_labels = []

    learned_classifier.eval()
    with torch.no_grad():
        for batch in tqdm(target_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            output = learned_classifier(batch)

            output["labels"] = output["labels"].argmax(dim=1)

            all_preds.append(output['logits'].cpu())
            all_labels.append(output['labels'].cpu())

    plot_confusion_matrix(
        torch.cat(all_preds, dim=0),
        torch.cat(all_labels, dim=0),
        save_path=args.output,
        title=f'{args.config} - {args.split}'
    )
