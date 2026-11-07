import argparse

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src import data, model, task

parser = argparse.ArgumentParser(description="Plot cross-attention heatmaps for the ef-unishape-mantis-nutime-tirex ensemble.")
parser.add_argument("--checkpoint", required=True, help="Path to a Lightning .ckpt file")
parser.add_argument("--output", default="cross_attention.pdf")
args = parser.parse_args()

# ==========================================
# 1. Setup & Data Loading
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

active_seq_models = ['unishape', 'mantis', 'nutime', 'trex']

# Initialize Dataset (excluding tspulse, including trex)
dataset = data.HARFeatureData(
    lmdb_path='dataset/lmdb_features_all',
    models=['unishape', 'mantis', 'nutime', 'trex'],
    task='all',
)

data_module = task.DataModule(
    data=dataset,
    batch_size=1024,
    num_workers=0,
    shuffle=False
)
data_module.setup(stage="test")
target_loader = data_module.val_dataloader()

learned_classifier = model.Ensemble.load(
    'configs/task/ef-unishape-mantis-nutime-tirex/model.json',
    args.checkpoint,
)
learned_classifier.to(device)
learned_classifier.eval()

# ==========================================
# 3. Hook Setup for Feature Extraction
# ==========================================
activations = {'cross_attn': {}}

def get_mha_hook(mod_name):
    def hook(module, input, output):
        # output[1] shape: (Batch, Heads, Target_Seq, Source_Seq)
        attn_weights = output[1].detach().cpu()

        # Average over the batch dimension ONLY
        # Resulting shape: (Heads, Target, Source)
        attn_weights = attn_weights.mean(dim=0)
        activations['cross_attn'][mod_name] = attn_weights.numpy()
    return hook

handles = []
for mod in active_seq_models:
    handles.append(
        learned_classifier.cross_attns[mod].cross_attn.register_forward_hook(get_mha_hook(mod))
    )

# ==========================================
# 4. Run Inference
# ==========================================
batch = next(iter(target_loader))
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

with torch.no_grad():
    _ = learned_classifier(batch)

for h in handles:
    h.remove()

# ==========================================
# 5. Process Data & Build 4D Mega Matrix
# ==========================================
num_channels = 9
num_models = len(active_seq_models)
total_dim = num_models * num_channels
num_heads = 4

# Shape: (4 Heads, 36 Target, 36 Source). NaNs keep self-attention empty.
mega_attn = np.full((num_heads, total_dim, total_dim), np.nan)

for row_idx, mod in enumerate(active_seq_models):
    attn_data = activations['cross_attn'][mod] # Shape: (4, 9, 27)

    others = [m for m in active_seq_models if m != mod]

    row_start = row_idx * num_channels
    row_end = row_start + num_channels

    for col_internal_idx, other_mod in enumerate(others):
        col_idx = active_seq_models.index(other_mod)
        col_start = col_idx * num_channels
        col_end = col_start + num_channels

        # Extract block for all heads simultaneously
        block = attn_data[:, :, col_internal_idx * num_channels : (col_internal_idx + 1) * num_channels]
        mega_attn[:, row_start:row_end, col_start:col_end] = block

# ==========================================
# 6. Plotting the 2x2 Grid
# ==========================================
mod_prefixes = {'unishape': 'U', 'mantis': 'M', 'nutime': 'N', 'trex': 'T'}
# labels = [f"{mod_prefixes[mod]}{i}" for mod in active_seq_models for i in range(1, num_channels + 1)]
labels = [f"{i}" for mod in active_seq_models for i in range(1, num_channels + 1)]

# Create 2x2 grid and a dedicated axis for a single shared colorbar
fig, axes = plt.subplots(2, 2, figsize=(20, 18), sharex=True, sharey=False)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]

for i in range(num_heads):
    row = i // 2
    col = i % 2
    ax = axes[row, col]

    sns.heatmap(
        mega_attn[i],
        cmap="magma",
        ax=ax,
        cbar=(i == 0), # Only draw colorbar once
        cbar_ax=cbar_ax if i == 0 else None,
        xticklabels=labels if row == 1 else False,
        yticklabels=labels if col == 0 else False,
        vmin=0.0,
        rasterized=True
    )

    # Add embedded label instead of a title
    # ax.text(0.02, 0.98, f"Head {i+1}", transform=ax.transAxes,
    #         fontsize=16, fontweight='bold', color='white',
    #         verticalalignment='top',
    #         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # Visual dividers
    for j in range(1, num_models):
        ax.axhline(j * num_channels, color='white', linewidth=2)
        ax.axvline(j * num_channels, color='white', linewidth=2)

    # Enforce strict axis label placement
    # if col == 0:
    #     ax.set_ylabel("Target Tokens (Queries)", fontsize=16, fontweight='bold')
    # else:
    ax.set_ylabel("")

    # if row == 1:
    #     # ax.set_xlabel("Source Tokens (Keys / Values)", fontsize=16, fontweight='bold')
    #     # ax.tick_params(axis='x', rotation=90, labelsize=10)
    #     plt.setp(ax.get_xticklabels(), rotation=90)
    # else:
    #     ax.tick_params(axis='x', which='both', bottom=False)
    # ax.set_xlabel("")

    centers = [i * num_channels + (num_channels / 2) for i in range(num_models)]
    model_display_names = ['UniShape', 'Mantis', 'NuTime', 'TiRex']


    if col == 0:
        # ax.set_ylabel("Target Tokens (Queries)", fontsize=16, fontweight='bold', labelpad=60)
        # Add model names centered on the Y-axis
        plt.setp(ax.get_yticklabels(), rotation=0)
        for center, name in zip(centers, model_display_names):
            ax.text(-0.05, center, name, transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=16, fontweight='bold', rotation=90)

    if row == 1:
        # ax.set_xlabel("Source Tokens (Keys / Values)", fontsize=16, fontweight='bold', labelpad=70)
        # plt.setp(ax.get_xticklabels(), rotation=90)
        # Add model names centered on the X-axis
        for center, name in zip(centers, model_display_names):
            ax.text(center, -0.05, name, transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=16, fontweight='bold')

    # if col == 0:
    #     ax.tick_params(axis='y', labelsize=10)
    # else:
    #     ax.tick_params(axis='y', which='both', left=False)

# cbar_ax.set_ylabel('Attention Weight', fontsize=14, fontweight='bold')
plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.9)

plt.savefig(args.output, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved as {args.output}")