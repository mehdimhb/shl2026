import argparse
import os
import pickle
import lmdb
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from src import data
from mantis.architecture import MantisV2

from src.model import get_loaded_nutime_channel_encoder

def extract_features(batch_frames: torch.Tensor, unishape_model, mantis_model, nutime_model, device: torch.device):
    x_raw = batch_frames.to(device)
    if not x_raw.is_floating_point():
        x_raw = x_raw.float()

    # Unishape and Mantis require interpolation to 512
    x_interpolated = F.interpolate(x_raw, size=512, mode='linear', align_corners=False)

    B, C, L = x_interpolated.shape
    x_univariate = x_interpolated.view(B * C, 1, L)

    uni_features, mantis_features, nutime_features = None, None, None

    with torch.no_grad():
        # --- Unishape ---
        if unishape_model is not None:
            uni = unishape_model(x_univariate)
            uni_features = uni.reshape(B, C, -1)

        # --- Mantis ---
        if mantis_model is not None:
            mantis = mantis_model(x_univariate)
            mantis_features = mantis.reshape(B, C, -1)

        # --- NuTime ---
        if nutime_model is not None:
            nutime_input = x_raw
            # NuTime expects (B, 9, 500). Fix permutation if it's (B, 500, 9)
            if nutime_input.shape[1] == 500 and nutime_input.shape[2] == 9:
                nutime_input = nutime_input.permute(0, 2, 1)

            # Extract frozen features -> expected output: (B, 9, 128)
            nutime_features = nutime_model(nutime_input)

    return uni_features, mantis_features, nutime_features


def save_split_to_lmdb(loader, unishape_model, mantis_model, nutime_model, precomputed, txn, split_name, device):
    print(f"Processing {split_name} split...")
    keys = []
    global_idx = 0

    for batch in tqdm(loader, desc=f"Extracting {split_name}"):
        if isinstance(batch, dict):
            frames, labels = batch['frames'], batch['labels']
        else:
            frames, labels = batch[0], batch[1]

        # Added nutime_model to the extraction call
        uni_features, mantis_features, nutime_features = extract_features(
            frames, unishape_model, mantis_model, nutime_model, device
        )

        uni_features_np = uni_features.cpu().numpy() if uni_features is not None else None
        mantis_features_np = mantis_features.cpu().numpy() if mantis_features is not None else None
        nutime_features_np = nutime_features.cpu().numpy() if nutime_features is not None else None
        labels_np = labels.cpu().numpy()

        batch_size = labels_np.shape[0]

        # Precomputed TiRex / TSPulse features, aligned with the raw LMDB order
        precomputed_np = {
            name: feats[global_idx : global_idx + batch_size].cpu().numpy()
            for name, feats in precomputed.items()
        }

        for i in range(batch_size):
            key_str = f"{split_name}_{global_idx}"
            key = key_str.encode('ascii')

            data_dict = {"label": labels_np[i]}
            if uni_features_np is not None: data_dict["unishape_features"] = uni_features_np[i]
            if mantis_features_np is not None: data_dict["mantis_features"] = mantis_features_np[i]
            if nutime_features_np is not None: data_dict["nutime_features"] = nutime_features_np[i]
            for name, feats in precomputed_np.items():
                data_dict[f"{name}_features"] = feats[i]

            txn.put(key, pickle.dumps(data_dict))
            keys.append(key_str)
            global_idx += 1

    txn.put(f"{split_name}_keys".encode('ascii'), pickle.dumps(keys))
    print(f"Finished {split_name}. Saved {len(keys)} samples.\n")


def load_precomputed(files_per_split: dict[str, list[str]], key: str) -> dict:
    """Load per-split feature tensors, concatenating multi-part files. Missing splits map to None."""
    out = {}
    for split, paths in files_per_split.items():
        tensors = [torch.load(p, map_location="cpu")[key] for p in paths if os.path.exists(p)]
        out[split] = torch.cat(tensors, dim=0) if tensors else None
    return out


def main(models_to_use=("unishape", "mantis", "nutime", "trex", "tspulse")):
    models_to_use = set(models_to_use)
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}\nUsing models: {models_to_use}")

    # -----------------------------
    # 1. Load Raw Data Separately
    # -----------------------------
    print("Initializing DataLoaders...")
    ds_train = data._HARDataset(mode='train', lmdb_path='dataset/lmdb')
    ds_val   = data._HARDataset(mode='val', lmdb_path='dataset/lmdb')
    ds_test  = data._HARDataset(mode='test', lmdb_path='dataset/lmdb')

    loaders = {
        "train": DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=4),
        "val": DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=4),
        "test": DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=4)
    }

    # -----------------------------
    # 2. Load Models Conditionally
    # -----------------------------
    uni, mantis, nutime_model = None, None, None

    if "unishape" in models_to_use:
        try:
            from src.model.uni_shape import get_loaded_uni_shape
        except ImportError as e:
            raise RuntimeError(
                "UniShape code is not distributed with this repository. "
                "See 'UniShape' in README.md, or pass models without 'unishape'."
            ) from e
        uni = get_loaded_uni_shape('weights/unishape_checkpoint_zeroshot.pth', return_all_cls_tokens=False).to(device).eval()
    if "mantis" in models_to_use:
        mantis = MantisV2(device=str(device)).from_pretrained("paris-noah/MantisV2").to(device).eval()
    if "nutime" in models_to_use:
        # Load the frozen NuTime Channel Encoder
        nutime_model = get_loaded_nutime_channel_encoder(
            weights_path='weights/checkpoint_bias9.pth',
            device=device
        )

    # -----------------------------
    # 3. Load precomputed TiRex / TSPulse features
    # -----------------------------
    # These two models are not run by this repository. Their features must be
    # computed elsewhere and saved in the formats described in README.md.
    precomputed = {}
    if "trex" in models_to_use:
        print("Loading precomputed TiRex features...")
        precomputed["trex"] = load_precomputed(
            {
                "train": [f"dataset/TRex_train_part{i}.pt" for i in range(1, 5)],
                "val": ["dataset/TRex_val.pt"],
                "test": ["dataset/TRex_test.pt"],
            },
            key="time_channels_512",
        )
    if "tspulse" in models_to_use:
        print("Loading precomputed TSPulse features...")
        precomputed["tspulse"] = load_precomputed(
            {split: [f"dataset/TSPulse_{split}.pt"] for split in ["train", "val", "test"]},
            key="features",
        )

    # -----------------------------
    # 4. Feature Extraction to LMDB
    # -----------------------------
    target_lmdb_path = 'dataset/lmdb_features_all'
    os.makedirs(target_lmdb_path, exist_ok=True)
    env = lmdb.open(target_lmdb_path, map_size=1099511627776)

    with env.begin(write=True) as txn:
        for split_name in ["train", "val", "test"]:
            save_split_to_lmdb(
                loader=loaders[split_name],
                unishape_model=uni,
                mantis_model=mantis,
                nutime_model=nutime_model,
                precomputed={k: v[split_name] for k, v in precomputed.items() if v[split_name] is not None},
                txn=txn,
                split_name=split_name,
                device=device
            )

    env.close()
    print(f"Feature extraction complete! LMDB saved to {target_lmdb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract foundation-model features into an LMDB.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["unishape", "mantis", "nutime", "trex", "tspulse"],
        choices=["unishape", "mantis", "nutime", "trex", "tspulse"],
    )
    main(parser.parse_args().models)
