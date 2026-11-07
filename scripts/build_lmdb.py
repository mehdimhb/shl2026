import os
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt


def bandpass_filter_1d(x, fs=100, low=0.5, high=20.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def filter_sample_keep_100hz(frame, fs=100, low=0.5, high=20.0, order=4):
    """
    frame: np.ndarray shape (C, T)
    returns: np.ndarray shape (C, T) (same length; no resampling)
    """
    c, _ = frame.shape
    out = np.empty_like(frame, dtype=np.float32)
    for ch in range(c):
        out[ch] = bandpass_filter_1d(frame[ch], fs=fs, low=low, high=high, order=order)
    return out.astype(np.float32)


class LMDBBuilder:
    """
    Build an LMDB from raw sensor text files with:
      - NaN DROP only (skip frames containing any NaN)
      - optional bandpass filtering (keeps 100 Hz; no resampling)
      - normalization by global per-channel maxima (computed in a first pass)

    Each LMDB entry:
      { "frame": np.ndarray(C,T) float32, "label": int, "device": str }

    __meta__ contains:
      - index_map, devices, sensors, axes, mode
      - norm: { "type": "global_channel_max", "channel_max": np.ndarray(C,1) }
    """

    def __init__(
        self,
        root_dir: str,
        lmdb_path: str,
        map_size: float = 1e12,
        fs: int = 100,
        preprocessing: bool = False,
        bp_low: float = 0.5,
        bp_high: float = 20.0,
        bp_order: int = 4,
        eps: float = 1e-8,
    ):
        self.root_dir = root_dir
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, map_size=int(map_size))

        self.fs = fs
        self.preprocessing = preprocessing
        self.bp_low = bp_low
        self.bp_high = bp_high
        self.bp_order = bp_order

        # for safe division if a channel is all zeros
        self.eps = eps

        self.index = 0
        self.index_encoder = lambda idx: f"{idx:09d}".encode()

        self.meta = {
            "index_map": [],
            "classes": {},
            "devices": {},
            "sensors": {},
            "axes": {},
            "mode": {},
            "norm": {}
        }

    def close(self):
        self.env.close()

    # ---------- file/metadata utilities ----------

    def _parse_filename(self, filename: str):
        """
        Parse filenames like 'Acc_Bag.txt' -> ('Acc', 'Bag')
        """
        name = filename[:-4] if filename.endswith(".txt") else filename
        parts = name.split("_")
        if len(parts) < 2:
            raise ValueError(f"Unexpected sensor filename format: {filename}")
        return parts[0], parts[1]

    def _collect_device_files(self, device_path: str):
        """
        Returns (data_paths, label_path) and updates sensors/axes metadata.
        """
        filenames = sorted([f for f in os.listdir(device_path) if f != "Label.txt" and f.endswith(".txt")])
        data_paths = [os.path.join(device_path, f) for f in filenames]
        label_path = os.path.join(device_path, "Label.txt")

        for f in filenames:
            sensor, axis = self._parse_filename(f)
            if sensor not in self.meta["sensors"]:
                self.meta["sensors"][sensor] = len(self.meta["sensors"])
            if axis not in self.meta["axes"]:
                self.meta["axes"][axis] = len(self.meta["axes"])

        return data_paths, label_path

    def _iter_frames_from_device(self, data_paths, label_path):
        """
        Streams frames (C,T) and labels from one device directory.
        Drops any frame containing NaNs.
        """
        data_files = [open(p, "r") for p in data_paths]
        label_file = open(label_path, "r") if os.path.exists(label_path) else None

        try:
            while True:
                lines = [f.readline() for f in data_files]
                if not lines[0]:
                    break
                # if any file ends early, stop to avoid misalignment
                if any(line == "" for line in lines):
                    break

                label_line = label_file.readline() if label_file is not None else None

                # ROBUST PARSING: Handles extra spaces, commas, and irregular whitespace
                try:
                    frame = np.stack(
                        [np.array(line.replace(",", " ").strip().split(), dtype=np.float32) for line in lines],
                        axis=0
                    )
                except ValueError:
                    # If a line contains truly unparsable data (e.g., text instead of numbers), drop the frame
                    continue

                if label_line is not None:
                    try:
                        arr = np.array(label_line.replace(",", " ").strip().split(), dtype=np.float32)
                        label = int(arr[0] - 1) if len(arr) else -1
                    except ValueError:
                        label = -1
                else:
                    label = -1

                # NaN DROP ONLY
                if np.isnan(frame).any():
                    continue

                if self.preprocessing:
                    frame = filter_sample_keep_100hz(
                        frame, fs=self.fs, low=self.bp_low, high=self.bp_high, order=self.bp_order
                    )

                yield frame.astype(np.float32), label
        finally:
            for f in data_files:
                f.close()
            if label_file is not None:
                label_file.close()

    def _iter_all_frames(self, folders):
        """
        Yields (folder, device, frame, label) for all requested physical folders.
        Handles both folders with device subdirectories and folders with direct .txt files.
        """
        for folder in folders:
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            # Check if text files sit directly in the current folder
            if any(f.endswith(".txt") for f in os.listdir(folder_path)):
                data_paths, label_path = self._collect_device_files(folder_path)
                for frame, label in self._iter_frames_from_device(data_paths, label_path):
                    yield folder, "Unknown", frame, label
            else:
                for device in sorted(os.listdir(folder_path)):
                    device_path = os.path.join(folder_path, device)
                    if not os.path.isdir(device_path):
                        continue

                    data_paths, label_path = self._collect_device_files(device_path)
                    for frame, label in self._iter_frames_from_device(data_paths, label_path):
                        yield folder, device, frame, label

    # ---------- normalization ----------

    def compute_global_channel_max(self, folders=("train", "validation", "test")):
        """
        First pass: compute global per-channel maxima across all frames and time.
        Returns channel_max shaped (C,1).
        """
        channel_max = None

        for _, _, frame, _ in tqdm(self._iter_all_frames(folders), desc="Scan for global channel max"):
            # frame: (C,T)
            c = frame.shape[0]
            this_max = np.max(np.abs(frame), axis=1, keepdims=True)  # (C,1)

            if channel_max is None:
                channel_max = np.full((c, 1), -np.inf, dtype=np.float32)

            if channel_max.shape[0] != c:
                raise ValueError(
                    f"Inconsistent channel count: expected {channel_max.shape[0]}, got {c}. "
                    "Check sensor files across devices."
                )

            channel_max = np.maximum(channel_max, this_max)

        if channel_max is None:
            raise RuntimeError("No frames found. Check root_dir/folders structure and file contents.")

        # avoid division by zero
        channel_max = np.maximum(channel_max, self.eps).astype(np.float32)
        return channel_max

    # ---------- LMDB writing ----------

    def _store_entry(self, txn, entry):
        key = self.index_encoder(self.index)
        txn.put(key, pickle.dumps(entry))
        self.meta["index_map"].append(key)
        self.index += 1

    def build(self, split_mapping=None, normalize=True):
        """
        Two-pass build if normalize=True:
          1) compute global channel max (over specified physical folders)
          2) write normalized frames into LMDB based on logical split mapping
        """
        if split_mapping is None:
            # Default fallback mapping
            split_mapping = {
                "train": ["train"],
                "validation": ["validation"],
                "test": ["test"]
            }

        # Flatten the dictionary to get all physical folders we need to scan
        all_folders = [folder for folders in split_mapping.values() for folder in folders]

        if normalize:
            channel_max = self.compute_global_channel_max(folders=all_folders)
            self.meta["norm"] = {
                "type": "global_channel_max",
                "channel_max": channel_max  # (C,1) float32
            }
        else:
            channel_max = None
            self.meta["norm"] = {"type": "none"}

        with self.env.begin(write=True) as txn:
            for logical_split, physical_folders in split_mapping.items():
                start_index = self.index

                for folder in physical_folders:
                    folder_path = os.path.join(self.root_dir, folder)
                    if not os.path.isdir(folder_path):
                        print(f"Warning: Directory '{folder_path}' not found. Skipping.")
                        continue

                    # Check if text files sit directly in the current folder
                    if any(f.endswith(".txt") for f in os.listdir(folder_path)):
                        device = "Unknown"

                        if device not in self.meta["devices"]:
                            self.meta["devices"][device] = len(self.meta["devices"])

                        data_paths, label_path = self._collect_device_files(folder_path)

                        for frame, label in tqdm(
                            self._iter_frames_from_device(data_paths, label_path),
                            desc=f"Write {logical_split} (from {folder})/{device}"
                        ):
                            if channel_max is not None:
                                frame = frame / channel_max

                            entry = {"frame": frame.astype(np.float32), "label": label, "device": device}
                            self._store_entry(txn, entry)
                    else:
                        for device in sorted(os.listdir(folder_path)):
                            device_path = os.path.join(folder_path, device)
                            if not os.path.isdir(device_path):
                                continue

                            # Add device to meta safely without overwriting ID
                            if device not in self.meta["devices"]:
                                self.meta["devices"][device] = len(self.meta["devices"])

                            data_paths, label_path = self._collect_device_files(device_path)

                            for frame, label in tqdm(
                                self._iter_frames_from_device(data_paths, label_path),
                                desc=f"Write {logical_split} (from {folder})/{device}"
                            ):
                                if channel_max is not None:
                                    frame = frame / channel_max  # broadcast (C,T)/(C,1)

                                entry = {"frame": frame.astype(np.float32), "label": label, "device": device}
                                self._store_entry(txn, entry)

                # Store the index range for this logical split
                self.meta["mode"][logical_split] = [
                    self.index_encoder(i) for i in range(start_index, self.index)
                ]

            txn.put(b"__meta__", pickle.dumps(self.meta))


if __name__ == "__main__":
    builder = LMDBBuilder(
        root_dir='dataset',
        lmdb_path='dataset/lmdb',
        preprocessing=True,
    )

    # Map each LMDB split to its source folder under dataset/. These split names
    # are the ones scripts/extract_features.py reads.
    custom_split_mapping = {
        "train": ["train"],
        "val": ["validation"],
        "test": ["test"],
    }

    builder.build(split_mapping=custom_split_mapping, normalize=True)
    builder.close()
