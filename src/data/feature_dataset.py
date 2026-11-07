import lmdb
import pickle
import torch
import torch.nn.functional as F
import csv

from .base import BaseDataset, define_dataset

class _HARFeatureDataset(BaseDataset):
    def __init__(
        self,
        mode: str,
        lmdb_path: str,
        models: list[str] = ['unishape', 'mantis', 'trex', 'tspulse', 'nutime', 'fft'],
        task: str = 'all',
        val75_csv: str = 'splits/val75_indices.csv',
        val25_csv: str = 'splits/val25_indices.csv',
    ):
        self.lmdb_path = lmdb_path
        self.mode = mode
        self.models = models
        self.task = task
        self.env = None
        self.txn = None

        # Kinematic Groupings: valid classes + 1 OOD class
        if task == 'low_motion': self.num_classes = 4  # Still, Train, Subway, OOD
        elif task == 'active': self.num_classes = 4    # Walk, Run, Bike, OOD
        elif task == 'road': self.num_classes = 3      # Car, Bus, OOD
        elif task == 'all':  self.num_classes = 8       # Global Model
        else: raise ValueError(f"Unknown task: {task}")

        # Helper function to read indices from CSV
        def load_indices(csv_path: str) -> list[int]:
            indices = []
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    indices.append(int(row['original_validation_index']))
            return indices

        temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with temp_env.begin() as txn:

            def get_keys(split_target):
                keys_data = txn.get(f"{split_target}_keys".encode('ascii'))
                if keys_data is None:
                    raise ValueError(f"Key list '{split_target}_keys' not found in LMDB.")
                return pickle.loads(keys_data)

            # ----------------------------------------------------
            # Split routing: train = train + val75, val = val25
            # ----------------------------------------------------
            if self.mode == 'train':
                train_keys = get_keys('train')
                val_keys = get_keys('val')
                val75_keys = [val_keys[i] for i in load_indices(val75_csv)]
                all_keys = train_keys + val75_keys

            elif self.mode == 'val':
                val_keys = get_keys('val')
                all_keys = [val_keys[i] for i in load_indices(val25_csv)]

            elif self.mode == 'test':
                all_keys = get_keys('test')
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            self.keys = all_keys

        temp_env.close()

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(buffers=True)

    def __len__(self) -> int:
        return len(self.keys)

    def _get_single_sample(self, idx: int) -> tuple:
        key = self.keys[idx].encode('ascii')
        value = self.txn.get(key)
        data_dict = pickle.loads(bytes(value))

        features = {}
        if 'unishape' in self.models:
            features['unishape'] = torch.from_numpy(data_dict['unishape_features']).float()
        if 'mantis' in self.models:
            features['mantis'] = torch.from_numpy(data_dict['mantis_features']).float()
        if 'trex' in self.models:
            features['trex'] = torch.from_numpy(data_dict['trex_features']).float()
        if 'tspulse' in self.models:
            features['tspulse'] = torch.from_numpy(data_dict['tspulse_features']).float()
        if 'nutime' in self.models:
            features['nutime'] = torch.from_numpy(data_dict['nutime_features']).float()
        if 'fft' in self.models:
            features['fft'] = torch.from_numpy(data_dict['fft_features']).float()

        # Early return for test data (no labels)
        if self.mode == 'test':
            return features, None

        label = data_dict['label']

        # ----------------------------------------------------
        # OOD (Out-of-Distribution) Label Mapping
        # ----------------------------------------------------
        if self.task == 'low_motion':
            if label == 0: label = 0     # Still
            elif label == 6: label = 1   # Train
            elif label == 7: label = 2   # Subway
            else: label = 3              # OOD

        elif self.task == 'active':
            if label == 1: label = 0     # Walk
            elif label == 2: label = 1   # Run
            elif label == 3: label = 2   # Bike
            else: label = 3              # OOD

        elif self.task == 'road':
            if label == 4: label = 0     # Car
            elif label == 5: label = 1   # Bus
            else: label = 2              # OOD

        label_tensor = torch.tensor(label, dtype=torch.long)
        label_onehot = F.one_hot(label_tensor, num_classes=self.num_classes).float()

        return features, label_onehot

    def __getitem__(self, idx: int) -> dict:
        if self.env is None:
            self._init_db()

        features, label = self._get_single_sample(idx)

        output = {}
        if 'unishape' in self.models: output['unishape_features'] = features['unishape']
        if 'mantis' in self.models: output['mantis_features'] = features['mantis']
        if 'trex' in self.models: output['trex_features'] = features['trex']
        if 'tspulse' in self.models: output['tspulse_features'] = features['tspulse'] # Added tspulse mapping to output
        if 'nutime' in self.models: output['nutime_features'] = features['nutime']
        if 'fft' in self.models: output['fft_features'] = features['fft']

        # Only append the label if it's not None (avoids key errors on test data)
        if label is not None:
            output['labels'] = label

        return output

HARFeatureDataset = define_dataset(_HARFeatureDataset)
