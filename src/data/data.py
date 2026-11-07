from .base import BaseData
from .feature_dataset import HARFeatureDataset


class HARFeatureData(BaseData):
    def __init__(
        self,
        lmdb_path: str,
        models: list[str] = ['unishape', 'mantis', 'trex', 'tspulse', 'nutime', 'fft'],
        task: str = 'all',
        val75_csv: str = 'splits/val75_indices.csv',
        val25_csv: str = 'splits/val25_indices.csv',
    ) -> None:
        self.lmdb_path = lmdb_path

        super().__init__(
            dataset=HARFeatureDataset(
                lmdb_path=self.lmdb_path,
                models=models,
                task=task,
                val75_csv=val75_csv,
                val25_csv=val25_csv
            ),
            collator=None,
        )

        self.capture_config()
