import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from .base import BaseMetric, define_metric


@define_metric
class HARAccuracy(BaseMetric):
    def __init__(self, stage: str, num_classes: int = 8, ignore_index: int = -100) -> None:
        # Initialize the torchmetrics accuracy object
        metric_fn = MulticlassAccuracy(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='micro' # 'micro' for standard overall accuracy
        )
        super().__init__(
            metric_fn=metric_fn, name="Accuracy", stage=stage
        )

    def step(self, output: dict[str, torch.Tensor]) -> None:
        # torchmetrics automatically handles logits vs argmax internally
        target_indices = torch.argmax(output["labels"], dim=-1)
        self.metric_fn.update(output["logits"], target_indices)


@define_metric
class HARF1Score(BaseMetric):
    def __init__(self, stage: str, num_classes: int = 8, ignore_index: int = -100) -> None:
        # Initialize the torchmetrics F1 object
        metric_fn = MulticlassF1Score(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='macro' # Macro average computes F1 per class and averages them
        )
        super().__init__(
            metric_fn=metric_fn, name="F1", stage=stage
        )

    def step(self, output: dict[str, torch.Tensor]) -> None:
        target_indices = torch.argmax(output["labels"], dim=-1)
        self.metric_fn.update(output["logits"], target_indices)
