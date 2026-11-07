import torch
from lightning.pytorch import LightningModule

from ..model import BaseModel
from ..loss import CompositeLoss
from ..metric import MetricSet
from ..optimizer import BaseOptimizer


class ModelModule(LightningModule):
    def __init__(
        self,
        model: BaseModel,
        loss: CompositeLoss,
        metric: MetricSet,
        optimizer: BaseOptimizer,
        n_supervision: int = 16,
        trm: bool = False
    ):
        super().__init__()

        self.model = model
        self.optimizers_config = optimizer
        self.n_supervision = n_supervision
        self.trm = trm

        if self.trm:
            self.automatic_optimization = False

        self.train_loss = loss(stage="train")
        self.train_metric = metric(stage="train")

        self.val_loss = loss(stage="val")
        self.val_metric = metric(stage="val")

        self.test_loss = loss(stage="test")
        self.test_metric = metric(stage="test")

    def forward(self, batch: dict[str, torch.Tensor], z=None, y=None) -> dict[str, torch.Tensor]:
        if self.trm:
            return self.model(batch, z_init=z, y_init=y)
        return self.model(batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.trm:
            opt = self.optimizers()
            scheduler = self.lr_schedulers()

            y, z = None, None

            for step in range(self.n_supervision):
                opt.zero_grad()
                output = self.forward(batch, z=z, y=y)

                loss = self.train_loss.step(output)
                self.train_metric.step(output)

                self.manual_backward(loss)

                # --- CRITICAL: Manual Gradient Clipping ---
                self.clip_gradients(
                    opt,
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm="norm"
                )

                opt.step()

                y, z = output["y_next"], output["z_next"]

                if output["q_logits"].mean().item() > 0:
                    break

            if scheduler is not None:
                scheduler.step()
        else:
            output = self.forward(batch)
            loss = self.train_loss.step(output)
            self.train_metric.step(output)
            return loss

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute_and_reset()
        metric = self.train_metric.compute_and_reset()
        self.logger.log_metrics(metrics=loss | metric, step=self.current_epoch)

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        if self.trm:
            y, z = None, None

            # The full N_sup steps are done at test/val time to maximize performance
            for step in range(self.n_supervision):
                output = self.forward(batch, z=z, y=y)
                y, z = output["y_next"], output["z_next"]

            # Compute loss/metrics ONLY on the final step's prediction
            self.val_loss.step(output)
            self.val_metric.step(output)
        else:
            output = self.forward(batch)
            # output["labels"] = output["labels"].argmax(dim=1)

            loss = self.val_loss.step(output)
            self.val_metric.step(output)
            return loss

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute_and_reset()
        metric = self.val_metric.compute_and_reset()
        self.logger.log_metrics(metrics=loss | metric, step=self.current_epoch)

        # val_loss_total = loss.get("val/loss/Total")
        # if val_loss_total is not None:
        #     self.log(
        #         name="val/loss/Total",
        #         value=val_loss_total,
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         sync_dist=True,
        #         logger=False,
        #     )

        val_acc = metric.get("val/metric/Accuracy")
        self.log(
            name="val/metric/Accuracy",
            value=val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=False,
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.trm:
            y, z = None, None

            for step in range(self.n_supervision):
                output = self.forward(batch, z=z, y=y)
                y, z = output["y_next"], output["z_next"]

            self.test_loss.step(output)
            self.test_metric.step(output)
        else:
            output = self.forward(batch)
            loss = self.test_loss.step(output)
            self.test_metric.step(output)
            return loss

    def on_test_epoch_end(self) -> None:
        loss = self.test_loss.compute_and_reset()
        metric = self.test_metric.compute_and_reset()
        self.logger.log_metrics(metrics=loss | metric, step=self.current_epoch)

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        return self.optimizers_config.setup(self, total_steps)
