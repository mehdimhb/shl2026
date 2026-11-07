"""Train a classification head or an early-fusion ensemble from a task config."""
import argparse
import json
from pathlib import Path

import torch

from src import task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ef-unishape-mantis-nutime-tirex", help="Folder name under configs/task")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", help="Override the configured device, e.g. cpu or cuda:1")
    parser.add_argument("--resume", action="store_true", help="Resume from the run's last.ckpt")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    config_dir = Path("configs/task") / args.config
    overrides = None
    if args.device:
        parameters = json.loads((config_dir / "task_config.json").read_text())["parameters"]
        overrides = {"task_config": task.TaskConfig(**{**parameters, "device": args.device})}

    the_task = task.Task.from_config(config_dir / "task.json", new_parameters=overrides)

    if args.resume:
        the_task.resume(epochs=args.epochs)
    else:
        the_task.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
