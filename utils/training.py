import math

import torch
import torch.optim as optim
from torch.optim import Optimizer

SCHEDULER_MODE = {
    "acc": "max",
    "mae": "min",
    "mse": "min",
    "rocauc": "max",
    "rmse": "min",
    "ap": "max",
    "mrr": "max",
    "mrr_self_filtered": "max",
    "f1_macro": "max",
}


class NoScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, *args, **kwargs):
        pass


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int = 5,
    num_training_steps: int = 250,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(args, target_metric, optimizer):
    if hasattr(args, "scheduler_type") and args.scheduler_type == "cos_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=(
                args.scheduler_warmup if hasattr(args, "scheduler_warmup") else 5
            ),
            num_training_steps=(
                args.scheduler_patience if hasattr(args, "scheduler_patience") else 50
            ),
        )
    elif hasattr(args, "scheduler_type") and args.scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=SCHEDULER_MODE[target_metric],
            factor=0.5,
            patience=(
                args.scheduler_patience if hasattr(args, "scheduler_patience") else 50
            ),
            min_lr=1.0e-5,
        )
    else:
        scheduler = NoScheduler(optimizer)

    return scheduler


class Trainer:
    def __init__(
        self,
        task,
        criterion,
        evaluator,
        target_metric,
        norm_target,
        grad_clip_val,
        device,
    ):
        super(Trainer, self).__init__()

        # only supports graph clf for now

        self.criterion = criterion
        self.evaluator = evaluator
        self.target_metric = target_metric
        self.norm_target = norm_target
        self.grad_clip_val = grad_clip_val
        self.device = device
        self.clear_stats()

    def clear_stats(self):
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.best_tst_metric = None
        self.patience = 0

    def train_epoch(self, model, trn_loader, optimizer, debug=False):
        model.train()

        if debug:
            counter = 0

        train_losses = []
        preds = []
        labels = []

        for data in trn_loader.loader:
            data = data.to(torch.device(self.device))

            # skip for BN during training
            if data.batch.shape[0] == 1:
                continue
            if hasattr(data, "x_batch") and data.x_batch.shape[0] == 1:
                continue
            if hasattr(data, "z_batch") and data.z_batch.shape[0] == 1:
                continue

            optimizer.zero_grad()

            out = model(data)

            if trn_loader.std is not None and self.norm_target:
                with torch.no_grad():
                    mean, std = trn_loader.std[0].to(self.device), trn_loader.std[1].to(
                        self.device
                    )
                    data.y = (data.y - mean) / std

            loss = self.criterion(out, data.y)
            loss.backward()

            optimizer.step()

            if trn_loader.std is not None and self.norm_target:
                with torch.no_grad():
                    data.y = data.y * std + mean
                    out = out * std + mean

            preds.append(out)
            labels.append(data.y)

            train_losses.append(loss.item())

            if debug:
                counter += 1
                if counter > 3:
                    break

        trn_loss = sum(train_losses) / len(train_losses)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        trn_metric = self.evaluator(labels, preds)

        loss_centr = 0

        return trn_loss + loss_centr, trn_metric

    def eval_epoch(self, model, loader, scheduler=None, validation=False):
        model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for data in loader.loader:
                data = data.to(torch.device(self.device))

                out = model(data, train=True)

                if loader.std is not None and self.norm_target:
                    mean, std = loader.std[0].to(self.device), loader.std[1].to(
                        self.device
                    )
                    out = out * std + mean

                preds.append(out)
                labels.append(data.y)

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            tst_loss = self.criterion(preds, labels).item()  # fix for rmse
            tst_metric = self.evaluator(labels, preds)

        if scheduler is not None and validation:
            if type(tst_metric[self.target_metric]) == torch.tensor:
                tst_metric[self.target_metric] = tst_metric[self.target_metric].item()
            (
                scheduler.step()
                if "LambdaLR" in str(type(scheduler))
                else scheduler.step(tst_metric[self.target_metric])
            )

        return tst_loss, tst_metric
