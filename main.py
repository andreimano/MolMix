import argparse
import logging

import torch
from schedulefree import AdamWScheduleFree
from tqdm import tqdm

import wandb as wandb
from datasets.datasets import get_data
from layers.models import get_model
from utils.datasets_utils import get_target_metric
from utils.metrics import Evaluator, IsBetter
from utils.misc import Config, args_canonize, args_unify
from utils.training import Trainer, get_scheduler

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)


def arg_parser():
    parser = argparse.ArgumentParser("training")
    parser.add_argument("--cfg", type=str, default="cfgs/default.yaml")
    args, opts = parser.parse_known_args()
    config = Config()
    config.load(args.cfg, recursive=True)
    config.update(opts)
    return args, config


def main(args, wandb):
    wandb.config.update(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trn_loader, val_loader, tst_loader, task = get_data(args)
    target_metric, criterion = get_target_metric(args)

    trainer = Trainer(
        task=task,
        criterion=criterion,
        evaluator=Evaluator(target_metric),
        target_metric=target_metric,
        norm_target=args.norm_target if hasattr(args, "norm_target") else False,
        grad_clip_val=args.grad_clip_val if hasattr(args, "grad_clip_val") else 1,
        device=device,
    )

    comparison = IsBetter(target_metric)

    final_val_metrics = []
    final_tst_metrics = []

    global_epoch = 1
    for _run in range(args.num_runs):
        logging.info(f"Run {_run}")
        model = get_model(args, device)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=10000,
        )
        scheduler = get_scheduler(
            args, target_metric=target_metric, optimizer=optimizer
        )

        val_metrics = []
        tst_metrics = []

        trainer.clear_stats()

        pbar = tqdm(range(1, args.max_epoch + 1))
        for epoch in pbar:
            trainer.epoch = epoch
            optimizer.train()
            trn_loss, trn_metric = trainer.train_epoch(
                model, trn_loader, optimizer, args.debug
            )
            # wandb.watch(model, log='all')
            optimizer.eval()
            val_loss, val_metric = trainer.eval_epoch(
                model, val_loader, scheduler, validation=True
            )

            if epoch < 2:
                num_params = sum(p.numel() for p in model.parameters())
                logging.info(f"Number of parameters: {num_params}")
                wandb.log({"num_params": num_params})
                if hasattr(args, "log_grads") and args.log_grads:
                    wandb.watch(
                        model,
                        log="all",
                    )

            val_metrics.append(val_metric[target_metric])

            if args.eval_test:
                # Should change this, whole program crashes if eval_test is False
                optimizer.eval()
                tst_loss, tst_metric = trainer.eval_epoch(
                    model, tst_loader, scheduler, validation=False
                )
                tst_metrics.append(tst_metric[target_metric])

            is_better, the_better = comparison(
                val_metric[target_metric], trainer.best_val_metric
            )

            if is_better:
                trainer.best_val_metric = the_better
                trainer.best_tst_metric = (
                    tst_metric[target_metric] if args.eval_test else None
                )
                trainer.best_val_idx = epoch - 1
                trainer.best_val_loss = val_loss
                trainer.patience = 0
            else:
                trainer.patience += 1

            if trainer.patience >= args.patience and epoch >= args.min_epoch:
                break

            log_dict = {
                "train_loss": trn_loss,
                "val_loss": val_loss,
                "lr": scheduler.optimizer.param_groups[0]["lr"],
            }

            for k, v in trn_metric.items():
                log_dict[f"train_{k}"] = v

            for k, v in val_metric.items():
                log_dict[f"val_{k}"] = v

            if args.eval_test:
                log_dict["test_loss"] = tst_loss
                for k, v in tst_metric.items():
                    log_dict[f"test_{k}"] = v

            pbar.set_postfix(log_dict)
            wandb.log(log_dict, step=global_epoch)
            global_epoch += 1

        final_val_metrics.append(trainer.best_val_metric)
        if args.eval_test:
            final_tst_metrics.append(trainer.best_tst_metric)


    wandb.log({"final_val_metric": sum(final_val_metrics) / len(final_val_metrics)})
    wandb.log(
        {"best_val_metric_std": torch.std(torch.tensor(final_val_metrics)).item()}
    )
    if args.dataset in ["bace", "esol", "freesolv", "lipo"]:
        final_metric_mse = [x**2 for x in final_val_metrics]
        wandb.log({"final_metric_mse": sum(final_metric_mse) / len(final_metric_mse)})
        wandb.log(
            {"best_metric_mse_std": torch.std(torch.tensor(final_metric_mse)).item()}
        )

    if args.eval_test:
        wandb.log({"final_tst_metric": sum(final_tst_metrics) / len(final_tst_metrics)})
        wandb.log(
            {"best_tst_metric_std": torch.std(torch.tensor(final_tst_metrics)).item()}
        )
        if args.dataset in ["bace", "esol", "freesolv", "lipo"]:
            final_metric_mse = [x**2 for x in final_tst_metrics]
            wandb.log(
                {"final_metric_mse": sum(final_metric_mse) / len(final_metric_mse)}
            )
            wandb.log(
                {
                    "best_metric_mse_std": torch.std(
                        torch.tensor(final_metric_mse)
                    ).item()
                }
            )

    logging.info(f"Final Val Metric: {sum(final_val_metrics)/len(final_val_metrics)}")
    logging.info(
        f"Final val Metric std: {torch.std(torch.tensor(final_val_metrics)).item()}"
    )
    if args.dataset in ["bace", "esol", "freesolv", "lipo"]:
        final_metric_mse = [x**2 for x in final_val_metrics]
        logging.info(f"Final Metric MSE: {sum(final_metric_mse)/len(final_metric_mse)}")
        logging.info(
            f"Final Metric MSE std: {torch.std(torch.tensor(final_metric_mse)).item()}"
        )

    if args.eval_test:
        logging.info(
            f"Final Tst Metric: {sum(final_tst_metrics)/len(final_tst_metrics)}"
        )
        logging.info(
            f"Final tst Metric std: {torch.std(torch.tensor(final_tst_metrics)).item()}"
        )
        if args.dataset in ["bace", "esol", "freesolv", "lipo"]:
            final_metric_mse = [x**2 for x in final_tst_metrics]
            logging.info(
                f"Final Metric MSE: {sum(final_metric_mse)/len(final_metric_mse)}"
            )
            logging.info(
                f"Final Metric MSE std: {torch.std(torch.tensor(final_metric_mse)).item()}"
            )


def parse_name_cfg(args):
    name_keys = ["model"]
    name = "|"
    for key in name_keys:
        for k in args[key]:
            name += f"{k}={args[key][k]}|"
        name += "+"
    return name


if __name__ == "__main__":
    _, args = arg_parser()
    args = args_unify(args_canonize(args))

    wandb_name = parse_name_cfg(args)
    wandb_name = (
        args.wandb.name + wandb_name if hasattr(args.wandb, "name") else None
    )  # None for sweeps

    wandb.init(
        project=args.wandb.project,
        name=wandb_name,
        mode="online" if args.wandb.use_wandb and not args.debug else "disabled",
        config=vars(args),
        entity=args.wandb.entity,
    )

    main(args, wandb)
