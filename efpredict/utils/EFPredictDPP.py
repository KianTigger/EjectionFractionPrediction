import os
import math
import time

import numpy as np
import click
import matplotlib.pyplot as plt
import torch
import torchvision
import sklearn.metrics
import tqdm

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader

import efpredict

@click.command("predictdpp")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
    default="r2plus1d_18")
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--num_epochs", type=int, default=45)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)

def is_distributed():
    return 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1

def ddp_setup(rank, world_size):
    """
    Args:
        rank (int): Rank (identifier) of the current process.
        world_size (int): Number of processes participating in the job.
    """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    # TODO think nccl is needed but might be gloo instead.
    # init_process_group(backend="gloo", rank=rank, world_size=world_size)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class EFPredictDPP:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int = 10,
        data_dir=None,
        output=None,
        task="EF",

        model_name="r2plus1d_18",

        # UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
        pretrained=True,

        # UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. 
        # The current behavior is equivalent to passing `weights=R2Plus1D_18_Weights.KINETICS400_V1`. 
        # You can also use `weights=R2Plus1D_18_Weights.DEFAULT` to get the most up-to-date weights.
        weights=None,

        run_test=False,
        num_epochs=45,
        lr=1e-4,
        weight_decay=1e-4,
        lr_step_period=15,
        frames=32,
        period=2,
        num_train_patients=None,
        num_workers=4,
        batch_size=20,
        device=None,
        seed=0,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.data_dir = data_dir
        self.output = output
        self.task = task
        self.model_name = model_name
        self.pretrained = pretrained
        self.weights = weights
        self.run_test = run_test
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step_period = lr_step_period
        self.frames = frames
        self.period = period
        self.num_train_patients = num_train_patients
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
    
    def _run_epoch(self, model, dataloader, train, optim, device, save_all=False, block_size=None):
        """Run one epoch of training/evaluation for ejection fraction prediction.

        Args:
            model (torch.nn.Module): Model to train/evaulate.
            dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
            train (bool): Whether or not to train model.
            optim (torch.optim.Optimizer): Optimizer
            device (torch.device): Device to run on
            save_all (bool, optional): If True, return predictions for all
                test-time augmentations separately. If False, return only
                the mean prediction.
                Defaults to False.
            block_size (int or None, optional): Maximum number of augmentations
                to run on at the same time. Use to limit the amount of memory
                used. If None, always run on all augmentations simultaneously.
                Default is None.
        """

        model.train(train)

        total = 0  # total training loss
        n = 0      # number of videos processed
        s1 = 0     # sum of ground truth EF
        s2 = 0     # Sum of ground truth EF squared

        yhat = []
        y = []

        with torch.set_grad_enabled(train):
            with tqdm.tqdm(total=len(dataloader)) as pbar:
                for (X, outcome) in dataloader:

                    y.append(outcome.numpy())
                    X = X.to(device)
                    outcome = outcome.to(device)

                    average = (len(X.shape) == 6)
                    if average:
                        batch, n_clips, c, f, h, w = X.shape
                        X = X.view(-1, c, f, h, w)

                    s1 += outcome.sum()
                    s2 += (outcome ** 2).sum()

                    if block_size is None:
                        outputs = model(X)
                    else:
                        outputs = torch.cat([model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])

                    if save_all:
                        yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                    if average:
                        outputs = outputs.view(batch, n_clips, -1).mean(1)

                    if not save_all:
                        yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                    loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)

                    if train:
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                    total += loss.item() * X.size(0)
                    n += X.size(0)

                    pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                    pbar.update()

        if not save_all:
            yhat = np.concatenate(yhat)
        y = np.concatenate(y)

        return total / n, yhat, y
    
    def _save_checkpoint(self, epoch, output, loss, bestLoss, y, yhat, optim, scheduler):
        #TODO change this to match original run.
        # Save checkpoint
        save = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'period': self.period,
            'frames': self.frames,
            'best_loss': bestLoss,
            'loss': loss,
            'r2': sklearn.metrics.r2_score(y, yhat),
            'opt_dict': optim.state_dict(),
            'scheduler_dict':scheduler.state_dict(),
        }
        torch.save(save, os.path.join(output, "checkpoint.pt"))
        if loss < bestLoss:
            torch.save(save, os.path.join(output, "best.pt"))
            bestLoss = loss
    
    def _optimizer_and_scheduler(self):
        # Set up optimizer and scheduler
        optim = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                momentum=0.9, weight_decay=self.weight_decay)
        if lr_step_period is None:
            lr_step_period = math.inf
        scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
        
        return optim, scheduler

    def _mean_and_std(self):
        # Compute mean and std
        mean, std = efpredict.utils.get_mean_and_std(efpredict.datasets.EchoDynamic(root=self.data_dir, split="train"))
        kwargs = {"target_type": self.task,
                "mean": mean,
                "std": std,
                "length": self.frames,
                "period": self.period,
                }
        
        return kwargs

    def _dataset(self, kwargs):
        # Set up datasets and dataloaders
        dataset = {}
        # TODO again replace efpredict with own file/functions.
        dataset["train"] = efpredict.datasets.EchoDynamic(root=self.data_dir, split="train", **kwargs, pad=12)
        if self.num_train_patients is not None and len(dataset["train"]) > self.num_train_patients:
            # Subsample patients (used for ablation experiment)
            indices = np.random.choice(len(dataset["train"]), self.num_train_patients, replace=False)
            dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
        dataset["val"] = efpredict.datasets.EchoDynamic(root=self.data_dir, split="val", **kwargs)

        return dataset

    def plot_results(self, y, yhat, split):
        # Plot actual and predicted EF
        fig = plt.figure(figsize=(3, 3))
        lower = min(y.min(), yhat.min())
        upper = max(y.max(), yhat.max())
        plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
        plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
        plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("Actual EF (%)")
        plt.ylabel("Predicted EF (%)")
        plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
        plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output, "{}_scatter.pdf".format(split)))
        plt.close(fig)

        # Plot AUROC
        fig = plt.figure(figsize=(3, 3))
        plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
        for thresh in [35, 40, 45, 50]:
            fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
            print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
            plt.plot(fpr, tpr)

        plt.axis([-0.01, 1.01, -0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output, "{}_roc.pdf".format(split)))
        plt.close(fig)

    def train(self):
        optim, scheduler = self._optimizer_and_scheduler()
        kwargs = self._mean_and_std()
        dataset = self._dataset(kwargs)
        
        # Run training and testing loops
        with open(os.path.join(self.output, "log.csv"), "a") as f:
            epoch_resume = 0
            bestLoss = float("inf")
            try:
                # Attempt to load checkpoint
                checkpoint = torch.load(os.path.join(self.output, "checkpoint.pt"))
                self.model.module.load_state_dict(checkpoint['state_dict'])
                optim.load_state_dict(checkpoint['opt_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_dict'])
                epoch_resume = checkpoint["epoch"] + 1
                bestLoss = checkpoint["best_loss"]
                f.write("Resuming from epoch {}\n".format(epoch_resume))
            except FileNotFoundError:
                f.write("Starting run from scratch\n")

            for epoch in range(epoch_resume, self.num_epochs):
                print("Epoch #{}".format(epoch), flush=True)
                for phase in ['train', 'val']:
                    start_time = time.time()
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(i)

                    ds = dataset[phase]
                    dataloader = prepare_dataloader(ds, self.batch_size, 
                        num_workers=self.num_workers, shuffle=True, 
                        pin_memory=(self.device.type == "cuda"), drop_last=(phase == "train")) 
                    # dataloader = torch.utils.data.DataLoader(
                    #     ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=(self.device.type == "cuda"), drop_last=(phase == "train"))

                    # loss, yhat, y = efpredict.utils.EFPredictDPP.run_epoch(self.model, dataloader, phase == "train", optim, self.device)
                    loss, yhat, y = self._run_epoch(self.model, dataloader, phase == "train", optim, self.device)

                    f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                phase,
                                                                loss,
                                                                sklearn.metrics.r2_score(y, yhat),
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                self.batch_size))
                    f.flush()
                scheduler.step()
                if torch.cuda.current_device() == 0:
                    self._save_checkpoint(epoch, self.output, loss, bestLoss, y, yhat, optim, scheduler)


            # Load best weights
            if self.num_epochs != 0:
                checkpoint = torch.load(os.path.join(self.output, "best.pt"))
                self.model.module.load_state_dict(checkpoint['state_dict'])
                f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
                f.flush()

            if self.run_test:
                for split in ["val", "test"]:
                    # # Performance without test-time augmentation
                    # dataloader = torch.utils.data.DataLoader(
                    #     efpredict.datasets.Echo(root=self.data_dir, split=split, **kwargs),
                    #     batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, 
                    #     pin_memory=(self.device.type == "cuda"), sampler=DistributedSampler(dataset["train"]) if distributed else None, drop_last=False)
                    
                    dataset = efpredict.datasets.Echo(root=self.data_dir, split=split, **kwargs)
                    #TODO swap this out for my ds loader.
                    dataloader = prepare_dataloader(dataset, self.batch_size, 
                        num_workers=self.num_workers, shuffle=True, 
                        pin_memory=(self.device.type == "cuda"), drop_last=False) 
                    
                    
                    # loss, yhat, y = efpredict.utils.EFPredictDPP.run_epoch(self.model, dataloader, False, None, self.device)
                    loss, yhat, y = self._run_epoch(self.model, dataloader, False, None, self.device)

                    f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *efpredict.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                    f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *efpredict.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                    f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, efpredict.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                    f.flush()

                    # Performance with test-time augmentation
                    ds = efpredict.datasets.Echo(root=self.data_dir, split=split, **kwargs, clips="all")
                    #TODO swap this out for my ds loader.
                    
                    dataloader = prepare_dataloader(ds, 1, 
                    num_workers=self.num_workers, shuffle=False, 
                    pin_memory=(self.device.type == "cuda")) 
                    # dataloader = torch.utils.data.DataLoader(
                    #     ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                    # loss, yhat, y = efpredict.utils.EFPredictDPP.run_epoch(self.model, dataloader, False, None, self.device, save_all=True, block_size=self.batch_size)
                    loss, yhat, y = self._run_epoch(self.model, dataloader, False, None, self.device, save_all=True, block_size=self.batch_size)

                    f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *efpredict.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                    f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *efpredict.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                    f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, efpredict.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                    f.flush()

                    # Write full performance to file
                    with open(os.path.join(self.output, "{}_predictions.csv".format(split)), "w") as g:
                        for (filename, pred) in zip(ds.fnames, yhat):
                            for (i, p) in enumerate(pred):
                                g.write("{},{},{:.4f}\n".format(filename, i, p))
                    efpredict.utils.latexify()
                    yhat = np.array(list(map(lambda x: x.mean(), yhat)))

                    # Plot actual and predicted EF
                    self.plot_results(y, yhat, split)


def load_train_objs():
    kwargs = EFPredictDPP._mean_and_std()
    train_set = EFPredictDPP._dataset(kwargs)  # load your dataset
    #TODO
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 0,
        pin_memory: bool = True, shuffle: bool = True, drop_last: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        sample=DistributedSampler(dataset) if is_distributed() else None,
    )

def run():
    world_size = torch.cuda.device_count()
    save_every = 1
    total_epochs = 5
    batch_size = 20
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)


def main(rank: int, world_size: int, save_every: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = EFPredictDPP(model, train_data, optimizer, rank, save_every)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    run()
    # import argparse
    # parser = argparse.ArgumentParser(description='simple distributed training job')
    # parser.add_argument('total_epochs', default=45, type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', default=1, type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    # args = parser.parse_args()
    
    # world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    
