import os
import math
import time

import numpy as np
import click
import torch
import torchvision
import sklearn.metrics
import tqdm

import efpredict
import efpredict.utils.helpFuncs as helpFuncs
from efpredict.datasets.datasets import LabelledDataset, UnlabelledDataset
from torch.utils.data import DataLoader

# import pretrained r2plus1d_18 model from torchvision
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights

@click.command("predict")
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
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--run_test", default=False, is_flag=True)
@click.option("--labelled_ratio", type=int, default=10)
@click.option("--unlabelled_ratio", type=int, default=1)
@click.option("--data_type", type=click.Choice(["ALL", "A4C", "PSAX"]), default="A4C")
@click.option("--percentage_dynamic_labelled", type=int, default=100)

def run(
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

    data_type="A4C",
    percentage_dynamic_labelled=100, 
    train_val_test_unlabel_split=[0.7, 0.15, 0.15, 0],

    labelled_ratio=10,
    unlabelled_ratio=1,

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
):
    # TODO Write docstrings, and explanations for args
    kwargs = helpFuncs.mean_and_std(data_dir, task, frames, period)

    dataset = helpFuncs.get_dataset(data_dir, kwargs, data_type, percentage_dynamic_labelled, train_val_test_unlabel_split)

    output, device, model, optim, scheduler = helpFuncs.setup_model(seed, model_name, pretrained, device, weights, frames, period, output, weight_decay, lr, lr_step_period, num_epochs, labelled_ratio, unlabelled_ratio)

    success = False

    while not success:
        try:
            print("Starting training loop")
            run_loops(output, device, model, optim, scheduler, dataset, num_epochs, batch_size, num_workers, period, frames, run_test, labelled_ratio, unlabelled_ratio)
            success = True
        except RuntimeError as e:
            if "DataLoader worker" in str(e) and "is killed by signal: Killed" in str(e):
                print("DataLoader worker killed. Restarting...")
            else:
                # raise e
                print("RuntimeError: {}".format(e))
                print("Restarting...")

def run_loops(output, device, model, optim, scheduler, dataset, num_epochs, batch_size, num_workers, period, frames, run_test, labelled_ratio, unlabelled_ratio):
    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        model, optim, scheduler, epoch_resume, step_resume, bestLoss = helpFuncs.get_checkpoint(model, optim, scheduler, output, f)
        epoch_resume += 1
        if epoch_resume > num_epochs and run_test:
            print("Running tests")
        else:
            print("Resuming from epoch #{}".format(epoch_resume), flush=True)
        for epoch in range(epoch_resume, num_epochs + 1):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
                               
                labelled_dataset = LabelledDataset(labelled_data=dataset[phase])
                unlabelled_dataset = UnlabelledDataset(unlabelled_data=dataset["unlabelled"])
                
                labelled_batch_size = max(1, int(batch_size * labelled_ratio / (labelled_ratio + unlabelled_ratio)))
                unlabelled_batch_size = batch_size - labelled_batch_size

                labelled_dataloader = DataLoader(
                    labelled_dataset, batch_size=labelled_batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                
                unlabelled_dataloader = None
                if phase == "train" and unlabelled_batch_size > 0 and len(unlabelled_dataset) > 0:
                    unlabelled_dataloader = DataLoader(
                        unlabelled_dataset, batch_size=unlabelled_batch_size, num_workers=num_workers, shuffle=True, 
                        pin_memory=(device.type == "cuda"), drop_last=True,  collate_fn=helpFuncs.custom_collate)
                    
                checkpoint_args = {"period": period, "frames": frames, "epoch": epoch, "output": output, "loss": bestLoss, "bestLoss": bestLoss, "scheduler": scheduler}
                loss, yhat, y = efpredict.utils.EFPredict.run_epoch(model, labelled_dataloader, phase == "train", optim, device, step_resume, checkpoint_args, unlabelled_dataloader=unlabelled_dataloader)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for _ in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_reserved() for _ in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()

                step_resume = 0


            scheduler.step()
            
            bestLoss = helpFuncs.save_checkpoint(model, period, frames, epoch, 0, output, loss, bestLoss, optim, scheduler)

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()
            print("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            helpFuncs.test_resuls(f, output, model, dataset, batch_size, num_workers, device)  



def run_epoch(model, labelled_dataloader, train, optim, device, step_resume, checkpoint_args, save_all=False, block_size=None, unlabelled_dataloader=None):
    """Run one epoch of training/evaluation for ejection fraction prediction.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (DataLoader): Dataloader for dataset.
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

    num_items = len(labelled_dataloader)

    unlabelled_iterator = None
    if unlabelled_dataloader is not None:
        unlabelled_iterator = iter(unlabelled_dataloader)

    with torch.set_grad_enabled(train):
        if step_resume > 0:
            print("Skipping {} steps".format(step_resume))
        with tqdm.tqdm(total=num_items) as pbar:
            for step, (X, outcome) in enumerate(labelled_dataloader):
                if step_resume > 0 and step < step_resume:
                    # Skip steps before step_resume
                    pbar.update(1)
                    continue

                if step != 0 and step % (int(num_items//10)) == 0 and train:
                    helpFuncs.save_checkpoint(model, checkpoint_args["period"], checkpoint_args["frames"], 
                            checkpoint_args["epoch"], step, checkpoint_args["output"], checkpoint_args["loss"], 
                            checkpoint_args["bestLoss"], optim, checkpoint_args["scheduler"])

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

                yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)
                 
                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)

                if train and (unlabelled_dataloader is not None):
                    
                    # Sample a batch from the unlabelled dataset
                    try:
                        unlabelled_X, _ = next(unlabelled_iterator)
                    except StopIteration:
                        unlabelled_iterator = iter(unlabelled_dataloader)
                        unlabelled_X, _ = next(unlabelled_iterator)

                    if len(unlabelled_X) > 0 and isinstance(unlabelled_X[0], torch.Tensor) and unlabelled_X[0].shape[0] != 0 and unlabelled_X is not None:
                        unlabelled_X = unlabelled_X.to(device)
                         
                        # Compute consistency loss between labelled and unlabelled data
                        unlabelled_outputs = model(unlabelled_X)
                        size_diff = outputs.size(0) - unlabelled_outputs.size(0)

                        # Pad the smaller tensor with zeros
                        if size_diff > 0:
                            padding = torch.zeros(size_diff, *unlabelled_outputs.size()[1:], device=unlabelled_outputs.device)
                            unlabelled_outputs = torch.cat((unlabelled_outputs, padding), dim=0)
                        elif size_diff < 0:
                            padding = torch.zeros(-size_diff, *outputs.size()[1:], device=outputs.device)
                            outputs = torch.cat((outputs, padding), dim=0)

                        consistency_loss = torch.nn.functional.mse_loss(outputs.view(-1), unlabelled_outputs.view(-1))
                        # Add consistency loss to the original loss
                        loss += consistency_loss

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
