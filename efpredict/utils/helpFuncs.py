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

import efpredict

from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights
from torch.utils.data._utils.collate import default_collate

def get_checkpoint(model, optim, scheduler, output, f):
    epoch_resume = 0
    bestLoss = float("inf")
    try:
        # Attempt to load checkpoint
        checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['opt_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        epoch_resume = checkpoint["epoch"] + 1
        bestLoss = checkpoint["best_loss"]
        f.write("Resuming from epoch {}\n".format(epoch_resume))
    except FileNotFoundError:
        f.write("Starting run from scratch\n")
    
    return model, optim, scheduler, epoch_resume, bestLoss

def save_checkpoint(model, period, frames, epoch, output, loss, bestLoss, y, yhat, optim, scheduler):
        #TODO change this to match original run.
        # Save checkpoint
        save = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'period': period,
            'frames': frames,
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
        return bestLoss

def generate_model(model_name, pretrained):
    model_name = model_name.lower()
    assert model_name in ["r2plus1d_18", "mc3_18", "r3d_18"], "Model name must be one of r2plus1d_18, mc3_18, r3d_18"
    weights = None
    if not pretrained:
        model = torchvision.models.video.__dict__[
            model_name](weights=weights)
    else:  
        #Set weights to default of whichever model is chosen
        weights = R2Plus1D_18_Weights.DEFAULT
        if model_name == "mc3_18":
            weights = MC3_18_Weights.DEFAULT
        elif model_name == "r3d_18":
            weights = R3D_18_Weights.DEFAULT

        model = torchvision.models.video.__dict__[
            model_name](weights=weights)
    print("Using model: ", model_name)
    if weights is None:
        print("Using random weights")
    return model

def setup_model(seed, model_name, pretrained, device, weights, frames, period, output, weight_decay, lr, lr_step_period, num_epochs):
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # TODO make output more specific so it doesn't overwrite previous runs, e.g. foldername contains features, model, and hyperparameters
    # Set default output directory
    if output is None:
        pretrained_str = "pretrained" if pretrained else "random"
        output_dir = f"output/{pretrained_str}-num_epochs-{num_epochs}"
        output = os.path.join(output_dir, "video", f"{model_name}_{frames}_{period}_{pretrained_str}")
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model based on model_name
    model = generate_model(model_name, pretrained)

    # Replaced the last layer with a linear layer with 1 output
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    # TODO, echonet repository uses 55.6, but may need to be changed.
    model.fc.bias.data[0] = 55.6

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:  # TODO Check if this is correct - from echonet repository
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

     # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr,
                            momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    return output, device, model, optim, scheduler

def mean_and_std(data_dir, task, frames, period):
    # Compute mean and std
    mean, std = efpredict.utils.get_mean_and_std(efpredict.datasets.EchoDynamic(root=data_dir, split="train"))
    kwargs = {"target_type": task,
            "mean": mean,
            "std": std,
            "length": frames,
            "period": period,
            }
    
    return kwargs

def get_dataset(data_dir, num_train_patients, kwargs):
    # Set up datasets and dataloaders
    dataset = {}

    dataset["unlabelled"] = get_unlabelled_dataset(data_dir)

    # TODO again replace efpredict with own file/functions.
    dataset["train"] = efpredict.datasets.EchoDynamic(root=data_dir, split="train", **kwargs, pad=12)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = efpredict.datasets.EchoDynamic(root=data_dir, split="val", **kwargs)

    return dataset

def get_unlabelled_dataset(data_dir):
    unlabelled_dataset = efpredict.datasets.EchoUnlabelled(root=data_dir)
    return unlabelled_dataset

def plot_results(y, yhat, split, output):  
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
        plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
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
        plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
        plt.close(fig)

def custom_collate(batch):
    input_shape = (3, 112, 112)  # Assuming grayscale images with a single channel
    output_shape = (1,)          # Assuming a single output value

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.empty(0, *input_shape), torch.empty(0, *output_shape)
    return default_collate(batch)