import os
import math
import random
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
    step_resume = 0
    bestLoss = float("inf")
    try:
        # Attempt to load checkpoint
        checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['opt_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        epoch_resume = checkpoint["epoch"]
        step_resume = checkpoint["step"] 
        bestLoss = checkpoint["best_loss"]
        f.write("Resuming from epoch {}\n".format(epoch_resume))
    except FileNotFoundError:
        f.write("Starting run from scratch\n")
    
    return model, optim, scheduler, epoch_resume, step_resume, bestLoss

def save_checkpoint(model, period, frames, epoch, step, output, loss, bestLoss, optim, scheduler):
        #TODO change this to match original run.
        # Save checkpoint
        save = {
            'epoch': epoch,
            'step': step,
            'state_dict': model.state_dict(),
            'period': period,
            'frames': frames,
            'best_loss': bestLoss,
            'loss': loss,
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

def setup_model(seed, model_name, pretrained, device, weights, frames, period, output, weight_decay, lr, lr_step_period, num_epochs, labelled_ratio=False, unlabelled_ratio=False):
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # TODO make output more specific so it doesn't overwrite previous runs, e.g. foldername contains features, model, and hyperparameters
    # Set default output directory
    if output is None:
        pretrained_str = "pretrained" if pretrained else "random"
        output_dir = f"output/{pretrained_str}/epochs-{num_epochs}/"
        if labelled_ratio != False and unlabelled_ratio != False:
            output_dir += f"semisupervised/ratioLU-{labelled_ratio}-{unlabelled_ratio}/"
        else:
            output_dir += "supervised/"
        output = os.path.join(output_dir, f"{model_name}_{frames}_{period}_{pretrained_str}")

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

def get_dataset(data_dir, kwargs, data_type="ALL", percentage_dynamic_labelled=100, percentage_pediatric_labelled=100):
    # Set up datasets and dataloaders
    dataset = {}

    pediatric_train = efpredict.datasets.EchoPediatric(root=data_dir, split="train", data_type=data_type, **kwargs)
    pediatric_val = efpredict.datasets.EchoPediatric(root=data_dir, split="val", data_type=data_type,**kwargs)
    pediatric_test = efpredict.datasets.EchoPediatric(root=data_dir, split="test", data_type=data_type,**kwargs)
    print("Pediatric train: ", len(pediatric_train))

    dynamic_train = efpredict.datasets.EchoDynamic(root=data_dir, split="train", **kwargs)
    dynamic_val = efpredict.datasets.EchoDynamic(root=data_dir, split="val", **kwargs)
    dynamic_test = efpredict.datasets.EchoDynamic(root=data_dir, split="test", **kwargs)
    print("Dynamic train: ", len(dynamic_train))

    pediatric_train_loader = torch.utils.data.DataLoader(pediatric_train, batch_size=1)
    pediatric_val_loader = torch.utils.data.DataLoader(pediatric_val, batch_size=1)
    pediatric_test_loader = torch.utils.data.DataLoader(pediatric_test, batch_size=1)
    print("1")

    dynamic_train_loader = torch.utils.data.DataLoader(dynamic_train, batch_size=1)
    dynamic_val_loader = torch.utils.data.DataLoader(dynamic_val, batch_size=1)
    dynamic_test_loader = torch.utils.data.DataLoader(dynamic_test, batch_size=1)
    print("2")

    pediatric_train_list = [item for batch in pediatric_train_loader for item in batch]
    pediatric_val_list = [item for batch in pediatric_val_loader for item in batch]
    pediatric_test_list = [item for batch in pediatric_test_loader for item in batch]
    print("3")

    dynamic_train_list = [item for batch in dynamic_train_loader for item in batch]
    dynamic_val_list = [item for batch in dynamic_val_loader for item in batch]
    dynamic_test_list = [item for batch in dynamic_test_loader for item in batch]
    print("4")



    # pediatric_train_list = list(pediatric_train)
    # pediatric_val_list = list(pediatric_val)
    # print("Pediatric val list: ", len(pediatric_val_list))
    # pediatric_test_list = list(pediatric_test)
    # print("Pediatric test list: ", len(pediatric_test_list))

    # dynamic_train_list = list(dynamic_train)
    # dynamic_val_list = list(dynamic_val)
    # dynamic_test_list = list(dynamic_test)
    print("Dynamic train list: ", len(dynamic_train_list))

    pediatric_train_labelled = random.sample(pediatric_train_list, int(len(pediatric_train_list) * percentage_pediatric_labelled / 100))
    pediatric_val_labelled = random.sample(pediatric_val_list, int(len(pediatric_val_list) * percentage_pediatric_labelled / 100))
    pediatric_test_labelled = random.sample(pediatric_test_list, int(len(pediatric_test_list) * percentage_pediatric_labelled / 100))
    print("Pediatric labelled: ", len(pediatric_train_labelled) + len(pediatric_val_labelled) + len(pediatric_test_labelled))

    dynamic_train_labelled = random.sample(dynamic_train_list, int(len(dynamic_train_list) * percentage_dynamic_labelled / 100))
    dynamic_val_labelled = random.sample(dynamic_val_list, int(len(dynamic_val_list) * percentage_dynamic_labelled / 100))
    dynamic_test_labelled = random.sample(dynamic_test_list, int(len(dynamic_test_list) * percentage_dynamic_labelled / 100))
    print("Dynamic labelled: ", len(dynamic_train_labelled) + len(dynamic_val_labelled) + len(dynamic_test_labelled))

    dataset["train"] = torch.utils.data.ConcatDataset(pediatric_train_labelled + dynamic_train_labelled)
    dataset["val"] = torch.utils.data.ConcatDataset(pediatric_val_labelled + dynamic_val_labelled)
    dataset["test"] = torch.utils.data.ConcatDataset(pediatric_test_labelled + dynamic_test_labelled)
    print("Total labelled: ", len(dataset["train"]) + len(dataset["val"]) + len(dataset["test"]))

    pediatric_train_unlabelled = list(set(pediatric_train_list) - set(pediatric_train_labelled))
    pediatric_val_unlabelled = list(set(pediatric_val_list) - set(pediatric_val_labelled))
    pediatric_test_unlabelled = list(set(pediatric_test_list) - set(pediatric_test_labelled))
    print("Pediatric unlabelled: ", len(pediatric_train_unlabelled) + len(pediatric_val_unlabelled) + len(pediatric_test_unlabelled))

    dynamic_train_unlabelled = list(set(dynamic_train_list) - set(dynamic_train_labelled))
    dynamic_val_unlabelled = list(set(dynamic_val_list) - set(dynamic_val_labelled))
    dynamic_test_unlabelled = list(set(dynamic_test_list) - set(dynamic_test_labelled))
    print("Dynamic unlabelled: ", len(dynamic_train_unlabelled) + len(dynamic_val_unlabelled) + len(dynamic_test_unlabelled))

    dataset["unlabelled"] = torch.utils.data.ConcatDataset(pediatric_train_unlabelled + pediatric_val_unlabelled + pediatric_test_unlabelled + dynamic_train_unlabelled + dynamic_val_unlabelled + dynamic_test_unlabelled)
    print("Total unlabelled: ", len(dataset["unlabelled"]))
    
    return dataset

def get_unlabelled_dataset(data_dir):
    unlabelled_dataset = efpredict.datasets.EchoUnlabelled(root=data_dir)
    return unlabelled_dataset

def test_resuls(f, output, model, dataset, batch_size, num_workers, device):
    for split in ["val", "test"]:
        # Performance without test-time augmentation
        ds = dataset[split]
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
        loss, yhat, y = efpredict.utils.EFPredictSupervised.run_epoch(model, dataloader, False, None, device, 0, None)
        f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *efpredict.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
        f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *efpredict.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
        f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, efpredict.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
        f.flush()

        # Write full performance to file
        with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
            for ds_part in ds.datasets:  # Iterate through the datasets within the ConcatDataset
                for (filename, pred) in zip(ds_part.fnames, yhat):
                    if np.isscalar(pred) or isinstance(pred, (int, float)):  # check if pred is a single value
                        pred = [pred]  # wrap it in a list to handle it properly
                    for (i, p) in enumerate(pred):
                        g.write("{},{},{:.4f}\n".format(filename, i, p))
        efpredict.utils.latexify()
        yhat = np.array(list(map(lambda x: x.mean(), yhat)))

        # Calculate the mean and standard deviation of the predictions and print them
        mean = np.mean(yhat)
        std = np.std(yhat)
        print("Mean: {:.2f}".format(mean))
        print("Std: {:.2f}".format(std))

        #TODO check that y is the real EF
        #Print the MAE
        print("MAE: {:.2f}".format(sklearn.metrics.mean_absolute_error(y, yhat)))
        #Print the RMSE
        print("RMSE: {:.2f}".format(math.sqrt(sklearn.metrics.mean_squared_error(y, yhat))))
        #Print the R2
        print("R2: {:.2f}".format(sklearn.metrics.r2_score(y, yhat)))

        # plot_results(y, yhat, split, output)

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
    #TODO make sure this is preprocessing the data correctly.
    input_shape = (3, 112, 112)  # Assuming grayscale images with a single channel
    output_shape = (1,)          # Assuming a single output value

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.empty(0, *input_shape), torch.empty(0, *output_shape)
    return default_collate(batch)