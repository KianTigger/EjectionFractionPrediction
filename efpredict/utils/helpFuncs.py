import os
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import sklearn.metrics

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

def delete_checkpoint(output):
    try:
        os.remove(os.path.join(output, "checkpoint.pt"))
    except FileNotFoundError:
        pass

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

def setup_model(seed, model_name, pretrained, device, weights, frames, 
                period, output, weight_decay, lr, lr_step_period, 
                num_epochs, labelled_ratio=False, unlabelled_ratio=False,
                data_type=None, percentage_dynamic_labelled=None, 
                train_val_test_unlabel_split=None, loss_type=None, 
                alpha=None, num_augmented_videos=None, dropout_only=None, 
                rotation_only=None, dropout_int=None, rotation_int=None):
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # TODO make output more specific so it doesn't overwrite previous runs, e.g. foldername contains features, model, and hyperparameters
    # Set default output directory
    if output is None:
        pretrained_str = "pretrained" if pretrained else "random"
        output_dir = f"output/{pretrained_str}/epochs-{num_epochs}/"
        if labelled_ratio != False and (unlabelled_ratio != False or unlabelled_ratio != 0):
            output_dir += f"semisupervised/ratioLU-{labelled_ratio}-{unlabelled_ratio}/"
        else:
            output_dir += "supervised/"
        if data_type != None:
            output_dir += f"{data_type}/"
        if percentage_dynamic_labelled != None:
            output_dir += f"percentageDynamicLabelled-{percentage_dynamic_labelled}/"
        if train_val_test_unlabel_split != None:
            output_dir += f"trainValTestUnlabelSplit-{train_val_test_unlabel_split}/"
        if loss_type != None:
            output_dir += f"lossType-{loss_type}"
        if alpha != None:
            output_dir += f"alpha-{alpha}/"
        if num_augmented_videos != 0:
            output_dir += f"numAugmentations-{num_augmented_videos}"
            if dropout_only == True:
                output_dir += f"-dropoutOnly-{dropout_only}/dropoutInt-{dropout_int}"
            elif rotation_only == True:
                output_dir += f"-rotationOnly-{rotation_only}/rotationInt-{rotation_int}/"
            else:
                output_dir += f"/dropoutInt-{dropout_int}-rotationInt-{rotation_int}/"
            
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

def get_dataset(data_dir, kwargs, data_type="A4C", percentage_dynamic_labelled=100, train_val_test_unlabel_split=[0.8, 0.2, 0, 0], augmented_args=[0, False, False, 0, 0]):
    # Set up datasets and dataloaders
    dataset = {}
    pediatric_train, pediatric_val, pediatric_test, pediatric_unlabel = get_pediatric(data_dir, kwargs, data_type, train_val_test_unlabel_split, augmented_args)

    dynamic_train, dynamic_val, dynamic_test, dynamic_unlabel = get_dynamic(data_dir, kwargs, percentage_dynamic_labelled, augmented_args)

    print("Pediatric train: ", len(pediatric_train))
    print("Pediatric val: ", len(pediatric_val))
    print("Pediatric test: ", len(pediatric_test))
    print("Pediatric unlabelled: ", len(pediatric_unlabel))

    print("Dynamic train: ", len(dynamic_train))
    print("Dynamic val: ", len(dynamic_val))
    print("Dynamic test: ", len(dynamic_test))
    print("Dynamic unlabelled: ", len(dynamic_unlabel))

    dataset["train"] = concat_dataset(pediatric_train, dynamic_train)

    dataset["val"] = concat_dataset(pediatric_val, dynamic_val)

    dataset["test"] = concat_dataset(pediatric_test, dynamic_test)
    
    dataset["unlabelled"] = concat_dataset(pediatric_unlabel, dynamic_unlabel)
    
    print("Total train: ", len(dataset["train"]))
    print("Total val: ", len(dataset["val"]))
    print("Total test: ", len(dataset["test"]))
    print("Total unlabelled: ", len(dataset["unlabelled"]))
    
    return dataset

def concat_dataset(pediatric, dynamic):
    if pediatric is not None and dynamic is not None:
        return torch.utils.data.ConcatDataset([pediatric, dynamic])
    elif pediatric is not None:
        return pediatric
    elif dynamic is not None:
        return dynamic
    else:
        return pd.DataFrame()
    
def get_pediatric(data_dir, kwargs, data_type, train_val_test_unlabel_split, augmented_args=[0, False, False, 0, 0]):
    num_augmented_videos = augmented_args[0]
    dropout_only = augmented_args[1]
    rotation_only = augmented_args[2]
    dropout_int = augmented_args[3]
    rotation_int = augmented_args[4]
    if train_val_test_unlabel_split[0] <= 0 or train_val_test_unlabel_split[0] > 1:
        pediatric_train = None
    else:
        pediatric_train = efpredict.datasets.EchoPediatric(root=data_dir, split="train", data_type=data_type, 
            tvtu_split=train_val_test_unlabel_split, num_augmented_videos=num_augmented_videos, 
            dropout_only=dropout_only, rotation_only=rotation_only, dropout_int=dropout_int, 
            rotation_int=rotation_int, **kwargs)
    if train_val_test_unlabel_split[1] <= 0 or train_val_test_unlabel_split[1] > 1:
        pediatric_val = None
    else:
        pediatric_val = efpredict.datasets.EchoPediatric(root=data_dir, split="val", data_type=data_type, 
            tvtu_split=train_val_test_unlabel_split, num_augmented_videos=num_augmented_videos, 
            dropout_only=dropout_only, rotation_only=rotation_only, dropout_int=dropout_int, 
            rotation_int=rotation_int, **kwargs)
    if train_val_test_unlabel_split[2] <= 0 or train_val_test_unlabel_split[2] > 1:
        pediatric_test = None
    else:
        pediatric_test = efpredict.datasets.EchoPediatric(root=data_dir, split="test", data_type=data_type, 
            tvtu_split=train_val_test_unlabel_split, num_augmented_videos=num_augmented_videos, 
            dropout_only=dropout_only, rotation_only=rotation_only, dropout_int=dropout_int, 
            rotation_int=rotation_int, **kwargs)    
    if train_val_test_unlabel_split[3] <= 0 or train_val_test_unlabel_split[3] > 1:
        pediatric_unlabel = None
    else:
        pediatric_unlabel = efpredict.datasets.EchoPediatric(root=data_dir, split="unlabel", data_type=data_type, 
            tvtu_split=train_val_test_unlabel_split, num_augmented_videos=num_augmented_videos, 
            dropout_only=dropout_only, rotation_only=rotation_only, dropout_int=dropout_int, 
            rotation_int=rotation_int, **kwargs)    
    return pediatric_train, pediatric_val, pediatric_test, pediatric_unlabel

def get_dynamic(data_dir, kwargs, percentage_dynamic_labelled, augmented_args=[0, False, False, 0, 0]):
    num_augmented_videos = augmented_args[0]
    dropout_only = augmented_args[1]
    rotation_only = augmented_args[2]
    dropout_int = augmented_args[3]
    rotation_int = augmented_args[4]
    if percentage_dynamic_labelled == 0:
        dynamic_train = None
        dynamic_val = None
        dynamic_test = None
        dynamic_unlabel = None
    else:    
        dynamic_train = efpredict.datasets.EchoDynamic(root=data_dir, split="train", percentage_dynamic_labelled=percentage_dynamic_labelled,
            num_augmented_videos=num_augmented_videos, dropout_only=dropout_only, rotation_only=rotation_only, 
            dropout_int=dropout_int, rotation_int=rotation_int, **kwargs)    
        dynamic_val = efpredict.datasets.EchoDynamic(root=data_dir, split="val", percentage_dynamic_labelled=percentage_dynamic_labelled, 
            num_augmented_videos=num_augmented_videos, dropout_only=dropout_only, rotation_only=rotation_only, 
            dropout_int=dropout_int, rotation_int=rotation_int, **kwargs)    
        dynamic_test = efpredict.datasets.EchoDynamic(root=data_dir, split="test", percentage_dynamic_labelled=100, 
            num_augmented_videos=num_augmented_videos, dropout_only=dropout_only, rotation_only=rotation_only, 
            dropout_int=dropout_int, rotation_int=rotation_int, **kwargs)   
        if percentage_dynamic_labelled != 100:
            dynamic_unlabel = efpredict.datasets.EchoDynamic(root=data_dir, split="unlabel", percentage_dynamic_labelled=percentage_dynamic_labelled, 
            num_augmented_videos=num_augmented_videos, dropout_only=dropout_only, rotation_only=rotation_only, 
            dropout_int=dropout_int, rotation_int=rotation_int, **kwargs)   
        else:
            dynamic_unlabel = None

    return dynamic_train, dynamic_val, dynamic_test, dynamic_unlabel

def get_unlabelled_dataset(data_dir):
    unlabelled_dataset = efpredict.datasets.EchoUnlabelled(root=data_dir)
    return unlabelled_dataset

def test_results(f, output, model, dataset, batch_size, num_workers, device):
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

        # check if ds.datasets exists, if so, set datasets to ds.datasets, if not set datasets to [ds]
        datasets = ds.datasets if hasattr(ds, "datasets") else [ds]

        # Write full performance to file
        with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
            for ds_part in datasets:  # Iterate through the datasets within the ConcatDataset
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

def log_cosh_loss(predicted, target):
    loss = torch.log(torch.cosh(predicted - target))
    return torch.mean(loss)

def get_labelled_loss(loss_type, outputs, outcome):
    if loss_type == "MSE" or loss_type == "PSEUDO":
        loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)
    elif loss_type == "MAE":
        loss = torch.nn.functional.l1_loss(outputs.view(-1), outcome)
    elif loss_type == "HUBER":
        loss = torch.nn.functional.smooth_l1_loss(outputs.view(-1), outcome)
    elif loss_type == "LOGCOSH":
        loss = log_cosh_loss(outputs.view(-1), outcome)
    else:
        raise ValueError("Unknown loss type {}".format(loss_type))
    
    return loss

def get_unlabelled_loss(loss_type, model, unlabelled_X, outputs, alpha=0.1):
    unlabelled_outputs = model(unlabelled_X)

    size_diff = outputs.size(0) - unlabelled_outputs.size(0)   
    # Pad the smaller tensor with zeros
    if size_diff > 0:
        padding = torch.zeros(size_diff, *unlabelled_outputs.size()[1:], device=unlabelled_outputs.device)
        unlabelled_outputs = torch.cat((unlabelled_outputs, padding), dim=0)
    elif size_diff < 0:
        padding = torch.zeros(-size_diff, *outputs.size()[1:], device=outputs.device)
        outputs = torch.cat((outputs, padding), dim=0)
    
    if loss_type == "PSEUDO":
        # Generate pseudo-labels for the unlabelled data
        with torch.no_grad():
            pseudo_labels = model(unlabelled_X).detach()
        # Compute loss using the pseudo-labels
        unlabelled_loss = torch.nn.functional.mse_loss(model(unlabelled_X).view(-1), pseudo_labels.view(-1))
    elif loss_type == "MSE":
        # Compute mse loss between labelled and unlabelled data
        unlabelled_loss = torch.nn.functional.mse_loss(outputs.view(-1), unlabelled_outputs.view(-1))
    elif loss_type == "MAE":
        # Compute mae loss between labelled and unlabelled data
        unlabelled_loss = torch.nn.functional.l1_loss(outputs.view(-1), unlabelled_outputs.view(-1))
    elif loss_type == "HUBER":
        # Compute huber loss between labelled and unlabelled data
        unlabelled_loss = torch.nn.functional.smooth_l1_loss(outputs.view(-1), unlabelled_outputs.view(-1))
    elif loss_type == "LOGCOSH":
        # Compute log-cosh loss between labelled and unlabelled data
        unlabelled_loss = log_cosh_loss(outputs.view(-1), unlabelled_outputs.view(-1))
    else:
        raise ValueError("Invalid loss_type specified")

    unlabelled_loss *= alpha

    return unlabelled_loss