import os
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import sklearn.metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix

import efpredict

from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights
from torch.utils.data._utils.collate import default_collate

def get_checkpoint(model, optim, scheduler, output, f):
    epoch_resume = 0
    step_resume = 0
    bestLoss = float("inf")
    try:
        # Attempt to load checkpoint
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
        except FileNotFoundError:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
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
                rotation_only=None, dropout_int=None, rotation_int=None,
                scheduler_params=None):
    
    ###
    # StepLR: {
    #   "scheduler_type": "StepLR",
    #   "lr_step_period": 15,
    #   "gamma": 0.1}
    # ExponentialLR:{
    #   "scheduler_type": "ExponentialLR",
    #   "gamma": 0.95}
    # CosineAnnealingLR:{
    #   "scheduler_type": "CosineAnnealingLR",
    #   "T_max": 50,
    #   "eta_min": 1e-5}
    # ReduceLROnPlateau:{
    #   "scheduler_type": "ReduceLROnPlateau",
    #   "factor": 0.1,
    #   "patience": 10,
    #   "min_lr": 1e-5,
    #   "cooldown": 0,
    #   "threshold": 1e-4,
    #   "threshold_mode": "rel",
    #   "verbose": true}
    # python script.py --scheduler_params '{"scheduler_type": "StepLR", "lr_step_period": 15, "gamma": 0.1}'
    ###

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    if scheduler_params is not None:
        scheduler_type = scheduler_params.pop("scheduler_type")
        try:
            lr_step_period = scheduler_params.pop("step_size")
        except KeyError:
            pass
    else:
        scheduler_type = None
    
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
        if scheduler_type != None:
            output_dir += f"-schedulerType-{scheduler_type} -lr-{lr}-lrStepPeriod-{lr_step_period}-weightdecay-{weight_decay}/"
        if scheduler_params != None:
            for key, value in scheduler_params.items():
                output_dir += f"{key}-{value}"
            output_dir += "/"
            
        output = os.path.join(output_dir, f"{model_name}_{frames}_{period}_{pretrained_str}")
        print("Output directory: ", output)

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


    # Set up learning rate scheduler based on scheduler_type
    if scheduler_params is None or scheduler_type is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
    elif scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_step_period, **scheduler_params)
    elif scheduler_type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, **scheduler_params)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **scheduler_params)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **scheduler_params)
    else:
        print("Invalid scheduler type. Using StepLR.")
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

    dataset["train"] = concat_dataset(pediatric_train, dynamic_train)

    dataset["val"] = concat_dataset(pediatric_val, dynamic_val)

    dataset["test"] = concat_dataset(pediatric_test, dynamic_test)
    
    dataset["unlabelled"] = concat_dataset(pediatric_unlabel, dynamic_unlabel)

    dataset["CAMUS_GOOD"] = efpredict.datasets.CAMUS(root=data_dir, split="good", **kwargs) 
    dataset["CAMUS_MEDIUM"] = efpredict.datasets.CAMUS(root=data_dir, split="medium", **kwargs) 
    dataset["CAMUS_POOR"] = efpredict.datasets.CAMUS(root=data_dir, split="poor", **kwargs) 
    dataset["CAMUS"] = efpredict.datasets.CAMUS(root=data_dir, split="all", **kwargs)

    
    print("Total train: ", len(dataset["train"]))
    print("Total val: ", len(dataset["val"]))
    print("Total test: ", len(dataset["test"]))
    print("Total unlabelled: ", len(dataset["unlabelled"]))

    print("Total CAMUS_GOOD: ", len(dataset["CAMUS_GOOD"]))
    print("Total CAMUS_MEDIUM: ", len(dataset["CAMUS_MEDIUM"]))
    print("Total CAMUS_POOR: ", len(dataset["CAMUS_POOR"]))
    print("Total CAMUS: ", len(dataset["CAMUS"]))
    
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

    #print number of parameters
    try:
        parameters = count_parameters(model)
        f.write("Number of parameters: {}\n".format(parameters))
        print("Number of parameters: {}\n".format(parameters))
    except:
        pass

    for split in ["CAMUS_GOOD", "CAMUS_MEDIUM", "CAMUS_POOR", "CAMUS", "val", "test"]:
        # Performance without test-time augmentation
        try:
            ds = dataset[split]
        except KeyError:
            continue
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

        print("Results from dataset: {}".format(split))

        print("Mean: {:.2f}".format(mean))
        print("Std: {:.2f}".format(std))

        #TODO check that y is the real EF
        #Print the MAE
        print("MAE: {:.2f}".format(sklearn.metrics.mean_absolute_error(y, yhat)))
        #Print the RMSE
        print("RMSE: {:.2f}".format(math.sqrt(sklearn.metrics.mean_squared_error(y, yhat))))
        #Print the R2
        r2 = sklearn.metrics.r2_score(y, yhat)
        print("R2: {:.3f}".format(r2))

        plot_results(y, yhat, split, output, r2)

def plot_results(y, yhat, split, output, r2=False): 
    print("Plotting results") 

    # Plot actual and predicted EF
    fig = plt.figure(figsize=(3, 3))
    lower = min(y.min(), yhat.min())
    upper = max(y.max(), yhat.max())
    title = "EF Predictions"
    if split == "val":
        title += " (validation)"
    plt.title(title)
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

    # Add R-squared value to the bottom right of the graph
    if r2:
        # plt.text(upper - 8, lower - 1, f'R² = {r2:.2f}', fontsize=9, ha='right')
        # instead of plt.text, use plt.textbbox
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(upper - 1, lower + 1, f'R² = {r2:.2f}', fontsize=9, ha='right', bbox=props)


    plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)), dpi=300)
    plt.close(fig)

    # Plot AUROC
    fig = plt.figure(figsize=(3, 3))
    plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")

    title = "Thresholded AUROC for EF Predictions"
    plt.title(title)

    colors = ["b", "g", "r", "c"]  # Define a list of colors for the ROC curves
    thresholds = [35, 40, 45, 50]

    for i, thresh in enumerate(thresholds):
        current_auc = sklearn.metrics.roc_auc_score(y > thresh, yhat)
        fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
        plt.plot(fpr, tpr, color=colors[i], label=f'Threshold {thresh}, AUC = {current_auc:.3f}')

    plt.axis([-0.01, 1.01, -0.01, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)), dpi=300)
    plt.close(fig)

    # Plot binary AUC for over/under 40
    fig = plt.figure(figsize=(3, 3))
    plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")

    binary_y = (y >= 40).astype(int)
    binary_yhat = (yhat >= 40).astype(int)
    fpr, tpr, _ = sklearn.metrics.roc_curve(binary_y, binary_yhat)
    auc = sklearn.metrics.roc_auc_score(binary_y, binary_yhat)

    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output, "{}_binary_roc.pdf".format(split)), dpi=300)
    plt.close(fig)

    # Create and save accuracy ranges plot
    fig = plot_accuracy_ranges(y, yhat)
    plt.savefig(os.path.join(output, f"{split}_accuracy_ranges.pdf"), dpi=300)
    plt.close(fig)

    # Create and save confusion matrix
    fig = plot_confusion_matrix(y, yhat)
    plt.savefig(os.path.join(output, f"{split}_confusion_matrix.pdf"), dpi=300)
    plt.close(fig)

def plot_accuracy_ranges(y, yhat):
    ranges = [(0, 40), (40, 50), (50, np.inf)]
    range_labels = ['0-40', '40-50', '50+']
    accuracies = []

    for r in ranges:
        mask = (y >= r[0]) & (y < r[1])
        y_range = y[mask]
        yhat_range = yhat[mask]
        accuracy = np.mean((y_range >= 40) == (yhat_range >= 40))
        accuracies.append(accuracy)

    fig = plt.figure(figsize=(6, 4))
    bar_width = 0.4
    index = np.arange(len(ranges))
    plt.bar(index, accuracies, bar_width, color='b', alpha=0.7)
    plt.xlabel("Actual EF Ranges (%)")
    plt.ylabel("Accuracy")
    plt.xticks(index, range_labels)
    plt.ylim([0.5, 1])
    plt.title("Model Accuracy for Different EF Ranges")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y, yhat):
    ranges = [(0, 40), (40, 50), (50, np.inf)]
    range_labels = ['HFrEF', 'HFmrEF', 'HFpEF']

    # Categorize y and yhat based on the specified ranges
    y_categorized = np.digitize(y, [r[1] for r in ranges[:-1]]) - 1
    yhat_categorized = np.digitize(yhat, [r[1] for r in ranges[:-1]]) - 1

    # Calculate confusion matrix
    cm = confusion_matrix(y_categorized, yhat_categorized)

    # Normalize confusion matrix to show percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix using seaborn
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap="Blues", ax=ax)

    # set cbar to be on the right side of the plot, 0 - 100%
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%'])

    ax.set(xticks=np.arange(len(range_labels)) + 0.5, yticks=np.arange(len(range_labels)) + 0.5,
           xticklabels=range_labels, yticklabels=range_labels,
           xlabel="Predicted EF Ranges", ylabel="Actual EF Ranges",
           title="Confusion Matrix for Different EF Ranges")
    ax.set_xticklabels(ax.get_xticklabels(), ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), va='center')
    fig.tight_layout()
    return fig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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