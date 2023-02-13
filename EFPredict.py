import os
import numpy as np
import math

import click
import torch
import torchvision

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

def run(
    data_dir=None,
    output=None,
    task="EF",

    model_name="r2plus1d_18",
    pretrained=True,
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
):
    # TODO Write docstrings, and explanations for args

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Set up model
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)

    # Replaced the last layer with a linear layer with 1 output
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    model.fc.bias.data[0] = 55.6 # TODO, echonet repository uses 55.6, but may need to be changed.

    # If using GPU, wrap model in DataParallel
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None: #TODO Check if this is correct - from echonet repository
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])
    
     # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    # TODO replace echonet with echonet and lvh, but just echonet for now.
    # mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    # kwargs = {"target_type": task,
    #           "mean": mean,
    #           "std": std,
    #           "length": frames,
    #           "period": period,
    #           }

    # Set up datasets and dataloaders
    dataset = {}
    # TODO again replace echonet with own file/functions.
    # dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    # if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
    #     # Subsample patients (used for ablation experiment)
    #     indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
    #     dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    # dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)




    return 1