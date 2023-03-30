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

    output, device, model, optim, scheduler = helpFuncs.setup_model(seed, model_name, pretrained, device, weights, frames, period, output, weight_decay, lr, lr_step_period, num_epochs, labelled_ratio, unlabelled_ratio)

    kwargs = helpFuncs.mean_and_std(data_dir, task, frames, period)

    dataset = helpFuncs.get_dataset(data_dir, num_train_patients, kwargs)



    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:

        model, optim, scheduler, epoch_resume, bestLoss = helpFuncs.get_checkpoint(model, optim, scheduler, output, f)

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                tempTime = time.time()
                print("Creating labelled and unlabelled batch sizes")
                labelled_batch_size = max(1, int(batch_size * labelled_ratio / (labelled_ratio + unlabelled_ratio)))
                unlabelled_batch_size = batch_size - labelled_batch_size
                print("created labelled and unlabelled batch sizes in {} seconds".format(time.time() - tempTime))

                tempTime = time.time()
                print("Creating dataloader")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=labelled_batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                print("Created dataloader in {} seconds".format(time.time() - tempTime))

                unlabelled_dataloader = None
                if phase == "train" and unlabelled_batch_size > 0:
                    tempTime = time.time()
                    print("Creating unlabelled dataloader")
                    unlabelled_dataloader = torch.utils.data.DataLoader(
                        dataset["unlabelled"], batch_size=unlabelled_batch_size, num_workers=num_workers, shuffle=True, 
                        pin_memory=(device.type == "cuda"), drop_last=True,  collate_fn=helpFuncs.custom_collate)
                    print("Created unlabelled dataloader in {} seconds".format(time.time() - tempTime))

                print("Running epoch")
                loss, yhat, y = efpredict.utils.EFPredict.run_epoch(model, dataloader, phase == "train", optim, device, unlabelled_dataloader=unlabelled_dataloader)
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

            scheduler.step()
            
            bestLoss = helpFuncs.save_checkpoint(model, period, frames, epoch, output, loss, bestLoss, y, yhat, optim, scheduler)

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

        if run_test:
            test_resuls(f, output, model, data_dir, batch_size, num_workers, device, **kwargs)  


def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None, unlabelled_dataloader=None):
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

    if unlabelled_dataloader is not None:
        tempTime = time.time()
        print("Creating unlabelled iterator")
        unlabelled_iterator = iter(unlabelled_dataloader)
        print("Created unlabelled iterator in {} seconds".format(time.time() - tempTime))

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:
                print("loaded data")

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

                tempTime = time.time()
                print("Calculating loss")
                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)
                print("Calculated loss in {} seconds".format(time.time() - tempTime))

                if train and (unlabelled_dataloader is not None):
                    tempTime = time.time()
                    print("Getting unlabelled data")
                    # Sample a batch from the unlabelled dataset
                    try:
                        unlabelled_X = next(unlabelled_iterator)
                    except StopIteration:
                        unlabelled_iterator = iter(unlabelled_dataloader)
                        unlabelled_X = next(unlabelled_iterator)
                    print("Got unlabelled data in {} seconds".format(time.time() - tempTime))

                    # Check whether unlabelled_X is valid, if not, skip consistency loss
                    while not (len(unlabelled_X) > 0 and isinstance(unlabelled_X[0], torch.Tensor) and unlabelled_X[0].shape[0] != 0 and unlabelled_X is not None):
                        #TODO see if this is needed
                        print("Checking unlabelled data")
                        try:
                            print("trying to get next unlabelled data")
                            unlabelled_X = next(unlabelled_iterator)
                        except StopIteration:
                            print("restarting unlabelled iterator!!!!!!!!!!!!!!")
                            unlabelled_iterator = iter(unlabelled_dataloader)
                            unlabelled_X = next(unlabelled_iterator)

                    if len(unlabelled_X) > 0 and isinstance(unlabelled_X[0], torch.Tensor) and unlabelled_X[0].shape[0] != 0 and unlabelled_X is not None:
                        print("Got unlabelled data")
                        unlabelled_X = unlabelled_X.to(device)
                        tempTime = time.time()
                        print("Computing consistency loss size diff")
                        # Compute consistency loss between labelled and unlabelled data
                        unlabelled_outputs = model(unlabelled_X)
                        size_diff = outputs.size(0) - unlabelled_outputs.size(0)
                        print("Computed consistency loss size diff in {} seconds".format(time.time() - tempTime))

                        tempTime = time.time()
                        print("Padding tensors")
                        # Pad the smaller tensor with zeros
                        if size_diff > 0:
                            padding = torch.zeros(size_diff, *unlabelled_outputs.size()[1:], device=unlabelled_outputs.device)
                            unlabelled_outputs = torch.cat((unlabelled_outputs, padding), dim=0)
                        elif size_diff < 0:
                            padding = torch.zeros(-size_diff, *outputs.size()[1:], device=outputs.device)
                            outputs = torch.cat((outputs, padding), dim=0)
                        print("Padded tensors in {} seconds".format(time.time() - tempTime))
                        
                        tempTime = time.time()
                        print("computing consistency loss")
                        consistency_loss = torch.nn.functional.mse_loss(outputs.view(-1), unlabelled_outputs.view(-1))
                        print("Computed consistency loss in {} seconds".format(time.time() - tempTime))

                        # Add consistency loss to the original loss
                        loss += consistency_loss

                if train:
                    tempTime = time.time()
                    print("Backpropagating")
                    optim.zero_grad()
                    print("Backpropagated in {} seconds".format(time.time() - tempTime))

                    tempTime = time.time()
                    print("Computing gradients")
                    loss.backward()
                    print("Computed gradients in {} seconds".format(time.time() - tempTime))

                    tempTime = time.time()
                    print("Optimizing")
                    optim.step()
                    print("Optimized in {} seconds".format(time.time() - tempTime)) 

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y

def test_resuls(f, output, model, data_dir, batch_size, num_workers, device, **kwargs):
    for split in ["val", "test"]:
        # Performance without test-time augmentation
        ds = efpredict.datasets.EchoDynamic(root=data_dir, split=split, **kwargs)
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
        loss, yhat, y = efpredict.utils.EFPredict.run_epoch(model, dataloader, False, None, device)
        f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *efpredict.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
        f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *efpredict.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
        f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, efpredict.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
        f.flush()

        # # Performance with test-time augmentation
        # ds = efpredict.datasets.EchoDynamic(root=data_dir, split=split, **kwargs, clips="all")
        # dataloader = torch.utils.data.DataLoader(
        #     ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
        # loss, yhat, y = efpredict.utils.EFPredict.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=batch_size)
        # f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *efpredict.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
        # f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *efpredict.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
        # f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, efpredict.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
        # f.flush()

        # Write full performance to file
        with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
            for (filename, pred) in zip(ds.fnames, yhat):
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