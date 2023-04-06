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
import torch.nn as nn


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
        if epoch_resume == 0:
            epoch_resume = 1
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
                if phase == "train" and unlabelled_batch_size > 0:
                    unlabelled_dataloader = DataLoader(
                        unlabelled_dataset, batch_size=unlabelled_batch_size, num_workers=num_workers, shuffle=True, 
                        pin_memory=(device.type == "cuda"), drop_last=True,  collate_fn=helpFuncs.custom_collate)
                
                criterion = nn.CrossEntropyLoss()

                # loss, yhat, y = efpredict.utils.EFPredict.run_epoch(model, labelled_dataloader, phase == "train", optim, device, unlabelled_dataloader=unlabelled_dataloader)
                loss, yhat, y = efpredict.utils.EFPredict.run_epoch(model, labelled_dataloader, unlabelled_dataloader, criterion, optim, device, pseudo_labeling_percentage=0.5)
                
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



def run_epoch(model, labelled_dataloader, unlabelled_dataloader, criterion, optimizer, device, pseudo_labeling_percentage=0.5):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    
    unlabelled_iterator = iter(unlabelled_dataloader)
    all_predictions = []


    for i, (inputs, labels) in enumerate(labelled_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Pseudo-labeling for unlabelled data
        try:
            unlabelled_inputs = next(unlabelled_iterator)
        except StopIteration:
            unlabelled_iterator = iter(unlabelled_dataloader)
            unlabelled_inputs = next(unlabelled_iterator)

        unlabelled_inputs = unlabelled_inputs.to(device)

        with torch.no_grad():
            model.eval()
            outputs = model(unlabelled_inputs)
            pseudo_labels = torch.argmax(outputs, dim=1)

        model.train()

        n_pseudo_labelled = int(pseudo_labeling_percentage * unlabelled_inputs.size(0))
        combined_inputs = torch.cat((inputs, unlabelled_inputs[:n_pseudo_labelled]), dim=0)
        combined_labels = torch.cat((labels, pseudo_labels[:n_pseudo_labelled]), dim=0)

        optimizer.zero_grad()
        outputs = model(combined_inputs)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == combined_labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(labelled_dataloader)
    epoch_acc = correct_predictions / (len(labelled_dataloader) * (1 + pseudo_labeling_percentage) * labelled_dataloader.batch_size)

    yhat = np.array(all_predictions)
    return epoch_loss, yhat, epoch_acc

def test_resuls(f, output, model, data_dir, batch_size, num_workers, device, **kwargs):
    for split in ["val", "test"]:
        # Performance without test-time augmentation
        ds = efpredict.datasets.EchoDynamic(root=data_dir, split=split, **kwargs)
        dataloader = DataLoader(
            ds,
            batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
        loss, yhat, y = efpredict.utils.EFPredict.run_epoch(model, dataloader, False, None, device)
        f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *efpredict.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
        f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *efpredict.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
        f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, efpredict.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
        f.flush()

        # # Performance with test-time augmentation
        # ds = efpredict.datasets.EchoDynamic(root=data_dir, split=split, **kwargs, clips="all")
        # dataloader = DataLoader(
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
        print("MAE: {:.2f}".format(sklearn.metrics.mean_absolute_error(y, yhat)))
        print("RMSE: {:.2f}".format(math.sqrt(sklearn.metrics.mean_squared_error(y, yhat))))
        print("R2: {:.2f}".format(sklearn.metrics.r2_score(y, yhat)))

        # plot_results(y, yhat, split, output)