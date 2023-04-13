# efpredict package init file
# Kian Kordtomeikel (13/02/2023)
# https://github.com/echonet/dynamic/tree/master/echonet/

"""
The efpredict package contains code for loading echocardiogram videos, and
functions for training and testing ejection fraction prediction models.
"""

import click

from efpredict.__version__ import __version__
from efpredict.config import CONFIG as config
import efpredict.datasets as datasets
import efpredict.utils as utils

import warnings
from torch.utils.data import dataloader

warnings.filterwarnings("ignore", category=UserWarning, module=dataloader.__name__)


@click.group()
def main():
    """Entry point for command line interface."""


del click


main.add_command(utils.EFPredict.run)
main.add_command(utils.EFPredictSupervised.run)

__all__ = ["__version__", "config", "datasets", "main", "utils"]
