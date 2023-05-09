"""CAMUS Dataset."""

import os
import collections
import pandas as pd

import numpy as np
import torchvision
import efpredict
import efpredict.datasets.datasetHelpFuncs as helpFuncs

class CAMUS(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,                 
                 split="test", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 percentage_dynamic_labelled=100,
                 num_augmented_videos=0,
                 dropout_only=False, rotation_only=False,
                 dropout_int=None, rotation_int=None,
                 clips=1,
                 createAllClips=False,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 use_phase_clips=False,
                 external_test_location=None):
        if root is None:
            root = efpredict.config.CAMUS_DIR
        
        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.percentage_dynamic_labelled = percentage_dynamic_labelled
        self.num_augmented_videos = num_augmented_videos
        self.dropout_only = dropout_only
        self.rotation_only = rotation_only
        self.dropout_int = dropout_int
        self.rotation_int = rotation_int
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.createAllClips = createAllClips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.use_phase_clips = use_phase_clips
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []
        self.phase_values = {}

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            self.get_EF_Labels()
    
    def get_EF_Labels(self, filename="Info_4CH.cfg"):
        # for each filename file in self.root, get the following values
        # ED, ES, NbFrame, Sex, Age, ImageQuality, EF, FrameRate
        # return a dictionary of the values
        # if the file is not found, return None
        
        for folder in os.listdir(self.root):
            # load the file
            try:
                cur_file = open(os.path.join(self.root, folder, filename), "r")
            except FileNotFoundError:
                print("File not found: {}".format(os.path.join(self.root, folder, filename)))
                continue
            # read the file
            lines = cur_file.readlines()
            # close the file
            cur_file.close()
            # get the values
            values = {}
            for line in lines:
                if line[0] == "#":
                    continue
                line = line.split(":")
                values[line[0]] = line[1].strip()
            # add the values to the dictionary
            self.phase_values[folder] = values
        
        #create self.fnames and self.outcome
        for folder in self.phase_values:
            if self.phase_values[folder]["EF"] == "NA":
                continue
            self.fnames.append(folder)
            self.outcome.append(float(self.phase_values[folder]["EF"]))

        # Assume avi if no suffix
        self.fnames = [
            fn if os.path.splitext(fn)[1] == ".avi" else fn + ".avi" for fn in self.fnames
            ]
    
    def __getitem__(self, index):

        fname = self.fnames[index]
        fnamewithoutsuffix = os.path.splitext(fname)[0]
        
        # Get video path
        video_path = os.path.join(self.root, fnamewithoutsuffix, 'video_' + self.fnames[index])
        # Get video

        video = efpredict.utils.loadvideo(video_path).astype(np.float32)

        # Pad video
        video = self.normalise_video(video)

        length = self.set_length(video)

        # Set number of frames
        # TODO see if this is needed
        video = self.set_frames(video, length)

        # Get target
        target = self.gather_targets(index)

        if self.pad is not None:
            video = self.pad_video(video)

        return video, target
    
    def gather_targets(self, index):
        # Gather targets
        target = []
        for t in self.target_type:
            if t == "Filename":
                target.append(self.fnames[index])
            # old code had more options for targets, see dynamic repo.
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(
                        self.outcome[index]["EF"]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        return target

    def __len__(self):
        return len(self.fnames)
    
    def normalise_video(self, video):

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        return video
    
    def set_frames(self, video, length):
        # Set number of frames
        c, f, h, w = video.shape

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            new_video = np.concatenate(
                (video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            # c, f, h, w = video.shape  # pylint: disable=E0633
        else:
            new_video = video

        return new_video
    
    def pad_video(self, video):
        # Add padding of zeros (mean color of videos)
        # Crop of original size is taken out
        # (Used as augmentation)
        c, l, h, w = video.shape
        temp = np.zeros((c, l, h + 2 * self.pad, w +
                        2 * self.pad), dtype=video.dtype)
        temp[:, :, self.pad:-self.pad, self.pad:-
             self.pad] = video  # pylint: disable=E1130
        i, j = np.random.randint(0, 2 * self.pad, 2)
        new_video = temp[:, :, i:(i + h), j:(j + w)]

        return new_video
    
    def set_length(self, video):
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        return length
    
def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)