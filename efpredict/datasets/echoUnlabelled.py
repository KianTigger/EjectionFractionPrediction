# echo dataset data organisation from echonet repository 
# - ammended without segmentation
# Kian Kordtomeikel (13/02/2023)
# https://github.com/echonet/dynamic/tree/master/echonet/datasets

"""EchoNet-Unlabelled Dataset."""

import os
import collections
import pandas as pd

import numpy as np
import skimage.draw
import torchvision
import efpredict

class EchoUnlabelled(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,
                 split="train",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 createAllClips=False,
                 pad=None,
                 noise=None,
                 use_phase_clips=True,
                 external_test_location=None):
        if root is None:
            root = efpredict.config.UNLABELLED_DIR

        super().__init__(root)

        self.split = split.upper()
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.createAllClips = createAllClips
        self.pad = pad
        self.noise = noise
        self.use_phase_clips = use_phase_clips
        self.external_test_location = external_test_location

        self.fnames = []
        self.phase_values = {}

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            self.get_unlabelled_data()

    def get_unlabelled_data(self):
        # Load data
        data = pd.read_csv(os.path.join(self.root, "MeasurementsList.csv"))
        # only keep the filenames
        data = np.unique(data["HashedFileName"].values)

        self.check_missing_files()

        data = self.update_batch_paths(data)

        self.fnames = list(data)

    def update_batch_paths(self, data):
        # Find existing batch folders
        batch_folders = []
        batch_num = 1
        while True:
            batch_folder = "Batch{}".format(batch_num)
            batch_path = os.path.join(self.root, batch_folder)

            if not os.path.exists(batch_path):
                break

            batch_folders.append(batch_folder)
            batch_num += 1

        # Initialize updated_data dictionary
        updated_data = {}

        # Iterate through files in batch folders
        for batch_folder in batch_folders:
            batch_path = os.path.join(self.root, batch_folder)
            for fname in os.listdir(batch_path):
                if fname.lower().endswith('.avi'):
                    fname_no_ext = os.path.splitext(fname)[0]

                    if fname_no_ext in data:
                        updated_data[fname_no_ext] = os.path.join(batch_folder, fname)

        data = np.array(list(updated_data.values()))
        return data

    def check_missing_files(self):
        # Check that files are present
        existing_files = []

        batch_num = 1
        while True:
            batch_folder = "Batch{}".format(batch_num)
            batch_path = os.path.join(self.root, "Videos", batch_folder)

            if not os.path.exists(batch_path):
                break
            
            existing_files.extend([os.path.join(batch_folder, f) for f in os.listdir(batch_path)])
            batch_num += 1

        missing = set(self.fnames) - set(existing_files)
        
        if len(missing) != 0:
            print("{} videos could not be found in {}:".format(
                len(missing), os.path.join(self.root, "Videos")))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

    # Modify the __getitem__ method to not gather targets
    def __getitem__(self, index):
        # Get video path
        video_path = self.video_path(index)

        # Load video into np.array
        video = efpredict.utils.loadvideo(video_path).astype(np.float32)

        if video is None:
            return None

        if self.noise is not None:
            # Add simulated noise (black out random pixels)
            video = self.add_random_noise(video)

        # Pad video
        video = self.normalise_video(video)

        length = self.set_length(video)

        # Set number of frames
        video = self.set_frames(video, length)

        video = self.select_clips(video, length)

        return video
    
    def __len__(self):
        return len(self.fnames)
    
    def video_path(self, index):
        # Find filename of video #TODO change from external and clinical test as I don't have them
        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(
                self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(
                self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_path = os.path.join(self.root, list(self.fnames)[index])
            # video_path = os.path.join(self.root, self.fnames[index])

        return video_path
    
    def add_random_noise(self, video):
        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        n = video.shape[1] * video.shape[2] * video.shape[3]
        ind = np.random.choice(n, round(self.noise * n), replace=False)
        f = ind % video.shape[1]
        ind //= video.shape[1]
        i = ind % video.shape[2]
        ind //= video.shape[2]
        j = ind
        video[:, f, i, j] = 0

        return video

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

    def select_clips(self, video, length):
        c, f, h, w = video.shape

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(
                f - (length - 1) * self.period, self.clips)

        # Select clips from video
        new_video = tuple(video[:, s + self.period *
                                np.arange(length), :, :] for s in start)

        if self.clips == 1:
            new_video = new_video[0]
        else:
            new_video = np.stack(new_video)

        return new_video
    

def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
