# echo dataset data organisation from echonet repository
# - ammended without segmentation
# Kian Kordtomeikel (13/02/2023)
# https://github.com/echonet/dynamic/tree/master/echonet/datasets

# TODO, ammend this without segmentation/using boolean remove options.

"""EchoNet-Pediatric Dataset."""

import os
import collections
import random
import pandas as pd

import numpy as np
import skimage.draw
import torchvision
import efpredict
from sklearn.model_selection import train_test_split


class EchoPediatric(torchvision.datasets.VisionDataset):
    """EchoNet-Pediatric Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `efpredict.config.DATA_DIR`)
        target_type (list or string, optional): Type of targets to use. Defaults to ["EF"].
            Can be a list to output a tuple with all specified target types.
            The targets represent:
                - "FileName" (string): filename of video
                - "EF" (float): ejection fraction
                - "Sex" (string): sex of the patient
                - "Age" (float): age of the patient
                - "Weight" (float): weight of the patient
                - "Height" (float): height of the patient
                - "Split" (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        mean (float, int or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (float, int or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken).
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
        """
    
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 data_type="A4C",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 tvtu_split=[0.7, 0.15, 0.15, 0], #train, val, test, unlabelled
                 num_augmented_videos=0,
                 createAllClips=False,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 use_phase_clips=True,
                 external_test_location=None):
        if root is None:
            root = efpredict.config.PEDIATRIC_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.data_type = data_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.tvtu_split = tvtu_split
        self.num_augmented_videos = num_augmented_videos
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
            self.get_labels()

            # Original code, volume tracing is not used in this project.
            # See echonet dynamic repository for loading volume tracing.

    def get_labels(self):

        self.createDataFrames()

        if self.use_phase_clips:
            self.create_EF_Labels()

        else:
            self.get_EF_Labels()

        self.check_missing_files()

    def create_EF_Labels(self, filename="FileList.csv", outputfilename="PediatricFileListPhaseClips.csv"):

        self.get_phase_labels()

        data = self.dataset

        self.fnames = data["FileName"].tolist()

        for dataset, datatype in [[self.dfa4c, 'A4C'], [self.dfpsax, 'PSAX']]:

            # Create a new csv file with the new file names, overwrite if it already exists
            with open(os.path.join(self.root, datatype, outputfilename), 'w', newline='') as f:
                headers = dataset.columns.values.tolist() + ['StartFrame', 'EndFrame']
                f.write(','.join(headers) + '\n')

                for row in dataset.iterrows():
                    name = row[1]["FileName"]
                    try:
                        ED_Predictions = self.phase_values[name][0]
                        ES_Predictions = self.phase_values[name][1]
                    except KeyError:
                        # print(f"Warning: {name} has no ED or ES predictions!")
                        ED_Predictions = [0]
                        ES_Predictions = [self.length]
                        self.phase_values[name] = [ED_Predictions, ES_Predictions]
                        
                    written_line = False
                    if not ED_Predictions or not ES_Predictions:
                        # TODO fix this, by generating the phase labels
                        # print(f"Warning: {name} has no ED or ES predictions!")
                        f.write(
                            f"{','.join(map(str, row[1].values.tolist()))},0,0,\n")
                    else:
                        for j in range(min(len(ED_Predictions), len(ES_Predictions))):
                            # ED_Prediciton must be less than ES_Prediciton
                            if ED_Predictions[j] > ES_Predictions[j]:
                                continue

                            # abs(ED_Prediciton - ES_Prediciton) must be greater than 1/4 of the fps
                            # if abs(ED_Predictions[j] - ES_Predictions[j]) < 0.25 * row[1]["FPS"]:
                            #     continue
                            # new_name = f"{name}_phase_{j}"
                            f.write(
                                f"{','.join(map(str, row[1].values.tolist()))},{ED_Predictions[j]},{ES_Predictions[j]},\n")
                            written_line = True

                    if not written_line:
                        # TODO fix this, by generating the phase labels
                        # print(f"Warning: {name} has no ED or ES predictions!")
                        f.write(
                            f"{','.join(map(str, row[1].values.tolist()))},0,0,\n")

        self.get_EF_Labels(filename=outputfilename)

    def createDataFrames(self, filename="FileList.csv"):
        with open(os.path.join(self.root, 'A4C', filename)) as f:
            self.dfa4c = pd.read_csv(f, index_col=False)
        
        with open(os.path.join(self.root, 'PSAX', filename)) as f:
            self.dfpsax = pd.read_csv(f, index_col=False)

        if self.data_type == "A4C":
            self.dataset = self.dfa4c
        elif self.data_type == "PSAX":
            self.dataset = self.dfpsax
        else:
            self.dataset = pd.concat([self.dfa4c, self.dfpsax]).reset_index(drop=True)

    def get_EF_Labels(self, filename="FileList.csv"):

        data = self.dataset

        if self.split != "ALL":
            if self.split == "TRAIN":
                split_to_use = self.tvtu_split[0]
            elif self.split == "VAL":
                split_to_use = self.tvtu_split[1]
            elif self.split == "TEST":
                split_to_use = self.tvtu_split[2]
            elif self.split == "UNLABEL":
                split_to_use = self.tvtu_split[3]
            else:
                raise ValueError("Invalid split")
            if split_to_use == 0.0 or split_to_use < 0.0 or split_to_use > 1.0:
                return None
            elif split_to_use < 1:
                data = train_test_split(data, test_size=1 - split_to_use, random_state=42)[0]

        self.header = data.columns.tolist()

        self.fnames = data["FileName"].tolist()

        # Assume avi if no suffix
        self.fnames = [
            fn if os.path.splitext(fn)[1] == ".avi" else fn + ".avi" for fn in self.fnames
            ]
        
        self.outcome = data.values.tolist()

    def get_phase_labels(self, filename="PhasesList.csv", predictionsFileName="PhasesPredictionsList.csv"):
        data = None

        try:
            with open(os.path.join(self.root, filename)) as f:
                data = pd.read_csv(f)

        except FileNotFoundError:
            try:
                with open(os.path.join(self.root, predictionsFileName)) as f:
                    data = pd.read_csv(f)

            except FileNotFoundError:
                print("No phase information found. TODO Will generate phase information.")
                # TODO, generate phase information from here

        if data is not None:
            missing_values = []
            for _, row in data.iterrows():
                if self.createAllClips:
                    number_of_frames = row["NumberOfFrames"]
                    length = self.length
                    # TODO, change this to be a list of tuples
                    self.phase_values[filename] = [list(range(
                        0, number_of_frames - length + 1)), list(range(length, number_of_frames + 1))]
                else:
                    filename = row[0]
                    ED_Predictions = pd.eval(row[1])
                    ES_Predictions = pd.eval(row[2])
                    if len(ED_Predictions) == 0 or len(ES_Predictions) == 0:
                        missing_values.append(filename)
                    self.phase_values[filename] = [
                        ED_Predictions, ES_Predictions]

            if len(missing_values) > 0:
                print("Missing phase information for {} videos.".format(
                    len(missing_values)))
                print("TODO TODO TODO Will generate phase information for these videos.")
                self.generate_phase_predictions(missing_values)
                # TODO, generate phase information from here

            self.check_missing_files()

        else:
            self.generate_phase_predictions()

    def generate_phase_predictions(self, filenames=None):
        # TODO generate phase information
        pass

    def check_missing_files(self):
        # Check that files are present in A4C and PSAX directories
        missing_a4c = set(self.dfa4c['FileName']) - set(os.listdir(os.path.join(self.root, "A4C", "Videos")))
        missing_psax = set(self.dfpsax['FileName']) - set(os.listdir(os.path.join(self.root, "PSAX", "Videos")))
        missing = missing_a4c.union(missing_psax)
        
        if len(missing) != 0:
            print("{} videos could not be found in {} or {}:".format(
                len(missing), os.path.join(self.root, "A4C", "Videos"), os.path.join(self.root, "PSAX", "Videos")))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, sorted(missing)[0]))

    def __getitem__(self, index):
        # Get video path
        video_path = self.video_path(index)

        # Load video into np.array
        video = efpredict.utils.loadvideo(video_path).astype(np.float32)

        if self.noise is not None:
            # Add simulated noise (black out random pixels)
            video = self.add_random_noise(video)

        # Pad video
        video = self.normalise_video(video)

        length = self.set_length(video)

        # Set number of frames
        # TODO see if this is needed
        video = self.set_frames(video, length)

        video = self.select_clips_phase(video, length, index)
        # video = self.select_clips(video, length)

        target = self.gather_targets(index)

        if self.pad is not None:
            video = self.pad_video(video)

        # Create rotated videos
        videos = [video]
        rotation_angles = random.sample([90, 180, 270], self.num_augmented_videos)
        for rotation_angle in rotation_angles:
            rotated_video = np.rot90(video, k=rotation_angle // 90, axes=(2, 3)).copy()
            videos.append(rotated_video)

        return videos, target

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

    def select_clips_phase(self, video, length, index):
        c, f, h, w = video.shape

        key = self.fnames[index]
        # get phase information
        phases = self.phase_values[key]  # remove .avi

        ED_Predictions = phases[0]
        ES_Predictions = phases[1]

        # clip_length = self.period * length
        clip_length = length

        # for each phase, generate a clip from the video and add to the list
        new_video = ()

        for i in range(min(len(ED_Predictions), len(ES_Predictions))):
            start = ED_Predictions[i]
            end = ES_Predictions[i]

            # clip = video, from start to end, regardless of length
            clip = video[:, start:end, :, :]

            # if clip is too short, pad with zeros
            # TODO Make it so it uses either interpolation, or repeating frames throughout the clip
            if clip.shape[1] < clip_length:
                clip = np.concatenate(
                    (clip, np.zeros((c, clip_length - clip.shape[1], h, w), clip.dtype)), axis=1)

            # if clip is too long, take every nth frame so that it is the correct length
            if clip.shape[1] > clip_length:
                # Compute the downsampling factor
                factor = clip.shape[1] // clip_length

                # Compute the number of frames to keep after downsampling
                # num_frames = factor * clip_length

                # TODO using num_frames sometimes results in a clip that is too long, fix or just use clip_length
                # Downsample the clip and keep only the first num_frames frames
                clip = clip[:, ::factor, :, :][:, :clip_length, :, :]

            new_video = new_video + (clip,)

        # if there are no clips, print a warning with the video name, and return 1 clip
        if len(new_video) == 0:
            print("Warning: No clips found for video {}".format(
                self.fnames[index]))
            # return first clip_length frames of video
            return video[:, :clip_length, :, :]
        else:
            if self.clips == 1:
                new_video = new_video[0]
            else:
                new_video = np.stack(new_video)

        return new_video

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

    def video_path(self, index):
        # Find filename of video #TODO change from external and clinical test as I don't have them
        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(
                self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(
                self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            if self.dataset['FileName'].iloc[index] in os.listdir(os.path.join(self.root, "A4C", "Videos")):
                video_path = os.path.join(
                    self.root, "A4C", "Videos", self.dataset['FileName'].iloc[index])
            else:
                video_path = os.path.join(
                    self.root, "PSAX", "Videos", self.dataset['FileName'].iloc[index])
        
        return video_path

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
                        self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        return target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
