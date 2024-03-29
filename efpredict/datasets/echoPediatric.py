# echo dataset data organisation from echonet repository
# - ammended for echonet pediatric dataset
# Kian Kordtomeikel (13/02/2023)
# https://github.com/echonet/dynamic/tree/master/echonet/datasets

# TODO, ammend this without segmentation/using boolean remove options.

"""EchoNet-Pediatric Dataset."""

import os
import collections
import random
import pandas as pd

import numpy as np
import torchvision
import efpredict
from sklearn.model_selection import train_test_split
import efpredict.datasets.datasetHelpFuncs as helpFuncs


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
                 tvtu_split=[0.8, 0.2, 0, 0], #train, val, test, unlabelled
                 num_augmented_videos=0,
                 dropout_only=False, rotation_only=False,
                 dropout_int=None, rotation_int=None,
                 createAllClips=False,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 use_phase_clips=False,
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
        self.dropout_only = dropout_only
        self.rotation_only = rotation_only
        self.dropout_int = dropout_int
        self.rotation_int = rotation_int
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

    def create_EF_Labels(self, filename="FileList.csv", outputfilename="FileListPhaseClips.csv"):

        data = self.dataset

        self.fnames = data["FileName"].tolist()

        self.get_phase_labels()

        with open(os.path.join(self.root, outputfilename), 'w', newline='') as f:
            headers = data.columns.values.tolist() + ['StartFrame', 'EndFrame']
            f.write(','.join(headers) + '\n')

            for row in data.iterrows():
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
                    for j in range(min(len(ED_Predictions), len(ES_Predictions), self.clips)):
                        # ED_Prediciton must be less than ES_Prediciton
                        if ED_Predictions[j] > ES_Predictions[j]:
                            continue

                        # abs(ED_Prediciton - ES_Prediciton) must be greater than 1/4 of the fps
                        # need another way to get fps for this dataset
                        # if abs(ED_Predictions[j] - ES_Predictions[j]) < 0.25 * row[1]["FPS"]:
                        if abs(ED_Predictions[j] - ES_Predictions[j]) < 0.25 * 50:       
                            continue
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

        # Load video-level labels
        with open(os.path.join(self.root, 'A4C', filename)) as f:
            a4cdata = pd.read_csv(f, index_col=False)

        with open(os.path.join(self.root, 'PSAX', filename)) as f:
            psaxdata = pd.read_csv(f, index_col=False)

        if self.data_type == "A4C":
            data = a4cdata
        elif self.data_type == "PSAX":
            data = psaxdata
        else:
            data = pd.concat([a4cdata, psaxdata]).reset_index(drop=True)

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
                data = None
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

    def get_phase_labels(self, outputFilename="PhasesList.csv"):
        try:
            if os.path.exists(outputFilename):
                existing_data = pd.read_csv(outputFilename)
                for _, row in existing_data.iterrows():
                    self.phase_values[row['FileName']] = [pd.eval(row['ED_Predictions']), pd.eval(row['ES_Predictions'])]
        except:
            predictionFiles = ["UVT_M_REG.csv", "UVT_R_CLA.csv", "UVT_R_REG.csv", "UVT_M_CLA.csv", "EchoPhaseDetection.csv"]
            output_data = []

            for dataType in ["A4C", "PSAX"]:
                for filename in predictionFiles:
                    try:
                        with open(os.path.join(self.root, dataType, filename)) as f:
                            data = pd.read_csv(f, index_col=False, header=None)
                            # Go through each name in self.fnames, and find the corresponding row in data,
                            # then add the ED and ES predictions to self.phase_values if they are not empty lists
                            for name in self.fnames:
                                # if self.phase_values[name] has already been set, then skip
                                if name in self.phase_values:
                                    continue
                                rows = data[(data.iloc[:, 0] == name) | (data.iloc[:, 1] == name) |
                                            (data.iloc[:, 0] == name.rstrip('.avi')) | (data.iloc[:, 1] == name.rstrip('.avi'))]
                                if len(rows) == 0:
                                    # print(f"Warning: {name} not found in {filename}, skipping")
                                    continue
                                if self.createAllClips:
                                    # need another way of getting number of frames
                                    print("TODO: get number of frames")
                                    quit()
                                    number_of_frames = "TODO" #TODO
                                    length = self.length
                                    self.phase_values[name] = [list(range(
                                        0, number_of_frames - length + 1)), list(range(length, number_of_frames + 1))]
                                else:
                                    # UVT is filename, ED predictions, ES predictions
                                    if filename == "EchoPhaseDetection.csv":
                                        rowED = rows[rows.iloc[:, 2] == "ED"]
                                        rowES = rows[rows.iloc[:, 2] == "ES"]
                                        ED_Predictions = pd.eval(rowED.values[0][3])
                                        ES_Predictions = pd.eval(rowES.values[0][3])
                                    else:
                                        ED_Predictions = pd.eval(rows.values[0][1])
                                        ES_Predictions = pd.eval(rows.values[0][2])
                                    if len(ED_Predictions) == 0 or len(ES_Predictions) == 0:
                                        # print(f"Warning: {name} has no ED or ES predictions in {filename}, skipping")
                                        continue
                                    self.phase_values[name] = [ED_Predictions, ES_Predictions]
                                    output_data.append({"FileName": name, "ED_Predictions": ED_Predictions, "ES_Predictions": ES_Predictions})


                    except FileNotFoundError:
                        print(f"Warning: {filename} not found, skipping")

            # Write output data to a new file
            with open(outputFilename, "w") as output_file:
                output_file.write("FileName,ED_Predictions,ES_Predictions\n")
                for row in output_data:
                    output_file.write(f"{row['FileName']},{row['ED_Predictions']},{row['ES_Predictions']}\n")

        # check if any of the names in self.fnames are not in self.phase_values
        # if so, then add them to self.phase_values with empty lists
        missing_values = []
        for name in self.fnames:
            if name not in self.phase_values:
                self.phase_values[name] = [[], []]
                missing_values.append(name)  

        if len(missing_values) > 0:
            print("Missing phase information for {} pediatric videos.".format(
                len(missing_values)))
            self.generate_phase_predictions(missing_values)

            #remove the missing values from self.fnames #TODO remove once phase information is generated
            self.fnames = [name for name in self.fnames if name not in missing_values]

        self.check_missing_files()


    def generate_phase_predictions(self, filenames=None):
        # TODO generate phase information
        pass

    def check_missing_files(self):
        # Check that files are present in A4C and PSAX directories
        fnames_with_ext = [fname + (".avi" if not fname.endswith(".avi") else "") for fname in self.fnames]

        missing_a4c = set(fnames_with_ext) - set(os.listdir(os.path.join(self.root, "A4C", "Videos")))
        missing_psax = set(fnames_with_ext) - set(os.listdir(os.path.join(self.root, "PSAX", "Videos")))
        missing = missing_a4c.intersection(missing_psax)

        if len(missing) != 0:
            print("{} videos could not be found in {} or {}:".format(
                len(missing), os.path.join(self.root, "A4C", "Videos"), os.path.join(self.root, "PSAX", "Videos")))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, sorted(missing)[0]))


    def __getitem__(self, index):
        # check phase number
        # if fnames[index] contains _phase_ then use the number following the last _
        # otherwise use 0
        if "_phase_" in self.fnames[index]:
            phase = int(self.fnames[index].split("_")[-1])
            # now remove the _phase_# from the filename
            self.fnames[index] = "_".join(self.fnames[index].split("_")[:-1])
        else:
            phase = 0
    
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

        if self.use_phase_clips:
            if phase != 0:
                video = self.select_clip_phase(video, length, index, phase)
            else:
                video = self.select_clips_phase(video, length, index)
        else:
            video = self.select_clips(video, length)

        target = self.gather_targets(index)

        if self.pad is not None:
            video = self.pad_video(video)

        videos = helpFuncs.augment_video(video, self.num_augmented_videos, self.dropout_only, self.rotation_only, self.dropout_int, self.rotation_int)

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

    def select_clip_phase(self, video, length, index, phase, outputfilename="FileListPhaseClips.csv"):
        # if FileListPhaseClips exists, use it and return the clip
        # otherwise, call select_clips_phase and return the clip
        if os.path.exists(os.path.join(self.root, outputfilename)):
            with open(os.path.join(self.root, outputfilename)) as f:
                data = pd.read_csv(f)
        else:
            return self.select_clips_phase(video, length, index)
        
        # get the clip from the file
        # get the video name
        key = self.fnames[index]
        # if key ends with .avi, remove it
        if key.endswith(".avi"):
            key = key[:-4]

        # get the rows for this video where data[0] contains the key
        rows = data[data[0].str.contains(key)]
        # get the row for this phase
        row = rows[phase]
        # get the start and end frames
        start = int(row[1])
        end = int(row[2])

        # clip = video, from start to end, regardless of length
        clip = video[:, start:end, :, :]

        return clip
    
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

        for i in range(min(len(ED_Predictions), len(ES_Predictions), self.clips)):
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
            # print("Warning: No clips found for video {}".format(
            #     self.fnames[index]))
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
