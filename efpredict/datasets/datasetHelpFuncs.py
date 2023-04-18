import random

import numpy as np
from scipy.ndimage import rotate


def augment_video(video, num_augmented_videos=0, dropout_only=False, rotation_only=False, dropout_int=None, rotation_int=None):
    videos = [video]
    if num_augmented_videos == 0:
        return videos
    
    # Combine rotation and pixel dropout augmentations
    dropout_percentages = [0.01, 0.02, 0.03], [0.02, 0.03, 0.05], [0.03, 0.05, 0.08], [0.05, 0.08, 0.13]
    # if droupout_percentages is an int between 0 and 3 inclusive, use the corresponding list of dropout percentages
    if isinstance(dropout_int, int) and 0 <= dropout_int <= 3:
        dropout_percentages = dropout_percentages[dropout_int]
    else:
        dropout_percentages = dropout_percentages[0]

    rotation_angles = [[1, 2, 3, -1, -2, -3], [2, 5, 8, -2, -5, -8], [5, 10, 15, -5, -10, -15]]
    if isinstance(rotation_int, int) and 0 <= rotation_int <= 2:
        rotation_angles = rotation_angles[rotation_int]
    else:
        rotation_angles = rotation_angles[0]

    if dropout_only:
        augmentations = [('pixel_dropout', percentage) for percentage in dropout_percentages]
    elif rotation_only:
        augmentations = [('rotate', angle) for angle in rotation_angles]
    else:
        augmentations = [('rotate', angle) for angle in rotation_angles] + [('pixel_dropout', percentage) for percentage in dropout_percentages]

    # Limit the number of augmentations to the number of augmentations available
    num_augmented_videos = min(num_augmented_videos, len(augmentations))

    selected_augmentations = random.sample(augmentations, num_augmented_videos)
    
    for aug_type, aug_param in selected_augmentations:
        if aug_type == 'rotate':
            rotation_angle = aug_param
            augmented_video = np.zeros_like(video)
            for i in range(video.shape[0]):
                for j in range(video.shape[1]):
                    augmented_video[i, j] = rotate(video[i, j], rotation_angle, reshape=False)
        elif aug_type == 'pixel_dropout':
            dropout_prob = aug_param
            mask = np.random.random(video.shape) > dropout_prob
            augmented_video = video * mask
            
        videos.append(augmented_video)
    
    return videos


#Doesn't work due to echocardiograms being a strict orientation
def augment_video_old(self, video):
    videos = [video]
    if self.num_augmented_videos == 0:
        return videos
    
    # Combine flipping and rotation augmentations
    augmentations = [('rotate', angle) for angle in [90, 180, 270]] + [('flip', flip_type) for flip_type in ['horizontal', 'vertical', 'both']]

    # Limit the number of augmentations to the number of augmentations available
    num_augmented_videos = min(self.num_augmented_videos, len(augmentations))

    selected_augmentations = random.sample(augmentations, num_augmented_videos)
    
    for aug_type, aug_param in selected_augmentations:
        if aug_type == 'rotate':
            rotation_angle = aug_param
            augmented_video = np.rot90(video, k=rotation_angle // 90, axes=(2, 3)).copy()
        elif aug_type == 'flip':
            flip_type = aug_param
            if flip_type == 'horizontal':
                augmented_video = np.flip(video, axis=3).copy()
            elif flip_type == 'vertical':
                augmented_video = np.flip(video, axis=2).copy()
            elif flip_type == 'both':
                augmented_video = np.flip(np.flip(video, axis=2), axis=3).copy()
        videos.append(augmented_video)
    
    return videos