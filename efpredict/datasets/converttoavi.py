import os

import pandas as pd
import numpy as np
import nibabel as nib

import cv2
import glob

import matplotlib.pyplot as plt

from PIL import Image as im


print(nib.__version__) 

patient_directories = os.listdir()
# print(patient_directories)
second_half_of_filename = "_4CH_half_sequence.nii"

pathview1 = 'view1/'
# pathview2 = 'view2/'
# pathview3 = 'view3/'

if not os.path.exists(pathview1):
    os.makedirs(pathview1)
# if not os.path.exists(pathview2):
#     os.makedirs(pathview2)
# if not os.path.exists(pathview3):
#     os.makedirs(pathview3)

def view(sizeview, patient_nii, patient, pathview, viewnum):
    print("View: ", viewnum, " Size: ", sizeview, " Patient: ", patient, " Path: ", pathview)
    frames = []

    for i in range(0, sizeview):
        if viewnum == 1:
            frame = patient_nii.dataobj[:,:,i]
        elif viewnum == 2:
            frame = patient_nii.dataobj[:,i,:]
        elif viewnum == 3:
            frame = patient_nii.dataobj[i,:,:]
        frames.append(frame)
    frames = np.array(frames)

    ## Convert each Frames to Images

    images = []
    output_folder = pathview + patient + '/'

    for i in range(sizeview):
        img = im.fromarray(frames[i])
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_name = str(i) + '.jpg' 
        # img.save(img_name)
        # save to view1/patient_name/
        
        output_path = output_folder + img_name
        img.save(output_path)

    ## Convert these images to video
    img_array = []

    i = 0
    for files in glob.glob(output_folder + '*.jpg'):
        # To maintain order
        filename = output_folder + str(i)+'.jpg'
        
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        i = i+1


    out = cv2.VideoWriter(output_folder + 'video_' + patient + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    ## delete all the images
    for files in glob.glob(output_folder + '*.jpg'):
        os.remove(files)
    
    #copy file Info_4CH.cfg from patient folder to output_folder without using os
    file_location = patient + '/' + 'Info_4CH.cfg'
    output_file_location = output_folder + 'Info_4CH.cfg'
    with open(file_location, 'r') as f:
        with open(output_file_location, 'w') as f1:
            for line in f:
                f1.write(line)
    



for patient in patient_directories:
    if not os.path.exists(pathview1 + patient):
        os.makedirs(pathview1 + patient)
    # if not os.path.exists(pathview2 + patient):
    #     os.makedirs(pathview2 + patient)
    # if not os.path.exists(pathview3 + patient):
    #     os.makedirs(pathview3 + patient)
    if patient == 'converttoavi.py':
        continue
    # print(patient)
    patient_path = patient + '/'

    filename = patient + second_half_of_filename
    try:
        patient_nii = nib.load(patient + '/' + filename)
    except:
        other_second_half_of_filename = "_4CH_half_sequence_gt.nii"
        temp_filename = patient + other_second_half_of_filename

        try:
            patient_nii = nib.load(patient + '/' + temp_filename)
        except:
            print("Error in loading file: ", filename)
            continue
    patient_data = patient_nii.get_fdata()
    # print(patient_data.shape)

    sizeview3, sizeview2, sizeview1 = patient_nii.shape
    # print(sizeview1, sizeview2, sizeview3)
    view(sizeview1, patient_nii, patient, pathview1, 1)
    # view(sizeview2, patient_nii, patient, pathview2, 2)
    # view(sizeview3, patient_nii, patient, pathview3, 3)



    



